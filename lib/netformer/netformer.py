import torch
import torch.nn as nn
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model  # Model's dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = d_model // num_heads  # Dimension of each head's key, query, and value

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model)  # Query transformation
        self.W_k = nn.Linear(d_model, d_model)  # Key transformation
        self.W_v = nn.Linear(d_model, d_model)  # Value transformation
        self.W_o = nn.Linear(d_model, d_model)  # Output transformation

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            # print(attn_scores.shape, mask.shape)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output



class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))



class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class FullEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout):
        super(FullEncoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.emb_token = nn.Parameter(torch.normal(0, 1, size=(1, d_model)))  # add embedding token

        self.fc_mu = nn.Linear(d_model, d_model - 26)
        self.fc_sigma = nn.Linear(d_model, d_model - 26)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

        self.ln1 = nn.Linear(4, d_model)

    def forward(self, src):
        src = self.ln1(src)
        enc_output = torch.cat([self.emb_token.repeat(src.shape[0], 1, 1), src], dim=1)
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output)
        enc_shape = enc_output.shape[0]

        enc_output = enc_output[0, 0]
        mu = self.fc_mu(enc_output)
        sigma = torch.exp(self.fc_sigma(enc_output))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()  # for loss
        z = z.repeat(1, enc_shape, 1)  # add embedding token

        return z


class FullDecoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout):
        super(FullDecoder, self).__init__()
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.ln1 = nn.Linear(4, d_model)
        self.ln2 = nn.Linear(d_model, 4)

    def forward(self, z, tgt, meta_features, tgt_mask):
        #         print(meta_features.unsqueeze(0).shape, z.shape)
        #         print(meta_features.tile((1, 20, meta_features.shape[-1])).shape)
        z = torch.cat([z, meta_features.unsqueeze(0)], dim=-1)
        #         print(z.shape)
        tgt = self.ln1(tgt)
        dec_output = tgt
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, z, tgt_mask)
        dec_output = self.ln2(dec_output)
        return dec_output


class NetFormer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout):
        super(NetFormer, self).__init__()
        self.encoder = FullEncoder(d_model, num_heads, num_layers, d_ff, dropout)
        self.decoder = FullDecoder(d_model, num_heads, num_layers, d_ff, dropout)

    def generate_mask(self, tgt, is_inference):
        tgt_mask = (tgt[:, :, 0] != 1e-9).unsqueeze(1).unsqueeze(3)
        nopeak_mask = (1 - torch.triu(torch.ones(1, tgt.size(1), tgt.size(1)), diagonal=1)).bool().to(device)
        tgt_mask = tgt_mask & nopeak_mask
        return tgt_mask

    def encode(self, src):
        return self.encoder(src)

    def decode(self, latent, tgt, mfs, tgt_mask):
        return self.decoder(latent, tgt, mfs, tgt_mask)

    def forward(self, src, tgt, mfs, is_inference=False):
        self.tgt_mask = self.generate_mask(tgt, is_inference)
        self.latent = self.encode(src)
        return self.decode(self.latent, tgt, mfs, self.tgt_mask)