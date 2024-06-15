import math
import torch
from torch import nn


def sinusoidal_embedding(vocab_len, inner_dim):
    embeddings = []
    for vocab_pos in range(vocab_len):
        cur_embed = []
        for embed_ind in range(inner_dim):
            val = vocab_pos / 10000 ** (2 * embed_ind / inner_dim)
            if embed_ind % 2 == 0:
                val = math.sin(val)
            else:
                val = math.cos(val)
            cur_embed.append(val)
        embeddings.append(torch.tensor(cur_embed))
    embeddings = torch.stack(embeddings)
    return embeddings


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, time_steps_number, inner_dim):
        super().__init__()
        self.time_to_embed = nn.Embedding(time_steps_number, inner_dim)
        self.time_to_embed.weight.data = sinusoidal_embedding(time_steps_number, inner_dim)
        self.time_to_embed.requires_grad_(False)

    def forward(self, time):
        return self.time_to_embed(time)


class UNetBlock(nn.Module):
    def __init__(self, shape, in_ch, out_ch, normalize=True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(shape) if normalize else nn.Identity(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.layers(x)