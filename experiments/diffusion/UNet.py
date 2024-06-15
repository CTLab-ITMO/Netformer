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


class MyUNet(nn.Module):
    def __init__(self, time_steps_number, time_emb_dim):
        super(MyUNet, self).__init__()
        self.time_embed = SinusoidalPositionEmbeddings(time_steps_number, time_emb_dim)

        # First half
        self.te1 = self.make_te_layer(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            UNetBlock((64, 64), 1, 10),
            UNetBlock((64, 64), 10, 10),
            UNetBlock((64, 64), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)  # 64x64 -> 32x32

        self.te2 = self.make_te_layer(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            UNetBlock((32, 32), 10, 20),
            UNetBlock((32, 32), 20, 20),
            UNetBlock((32, 32), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)  # 32x32 -> 16x16

        self.te3 = self.make_te_layer(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            UNetBlock((16, 16), 20, 40),
            UNetBlock((16, 16), 40, 40),
            UNetBlock((16, 16), 40, 40)
        )
        self.down3 = nn.Conv2d(40, 40, 4, 2, 1)  # 16x16 -> 8x8

        self.te4 = self.make_te_layer(time_emb_dim, 40)
        self.b4 = nn.Sequential(
            UNetBlock((8, 8), 40, 80),
            UNetBlock((8, 8), 80, 80),
            UNetBlock((8, 8), 80, 80)
        )
        self.down4 = nn.Conv2d(80, 80, 4, 2, 1)  # 8x8 -> 4x4

        # Bottleneck
        self.te_mid = self.make_te_layer(time_emb_dim, 80)
        self.b_mid = nn.Sequential(
            UNetBlock((4, 4), 80, 80),
            UNetBlock((4, 4), 80, 80),
            UNetBlock((4, 4), 80, 80)
        )

        # Second half
        self.up1 = nn.ConvTranspose2d(80, 80, 4, 2, 1)  # 4x4 -> 8x8
        self.te5 = self.make_te_layer(time_emb_dim, 160)
        self.b5 = nn.Sequential(
            UNetBlock((8, 8), 160, 80),
            UNetBlock((8, 8), 80, 40),
            UNetBlock((8, 8), 40, 40)
        )

        self.up2 = nn.ConvTranspose2d(40, 40, 4, 2, 1)  # 8x8 -> 16x16
        self.te6 = self.make_te_layer(time_emb_dim, 80)
        self.b6 = nn.Sequential(
            UNetBlock((16, 16), 80, 40),
            UNetBlock((16, 16), 40, 20),
            UNetBlock((16, 16), 20, 20)
        )

        self.up3 = nn.ConvTranspose2d(20, 20, 4, 2, 1)  # 16x16 -> 32x32
        self.te7 = self.make_te_layer(time_emb_dim, 40)
        self.b7 = nn.Sequential(
            UNetBlock((32, 32), 40, 20),
            UNetBlock((32, 32), 20, 10),
            UNetBlock((32, 32), 10, 10)
        )

        self.up4 = nn.ConvTranspose2d(10, 10, 4, 2, 1)  # 32x32 -> 64x64
        self.te_out = self.make_te_layer(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            UNetBlock((64, 64), 20, 10),
            UNetBlock((64, 64), 10, 10),
            UNetBlock((64, 64), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x, t_e):
        t_e = self.time_embed(t_e)
        n = len(x)
        out1 = self.b1(x + self.te1(t_e).reshape(n, -1, 1, 1))  # (N, 10, 64, 64)
        out2 = self.b2(self.down1(out1) + self.te2(t_e).reshape(n, -1, 1, 1))  # (N, 20, 32, 32)
        out3 = self.b3(self.down2(out2) + self.te3(t_e).reshape(n, -1, 1, 1))  # (N, 40, 16, 16)
        out4 = self.b4(self.down3(out3) + self.te4(t_e).reshape(n, -1, 1, 1))  # (N, 80, 8, 8)

        return out

    def make_te_layer(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )