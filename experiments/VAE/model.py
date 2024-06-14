import torch
import torch.nn as nn


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv_block(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1):
        x1 = self.up(x1)
        return x1


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down0 = nn.Conv2d(2, 4, 3, padding=1)
        self.down1 = (Down(4, 8))
        self.down2 = (Down(8, 16))

        self.fc_sigma = nn.Linear(4096, 4096)
        self.fc_mu = nn.Linear(4096, 4096)

        self.N = torch.distributions.Normal(0, 1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0
    def forward(self, x):
        # print(x.shape)
        x1 = self.down0(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x_flatten = x3.reshape(x.shape[0], -1)
        # print(x3.shape)

        mu = self.fc_mu(x_flatten)
        sigma = torch.exp(self.fc_sigma(x_flatten))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z



class CNNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up3 = (Up(16, 8))
        self.up4 = (Up(8, 4))
        self.up5 = nn.Conv2d(4, 1, 3, padding=1)

    def forward(self, x):
        x3 = x.view((-1, 16, 16, 16))
        # print(x3.shape)
        x = self.up3(x3)
        # print(x.shape)
        x = self.up4(x)
        # print(x.shape)
        x = self.up5(x)
        # print(x.shape)
        return x


if __name__ == "__main__":
    cnn_encoder = CNNEncoder()
    cnn_decoder = CNNDecoder()
