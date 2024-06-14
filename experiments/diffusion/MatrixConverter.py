import torch
from torch import nn

class RNet(nn.Module):
    def __init__(self, in_layers):
        super().__init__()
        self.layers = nn.Sequential()
        for i in in_layers:
            self.layers.append(i)
    def forward(self, x):
        return self.layers(x)

class MyLinear(nn.Module):
    def __init__(self, weights):
        super(MyLinear, self).__init__()

        self.linear = nn.Linear(weights.shape[1], weights.shape[0])
        with torch.no_grad():
            self.linear.weight.copy_(weights)

    def forward(self, x):
        x = self.linear(x)
        return x

class RConverter:
    def __init__(self):
        pass

