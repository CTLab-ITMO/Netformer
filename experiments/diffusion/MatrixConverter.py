import torch
from torch import nn

act_list = [nn.ReLU, nn.Tanh, nn.Sigmoid]

act_map = {layer: (i + 1) / len(act_list) for i, layer in enumerate(act_list)}


def matrix_converter(net):
    matrix = torch.zeros(64, 64) + torch.full((64, 64), 0)

    dims = [0, 0]
    if net.__class__ != nn.Sequential:
        net = net.layers
    for i in range(len(net)):
        layer = net[i]
        if layer.__class__ in act_map.keys():
            matrix[dims[0], dims[1]] = act_map[layer.__class__]
            dims[0] += 1
            dims[1] += 1
        else:
            matrix[dims[0]:dims[0] + layer.in_features, dims[1]:dims[1] + layer.out_features] = layer.weight.T
            dims[0] += layer.in_features
            dims[1] += layer.out_features
    return matrix.unsqueeze(0).detach()
