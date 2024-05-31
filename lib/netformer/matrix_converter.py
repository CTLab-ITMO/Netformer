import torch
from torch import nn

act_map = {nn.ReLU: 0,
           nn.Tanh: 1,
           nn.Sigmoid: 2}


def matrix_converter(net):
    dims = [0, 0]
    for i in range(len(net.layers)):
        layer = net.layers[i]
        if layer.__class__ in act_map.keys():
            dims[0] += 1
            dims[1] += 1
        else:
            dims[0] += layer.in_features
            dims[1] += layer.out_features
    matrix = torch.zeros(dims)

    dims = [0, 0]
    for i in range(len(net.layers)):
        layer = net.layers[i]
        if layer.__class__ in act_map.keys():
            matrix[dims[0], dims[1]] = act_map[layer.__class__]
            dims[0] += 1
            dims[1] += 1
        else:
            matrix[dims[0]:dims[0] + layer.in_features, dims[1]:dims[1] + layer.out_features] = layer.weight.T
            dims[0] += layer.in_features
            dims[1] += layer.out_features
    return matrix
