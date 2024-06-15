import torch
import torch.nn as nn


act_map = {nn.ReLU: 2,
           nn.Tanh: 3,
           nn.Sigmoid: 1}

def matrix_converter(net):

    matrix = torch.zeros(64, 64) + torch.full((64, 64), 0)
    count_matrix = torch.zeros(64, 64)
    dims = [0, 0]
    last_layer = 0
    layer_count = 1

    if net.__class__ != nn.Sequential:
        net = net.layers
    for i in range(len(net)):
        layer = net[i]
        if layer.__class__ in act_map.keys():
            for _ in range(last_layer):
                matrix[dims[0], dims[1]] = act_map[layer.__class__]
                count_matrix[dims[0], dims[1]] = layer_count
                dims[0] += 1
                dims[1] += 1
            layer_count += 1
        else:
            matrix[dims[0]:dims[0]+layer.out_features, dims[1]:dims[1]+layer.in_features] = layer.weight
            count_matrix[dims[0]:dims[0]+layer.out_features, dims[1]:dims[1]+layer.in_features] = layer_count * torch.eye(layer.out_features, layer.in_features)
            dims[0] += layer.out_features
            dims[1] += layer.in_features
            last_layer = layer.out_features
            layer_count += 1
    return matrix, count_matrix
