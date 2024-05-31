import torch
import torch.nn as nn


class Converter:
    # todo: delete this var (can't encode some nns)
    def __init__(self):
        self.lower_bound = 0
        self.upper_bound = 0

    @torch.no_grad()
    def convert_linear(self, weights):
        rows, cols = weights.T.shape
        self.upper_bound += rows
        parents = torch.arange(self.lower_bound, self.lower_bound+rows)
        children = torch.arange(self.upper_bound, self.upper_bound+cols)
        inds = torch.meshgrid(parents, children)
        self.lower_bound += rows

        return torch.stack([inds[0], inds[1], weights.T, torch.full((rows, cols), 0)]).T.reshape(-1, 4)

    @torch.no_grad()
    def convert_act(self, type, last_linear_num):
        inds = torch.arange(self.lower_bound, self.lower_bound + last_linear_num)
        if last_linear_num == 1:
            return torch.stack([inds, inds, torch.tensor([type]), torch.tensor([1])]).T
        return torch.stack([inds, inds, torch.full((last_linear_num, 1), type).squeeze(), torch.full((last_linear_num, 1), 1).squeeze()]).T

    @torch.no_grad()
    def convert_act_new(self, type, last_linear_num):
        rows, cols = last_linear_num, last_linear_num
        self.upper_bound += rows
        parents = torch.arange(self.lower_bound, self.lower_bound+rows)
        children = torch.arange(self.upper_bound, self.upper_bound+cols)
        self.lower_bound += rows

        return torch.stack([parents.unsqueeze(-1), children.unsqueeze(-1), torch.full((rows, 1), 1), torch.full((rows, 1), type)]).T.reshape(-1, 4)

    @torch.no_grad()
    def convert_pool(self, type, last_linear_num):
        inds = torch.arange(self.lower_bound, self.lower_bound + last_linear_num)
        return torch.stack([inds, inds, torch.full((last_linear_num, 1), 0).squeeze(), torch.full((last_linear_num, 1), 2).squeeze()]).T

    @torch.no_grad()
    def convert(self, net):
        all_weights = [torch.tensor([[-1, -1, -1, -1]])]
        for i in range(len(net)):
            if net[i].__class__ is nn.Linear:
                all_weights += [self.convert_linear(net[i].weight)]
            if net[i].__class__ is nn.Sigmoid:
                all_weights += [self.convert_act_new(1, net[i-1].weight.shape[0])]
            if net[i].__class__ is nn.ReLU:
                all_weights += [self.convert_act_new(2, net[i-1].weight.shape[0])]
            if net[i].__class__ is nn.Tanh:
                all_weights += [self.convert_act_new(3, net[i-1].weight.shape[0])]
            if net[i].__class__ is nn.MaxPool2d:
                all_weights += [self.convert_act_new(0, net[i-1].weight.shape[0])]
        return torch.cat(all_weights, dim=0)


