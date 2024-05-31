import torch.nn as nn
import torch


class RConverter:
    def __init__(self):
        pass

    @staticmethod
    def get_round_tokens(tokens):
        celled_out = torch.cat([torch.round(tokens[:, :, :2]), tokens[:, :, 2:3], torch.round(tokens[:, :, 3:4])],
                               dim=2)
        return celled_out.squeeze(0)

    @staticmethod
    def skip_minus_tokens(tokens):
        return tokens[(tokens[:, 0] >= 0) & (tokens[:, 1] >= 0)]

    @staticmethod
    def get_block_matrix(tokens):
        shape1, shape2 = int(tokens[:, 0].max()), int(tokens[:, 1].max())
        matrix_all_params = torch.zeros(shape2 + 1, shape1 + 1)
        for param in tokens[1:]:
            if param[-1] == 0:
                matrix_all_params[int(param[1]), int(param[0])] = param[2]
            if param[-1] == 1 or param[-1] == 2 or param[-1] == 3:
                matrix_all_params[int(param[1]), int(param[0])] = param[2]

        return matrix_all_params

    @staticmethod
    def get_count_matrix(block_matrix):
        params_len = int(torch.nonzero(block_matrix)[:, 1].unique().shape[0])
        another_matrix = [[] for _ in range(params_len)]
        for i in torch.nonzero(block_matrix):
            another_matrix[i[1]] += [int(i[0])]

        return another_matrix


    @staticmethod
    def write_generated(generated_matrix, true_matrix):
        return_matrix = true_matrix.clone()
        counter = 0
        for i, j in generated_matrix.nonzero():
            if i < true_matrix.shape[0] and j < true_matrix.shape[1]:
                if true_matrix[i, j] != torch.tensor([1.]) and true_matrix[i, j] != torch.tensor([0.]):
                    counter += 1
                    return_matrix[i, j] = generated_matrix[i, j]

        true_matrix[true_matrix == 1] = 0
        return_matrix[return_matrix == true_matrix] = 0.
        return return_matrix, counter

    @staticmethod
    def get_network(block_matrix, count_matrix):
        i = 0
        pointer = 0
        count = 0
        is_last_act = False
        layers = []
        while i < len(count_matrix):
            nonzeros = torch.nonzero(block_matrix[count_matrix[i]])
            if (nonzeros == torch.empty(nonzeros.shape)).all():
                count = count_matrix.count(count_matrix[i])
                is_last_act = False
                linear_weights = block_matrix[count_matrix[i]][:, i: i + count]
                layers.append(MyLinear(linear_weights))
            elif block_matrix[count_matrix[i]][0, nonzeros[0, 1]] == torch.full((1, 1), 1.).squeeze(
                    0) and not is_last_act:
                count = len(count_matrix[i - count])
                is_last_act = True
                layers.append(nn.Sigmoid())
            elif block_matrix[count_matrix[i]][0, nonzeros[0, 1]] == torch.full((1, 1), 2.).squeeze(
                    0) and not is_last_act:
                count = len(count_matrix[i - count])
                is_last_act = True
                layers.append(nn.ReLU())
            elif block_matrix[count_matrix[i]][0, nonzeros[0, 1]] == torch.full((1, 1), 3.).squeeze(
                    0) and not is_last_act:
                count = len(count_matrix[i - count])
                is_last_act = True
                layers.append(nn.Tanh())
            else:
                count = count_matrix.count(count_matrix[i])
                is_last_act = False
                linear_weights = block_matrix[count_matrix[i]][:, i: i + count]
                layers.append(MyLinear(linear_weights))
            i += count
        return RNet(layers)



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
