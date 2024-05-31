
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from lib.netformer.converter import Converter
import sys
import os

from random import randint, choice

from lib.netformer.metafeatures import get_meta_features

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

act = {
    "ReLU": {
        "layer": nn.ReLU,
        "args": {},
    },
    "Tanh": {
        "layer": nn.Tanh,
        "args": {},
    },
    "Sigmoid": {
        "layer": nn.Sigmoid,
        "args": {},
    }
}


class Net(nn.Module):
    def __init__(self, input_size, output_size, act):
        super(Net, self).__init__()

        self.layers = nn.Sequential()
        self.current_size = input_size
        self.count_layers = randint(1, 3)
        self.act = act

        for _ in range(self.count_layers):
            self.layers.append(self.make_linear())
            # if random() < .2: self.layers.append(self.make_normalization_layer())
            self.layers.append(self.make_activation_layer())

        self.layers.append(self.make_linear(output_size))

    def forward(self, x):
        return self.layers(x)

    def make_linear(self, output_size=None) -> nn.Linear:
        input_size = self.current_size

        if output_size is None:
            output_size = randint(4, 8)

        self.current_size = output_size

        return nn.Linear(in_features=input_size,
                         out_features=output_size,
                         bias=True)

    def make_activation_layer(self) -> nn.Module:
        layer_info = choice(list(self.act.values()))

        args = layer_info["args"]
        eval_args = {}
        for key, value in args.items():
            eval_args[key] = value() if callable(value) else value

        return layer_info["layer"](**eval_args)

    def make_normalization_layer(self) -> nn.Module:
        return choice([
            nn.BatchNorm1d(self.current_size),
            nn.LayerNorm(self.current_size),
        ])


class NetDataset(Dataset):
    def __init__(self, start, end, regs, working_dir):
        self.mfs = []
        self.nets = []
        self.regs = []
        for i in tqdm(range(len(regs))):
            meta_features = get_meta_features(regs[i].drop(['reg_id', 'y'], axis=1).to_numpy(),
                                              regs[i]['y'].to_numpy())
            for j in range(start, end):
                converter = Converter()  # переделать Converter.convert
                net = torch.load(f"{working_dir}/reg{i}/model{j}.pt", map_location=torch.device('cpu'))
                self.nets += [converter.convert(net.layers)]
                self.mfs += [meta_features]
                self.regs += [i]

    def __len__(self):
        return len(self.nets)

    def __getitem__(self, idx):
        return self.nets[idx], self.mfs[idx], self.regs[idx]


def collate_fn(data):
    seq_size = 350
    batch_size = 20
    labels = torch.empty(batch_size, seq_size, 4)
    inp = torch.empty(batch_size, seq_size, 4)
    mfs = torch.empty(batch_size, 26)
    reg = torch.empty(batch_size, 1)

    for i, elem in enumerate(data):
        labels[i] = torch.cat([elem[0][1:], torch.full((seq_size - elem[0].shape[0] + 1, 4), 0.)])
        inp[i] = torch.cat([elem[0], torch.tile(elem[0][-1], (seq_size - elem[0].shape[0], 1))])
        mfs[i] = torch.tensor(elem[1])
        reg[i] = torch.tensor(elem[2])
    return inp, labels, mfs, reg
