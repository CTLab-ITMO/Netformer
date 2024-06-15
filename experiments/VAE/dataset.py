import torch
from torch.utils.data import Dataset
from converter import matrix_converter
from configs import *
from tqdm import tqdm
from utils import get_meta_features

class NetDataset(Dataset):
    def __init__(self, start, end, regs):
        self.mfs = []
        self.nets = []
        self.regs = []
        for i in tqdm(range(len(regs))):
            meta_features = get_meta_features(regs[i])
            for j in range(start, end):
                net = torch.load(f"/content/fitted 2/reg{i}/model{j}.pt", map_location=torch.device('cpu'))
                matrix, count_matrix = matrix_converter(net.layers)
                self.nets += [matrix_converter(net.layers)]
                self.mfs += [meta_features]
                self.regs += [i]

    def __len__(self):
        return len(self.nets)

    def __getitem__(self, idx):
        return self.nets[idx], self.mfs[idx], self.regs[idx]

def collate_fn(data):
    nets = torch.empty(BATCHSIZE, 2, 64, 64)
    mfs = torch.empty(BATCHSIZE, 26)
    reg = torch.empty(BATCHSIZE, 1)

    for i, elem in enumerate(data):
        nets[i] = torch.cat([elem[0][0].unsqueeze(0), elem[0][1].unsqueeze(0)])
        mfs[i] = torch.tensor(elem[1])
        reg[i] = torch.tensor(elem[2])
    return nets, mfs, reg
