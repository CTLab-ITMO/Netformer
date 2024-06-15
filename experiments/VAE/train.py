import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.insert(0, project_root)


from lib.netformer.dataset import Net, act 
from lib.netformer.reverse_converter import RConverter
from converter import matrix_converter
from model import CNNVAE
from configs import *
from utils import *
from dataset import NetDataset, collate_fn, RegDataset

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import os
device = 'cuda' if torch.cuda.is_available() else "cpu"


regs = pd.read_csv(REG_PATH)
uniq_regs = regs["reg_id"].unique()
reg_data = [regs[regs["reg_id"] == i] for i in uniq_regs]

reg_datasets = []
reg_dataloaders = []
n_regs = 4
for reg in reg_data[:n_regs]:
    train_data, valid_data = train_test_split(reg, test_size=0.1, random_state = 42)
    train_dataset = RegDataset(train_data)
    valid_dataset = RegDataset(valid_data)
    reg_datasets += [[train_dataset, valid_dataset]]
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)
    reg_dataloaders += [[train_loader, valid_loader]]


train_dataset = NetDataset(0, 120, reg_data[:n_regs])
valid_dataset = NetDataset(120, 156, reg_data[:n_regs])

BATCHSIZE = 36
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=BATCHSIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=BATCHSIZE, shuffle=False)


cnnvae = CNNVAE()
cnnvae.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(cnnvae.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
cnnvae.train();


run = wandb.init(
    project="netformer",
    name="1024, diff loss",
    config={
        "learning_rate": lr,
        "d_model": 0,
        "num_heads": 0,
        "num_layers": 0,
        "d_ff": 0,
        "dropout": 0,
        "BATCHSIZE": BATCHSIZE,
        "data": "random",
        "edge_c": 0
    },
)

lrs = []
edge_c, vert_c = 0, 1
for epoch in range(50):
    cnnvae.train()
    train_loss, train_loss_vert, train_loss_edg, train_kl = 0, 0, 0, 0
    for i, batch in enumerate(tqdm(train_dataloader)):
        summ_diff = 0

        data = batch[0]
        data = data.to(device)
        optimizer.zero_grad()
        output = cnnvae(data)
        loss = criterion(data, output)
        print(cnnvae.cnn_encoder.kl * KL, loss)
        loss += cnnvae.cnn_encoder.kl * KL

        loss.backward(retain_graph=True)
        optimizer.step()

        train_loss += loss.item()
        train_kl += cnnvae.cnn_encoder.kl

    train_loss /= len(train_dataloader)
    print("train: epoch:", epoch, "loss:", round(train_loss, 4), "kl:", train_kl)

    cnnvae.eval()
    val_loss = 0
    val_metrics = {'val_loss': 0, 'val_loss_vert': 0, 'val_loss_edg': 0,
                   'val_kl': 0, 'val_edges': 0, 'mse':0,'mape':0,'cos_sim':0, 'edge_metric': 0}

    for i, batch in enumerate(tqdm(valid_dataloader)):
        data, mf, reg = batch
        with torch.no_grad():
            output = cnnvae(data.to(device))
        val_loss = criterion(data.to(device), output) + cnnvae.cnn_encoder.kl * KL
        for j in range(BATCHSIZE):
            count_matrix = RConverter.get_count_matrix(data[j][0])
            true_net = RConverter.get_network(data[j][0], count_matrix)
            writed_matrix, edges_num = RConverter.write_generated(output[j][0].squeeze(0), data[j][0])
            net = RConverter.get_network(writed_matrix, count_matrix)
            val_metrics['val_loss'] += val_loss.item()

            net.to(device)
            true_net.to(device)
            true_model_result = (valid_model(net, reg_dataloaders[int(reg[j])][1]))
            gen_model_result = (valid_model(true_net, reg_dataloaders[int(reg[j])][1]))
            val_metrics['val_edges'] += edges_num / count_parameters(net)

            val_cos_sim = valid_cos_sim(net, true_net, reg_dataloaders[int(reg[j])][1])
            val_metrics['cos_sim'] += val_cos_sim
            val_metrics['mse'] += (true_model_result - gen_model_result)**2
            val_metrics['mape'] += (true_model_result - gen_model_result).abs()/gen_model_result
            val_metrics['edge_metric'] += edge_metric(data[j][0], output[j][0])

        val_metrics['val_kl'] += cnnvae.cnn_encoder.kl

    val_metrics['val_edges'] /= (len(valid_dataloader) * BATCHSIZE)
    val_metrics['val_loss'] /= (len(valid_dataloader) * BATCHSIZE)
    val_metrics['val_loss_vert'] /= (len(valid_dataloader) * BATCHSIZE)
    val_metrics['val_loss_edg'] /= (len(valid_dataloader) * BATCHSIZE)
    val_metrics['val_kl'] /= len(valid_dataloader)
    val_metrics['cos_sim'] /= (len(valid_dataloader) * BATCHSIZE)
    val_metrics['mse'] /= (len(valid_dataloader) * BATCHSIZE)
    val_metrics['mape'] /= (len(valid_dataloader) * BATCHSIZE)
    val_metrics['edge_metric'] /= (len(valid_dataloader) * BATCHSIZE)

    print("valid: edges:", round(val_metrics['val_edges'], 4), "loss:", round(val_metrics['val_loss'], 4))

    wandb.log(val_metrics |
            {"train_loss": train_loss,
             "train_loss_vert": train_loss_vert,
             "train_loss_edg": train_loss_edg,
             "train_kl": train_kl})
