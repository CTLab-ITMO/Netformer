import torch.nn as nn
import pandas as pd
import torch
import os
import numpy as np
from copy import deepcopy as dc

from lib.generator.regression_dataset.dataset import get_dataloaders_and_datasets
from lib.netformer.dataset import Net, NetDataset, collate_fn
from lib.netformer.smart_loss import valid_model

from random import shuffle

class Saver:
    def __init__(self, dir="", columns=None):
        self.dir = os.path.join("", dir)
        os.makedirs(self.dir, exist_ok=True)

        if columns is None:
            columns = ["struct", "dataset_id", "metric"]
        self.columns = columns

        self.data = pd.DataFrame(columns=columns)

    def add(self, model, dataset_id, result):
        name_s = os.path.join(self.dir, f"model{len(self.data)}.pt")

        torch.save(model, name_s)

        self.data.loc[len(self.data)] = [name_s, dataset_id, result if type(result) is int else result.cpu().numpy()]

    def save(self, name="data.csv"):
        name = os.path.join(self.dir, name)
        self.data.to_csv(name, index=False)

    def load(self, name="data.csv"):
        name = os.path.join(self.dir, name)
        self.data = pd.read_csv(name)

    def clear(self):
        self.data = pd.DataFrame(columns=self.columns)
        self.dir = ""


def train_one_model(model, criterion, lr, train_loader, valid_loader, epochs=5000):
    params_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # print(f"{model=}, {params_count=}, {epochs=}")
    print(f"{params_count=}, {epochs=}")
    losses = []
    means = []

    for epoch in range(epochs):
        for _, data in enumerate(train_loader):
            x, y = data
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1), y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if epoch % 100 == 0:
            # print(f"MSE: {valid_model(model, valid_loader)}")
            print(f"Epoch: {epoch}, Loss: {np.mean(losses)}")

        if epoch % 50 == 0:
            means.append(np.mean(losses[-50:]))
        # if (len(means) > 2) and (abs(means[-1] - means[-2]) < 1):
        #     break

    return valid_model(model, valid_loader)





if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    duplicate = 0.2  # [0; 1)
    count_models = 100  # per regression
    min_epoch = 100
    max_epoch = 1000
    lr = 1e-2
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
    criterion = nn.MSELoss()

    data = pd.read_csv('data/random_regressions.csv')
    uniq_regs = data["reg_id"].unique()
    reg_data = [data[data["reg_id"] == i] for i in uniq_regs]
    col_names = [i for i in data.columns if i != "reg_id"]

    models = [Net(len(col_names) - 1, 1, act) for _ in range(int(count_models * (1 - duplicate)))]
    models += [dc(m) for m in models[:int(count_models * duplicate)]]
    shuffle(models)
    s = Saver("data/baseModels")
    for m in models:
        s.add(m, -1, -1)
    s.save()

    models = [torch.load(f"data/baseModels/model{i}.pt") for i in range(count_models)]
    len(models)

    from copy import deepcopy
    from random import randint, shuffle

    for i, data in enumerate(reg_data):
        saver = Saver(f"data/reg{i}")
        t_loader, v_loader = get_dataloaders_and_datasets(data)
        for j, model in enumerate(models):
            print('-------------------------------')
            print(f"Model: {j}/{len(models)}, reg: {i}/{len(reg_data)}")
            model.to(device)
            md = deepcopy(model)
            res = train_one_model(md, criterion, lr, t_loader, v_loader, randint(min_epoch, max_epoch))
            saver.add(md, i, res)
            saver.save()
