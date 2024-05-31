
import torch.nn as nn

from torch.utils.data import DataLoader

import torch
import torch.optim as optim

from tqdm import tqdm

import wandb

from netformer import *
from reverse_converter import RConverter
from dataset import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    wandb.login()

    d_model = 1024
    num_heads = 4
    num_layers = 4
    d_ff = 1024
    dropout = 0
    batchsize = 20
    lr = 4e-5
    num_train_nets = 5000
    num_val_nets = 1000
    pretrained = 'weights.pt'

    run = wandb.init(
        project="netformer",
        name="1024, diff loss",
        config={
            "learning_rate": lr,
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "d_ff": d_ff,
            "dropout": dropout,
            "batchsize": batchsize,
            "data": "random",
            "num_train_nets": num_train_nets,
            "num_val_nets": num_val_nets,
            "pretrained": pretrained
        },
    )

    train_dataset = NetDataset(num_train_nets)
    valid_dataset = NetDataset(num_val_nets)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batchsize, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=batchsize, shuffle=False)

    transformer = NetFormer(d_model, num_heads, num_layers, d_ff, dropout)
    # transformer.load_state_dict(torch.load("/kaggle/input/netformer-weights/netformer_weights (2).pth"))
    transformer.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    transformer.train();


    best_val_loss = 999
    for epoch in range(30):
        train_loss = 0
        train_kl = 0
        transformer.train();
        for i, batch in enumerate(tqdm(train_dataloader)):
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            output = transformer(data, data)
            loss = criterion(output[:, :, :2], labels[:, :, :2]) * 1.5 + criterion(output[:, :, 2:], labels[:, :,
                                                                                                     2:]) + transformer.encoder.kl * 0.01
            loss.backward()
            train_loss += loss.item()
            train_kl += transformer.encoder.kl * 0.01
            optimizer.step()
        train_loss /= len(train_dataloader)
        train_kl /= len(train_dataloader)
        print("train: epoch:", epoch, "loss:", round(train_loss, 4))

        transformer.eval();
        val_loss = 0
        val_edges = 0
        for i, batch in enumerate(tqdm(valid_dataloader)):
            inp, out = batch
            inp, out = inp.to(device), out.to(device)
            with torch.no_grad():
                infered_tokens = transformer(inp, inp)
            loss = criterion(infered_tokens, out) + transformer.encoder.kl * 0.01
            val_loss += loss.item()

            for j in range(batchsize):

                rounded_tokens = RConverter.get_round_tokens(infered_tokens[j].unsqueeze(0))
                skip_minus_tokens = RConverter.skip_minus_tokens(rounded_tokens)
                try:
                    generated_matrix = RConverter.get_block_matrix(skip_minus_tokens)
                except:
                    generated_matrix = RConverter.get_block_matrix(torch.tensor([[1, 1, 0, 0]]))
                matrix = RConverter.get_block_matrix(inp[j])
                count_matrix = RConverter.get_count_matrix(matrix)
                writed_matrix, edges_num = RConverter.write_generated(generated_matrix, matrix)
                net = RConverter.get_network(writed_matrix, count_matrix)
                val_edges += edges_num / count_parameters(net)
        val_loss /= len(valid_dataloader)
        val_edges = val_edges / len(valid_dataloader) / batchsize
        print("valid: edges:", round(val_edges, 4), "loss:", round(val_loss, 4))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(transformer.state_dict(), f'eopch_{epoch}.pth')

        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_edges": val_edges,
                "train_kl": train_kl
            })

    transformer.eval()
    run.finish()
