import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader

from lib.netformer.dataset import NetDataset, collate_fn
from lib.netformer.netformer import NetFormer
from lib.netformer.reverse_converter import RConverter

d_model = 1024
num_heads = 4
num_layers = 4
d_ff = 1024
dropout = 0
BATCHSIZE = 20
lr = 4e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transformer = NetFormer(d_model, num_heads, num_layers, d_ff, dropout)
# transformer.load_state_dict(torch.load("/kaggle/input/netformer-weights/netformer_weights (2).pth"))
transformer.to(device)
# criterion = nn.CrossEntropyLoss(ignore_index=0)
criterion = nn.MSELoss()
# criterion = mixed_loss
optimizer = optim.Adam(transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
transformer.train()
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
        "BATCHSIZE": BATCHSIZE,
        "data": "random",
    },
)
train_dataset = NetDataset(5000)
valid_dataset = NetDataset(1000)
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=20, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=20, shuffle=False)
len(train_dataloader), len(valid_dataloader)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
best_val_loss = 999
for epoch in range(30):
    train_loss = 0
    train_kl = 0
    transformer.train()
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
    transformer.eval()
    val_loss = 0
    val_edges = 0
    for i, batch in enumerate(tqdm(valid_dataloader)):
        inp, out = batch
        inp, out = inp.to(device), out.to(device)
        with torch.no_grad():
            infered_tokens = transformer(inp, inp)
        loss = criterion(infered_tokens, out) + transformer.encoder.kl * 0.01
        val_loss += loss.item()
        for j in range(BATCHSIZE):
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
    val_edges = val_edges / len(valid_dataloader) / BATCHSIZE
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
torch.save(transformer.state_dict(), 'weights_1024.pth')

# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import wandb
# from tqdm import tqdm
# from torch.utils.data import DataLoader
#
# from lib.generator.generator import valid_model
# from lib.generator.regression_dataset.dataset import get_dataloaders_and_datasets
# from lib.netformer.smart_loss import mixed_loss, get_diff_sense, valid_cos_sim
# from lib.netformer.dataset import NetDataset, collate_fn
# from lib.netformer.netformer import NetFormer
# from lib.netformer.reverse_converter import RConverter
#
# d_model = 1024
# num_heads = 4
# num_layers = 4
# d_ff = 1024
# dropout = 0
# BATCHSIZE = 20
# lr = 4e-5
# edge_c = 1e-6
#
# ALL_REGS = 20
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# transformer = NetFormer(d_model, num_heads, num_layers, d_ff, dropout)
# # transformer.load_state_dict(torch.load("/kaggle/input/netformer-weights/netformer_weights (2).pth"))
# transformer.to(device)
#
# # criterion = nn.CrossEntropyLoss(ignore_index=0)
# criterion = nn.MSELoss()
# # criterion = mixed_loss
# optimizer = optim.Adam(transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
#
# transformer.train()
#
# run = wandb.init(
#     project="netformer",
#     name="1024, diff loss",
#     config={
#         "learning_rate": lr,
#         "d_model": d_model,
#         "num_heads": num_heads,
#         "num_layers": num_layers,
#         "d_ff": d_ff,
#         "dropout": dropout,
#         "BATCHSIZE": BATCHSIZE,
#         "data": "random",
#         "edge_c": edge_c
#     },
# )
#
# regs = pd.read_csv(r"lib/generator/data/random_regressions.csv")
# uniq_regs = regs["reg_id"].unique()
# reg_data = [regs[regs["reg_id"] == i] for i in uniq_regs]
#
# datasets_dir = "lib/generator/data"
#
#
# train_dataset = NetDataset(0, 80, reg_data, datasets_dir)
# valid_dataset = NetDataset(80, 100, reg_data, datasets_dir)
#
# train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=20, shuffle=True)
# valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=20, shuffle=False)
#
# reg_dataloaders, _ = get_dataloaders_and_datasets(reg_data)
#
#
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#
# vall_epoches = {f"reg{i}": dict(mse=0, mape=0) for i in range(ALL_REGS)}
# for epoch in range(30):
#     transformer.train()
#     train_loss = 0
#     train_kl = 0
#     for i, batch in enumerate(tqdm(train_dataloader)):
#         data, labels, mfs, regs_id = batch
#         data, labels, mfs = data.to(device), labels.to(device), mfs.to(device)
#         optimizer.zero_grad()
#         output = transformer(data, data, mfs)
#         summ_diff = 0
#         loss = 0
#         for j in range(BATCHSIZE):
#             rounded_tokens = RConverter.get_round_tokens(output[j].unsqueeze(0))
#             skip_minus_tokens = RConverter.skip_minus_tokens(rounded_tokens)
#             try:
#                 generated_matrix = RConverter.get_block_matrix(skip_minus_tokens)
#             except:
#                 generated_matrix = RConverter.get_block_matrix(torch.tensor([[1, 1, 0, 0]]))
#             matrix = RConverter.get_block_matrix(data[j])
#             count_matrix = RConverter.get_count_matrix(matrix)
#             true_net = RConverter.get_network(matrix, count_matrix)
#             writed_matrix = RConverter.write_generated(generated_matrix, matrix)[0]
#             net = RConverter.get_network(writed_matrix, count_matrix)
#             summ_diff = get_diff_sense(orig_model=true_net, gen_model=net,
#                                        valid_dataloader=reg_dataloaders[int(regs_id[j])][1])
#             loss += mixed_loss(output[j].unsqueeze(0), labels[j].unsqueeze(0), summ_diff, edge_c)[0]
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()
#
#     train_loss /= len(train_dataloader)
#     print("train: epoch:", epoch, "loss:", round(train_loss, 4))
#
#     transformer.eval();
#     val_loss = 0
#     val_edges = 0
#     vall_epoches = {f"reg{i}": dict(mse=0, mape=0, cos_sim=0) for i in range(ALL_REGS)}
#     for i, batch in enumerate(tqdm(valid_dataloader)):
#
#         data, labels, mfs, regs_id = batch
#         data, labels, mfs = data.to(device), labels.to(device), mfs.to(device)
#
#         with torch.no_grad():
#             infered_tokens = transformer(data, data, mfs)
#
#         loss = 0
#         for j in range(BATCHSIZE):
#             rounded_tokens = RConverter.get_round_tokens(infered_tokens[j].unsqueeze(0))
#             skip_minus_tokens = RConverter.skip_minus_tokens(rounded_tokens)
#             try:
#                 generated_matrix = RConverter.get_block_matrix(skip_minus_tokens)
#             except:
#                 generated_matrix = RConverter.get_block_matrix(torch.tensor([[1, 1, 0, 0]]))
#             matrix = RConverter.get_block_matrix(data[j])
#             count_matrix = RConverter.get_count_matrix(matrix)
#             true_net = RConverter.get_network(matrix, count_matrix)
#             writed_matrix, edges_num = RConverter.write_generated(generated_matrix, matrix)
#             net = RConverter.get_network(writed_matrix, count_matrix)
#
#             summ_diff = get_diff_sense(orig_model=true_net, gen_model=net,
#                                        valid_dataloader=reg_dataloaders[int(regs_id[j])][1])
#             loss += mixed_loss(infered_tokens[j].unsqueeze(0), labels[j].unsqueeze(0), summ_diff, edge_c)[0]
#
#             net.to(device)
#             true_net.to(device)
#             true_model_result = (valid_model(net, reg_dataloaders[int(regs_id[j])][1]))
#             gen_model_result = (valid_model(true_net, reg_dataloaders[int(regs_id[j])][1]))
#             val_edges += edges_num / count_parameters(net)
#
#             val_cos_sim = valid_cos_sim(net, true_net, reg_dataloaders[int(regs_id[j])][1])
#
#             vall_epoches[f"reg{int(regs_id[j])}"]["cos_sim"] += val_cos_sim
#             vall_epoches[f"reg{int(regs_id[j])}"]["mse"] += (true_model_result - gen_model_result) ** 2
#             vall_epoches[f"reg{int(regs_id[j])}"]["mape"] += (
#                                                                          true_model_result - gen_model_result).abs() / gen_model_result
#         val_loss += loss.item()
#     val_edges /= len(valid_dataloader) * BATCHSIZE
#     val_loss /= len(valid_dataloader) * BATCHSIZE
#     #     for j in range(ALL_REGS):
#     #         vall_epoches[f"reg{j}"]["mse"] /= len(valid_dataloader) * BATCHSIZE
#     #         vall_epoches[f"reg{j}"]["mape"] /= len(valid_dataloader) * BATCHSIZE
#
#     print("valid: edges:", round(val_edges, 4), "loss:", round(val_loss, 4))
#
#     wandb.log(
#         {f"reg{i}_cos_sim": vall_epoches[f"reg{i}"]["cos_sim"] for i in range(ALL_REGS)} |
#         {f"reg{i}_mse": vall_epoches[f"reg{i}"]["mse"] for i in range(ALL_REGS)} |
#         {f"reg{i}_mape": vall_epoches[f"reg{i}"]["mape"] for i in range(ALL_REGS)} |
#         {"train_loss": train_loss,
#          "val_loss": val_loss,
#          "val_edges": val_edges})
#     edge_c *= 2
#
# transformer.eval()
# run.finish()
#
# torch.save(transformer.state_dict(), 'weights_1024.pth')


