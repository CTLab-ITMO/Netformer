import torch
from torch.utils.data import DataLoader
import pandas as pd

from lib.netformer.dataset import NetDataset, collate_fn
from lib.netformer.netformer import NetFormer


device = 'cuda' if torch.cuda.is_available() else 'cpu'

d_model = 1024
num_heads = 4
num_layers = 4
d_ff = 1024
dropout = 0

transformer = NetFormer(d_model, num_heads, num_layers, d_ff, dropout)
transformer.to(device)
transformer.load_state_dict(torch.load("lib/generator/weights/transformer.pth"))

transformer.eval()

regs = pd.read_csv(r"lib/generator/data/random_regressions.csv")
uniq_regs = regs["reg_id"].unique()
reg_data = [regs[regs["reg_id"] == i] for i in uniq_regs]

datasets_dir = "lib/generator/data"

train_dataset = NetDataset(0, 80, reg_data, datasets_dir)
valid_dataset = NetDataset(80, 100, reg_data, datasets_dir)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=20, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=20, shuffle=False)


data = next(iter(train_dataloader))
latent = torch.normal(mean=0.5, std=1, size=(1, 2048 - 26))
latent.to(device)
# latent = torch.cat([torch.normal(mean=0.5, std=1, size=(1, 1024-26)).squeeze(0), data[2][0]])
decoder_input = torch.tensor([[-1.0000 for _ in range(4)]]).to(device)
for i in range(150):
    tgt_mask = transformer.generate_mask(decoder_input.unsqueeze(0), is_inference=False).to(device)

    output = transformer.decode(latent.unsqueeze(0).to(device), decoder_input.unsqueeze(0),
                                data[2][0].unsqueeze(0).to(device), tgt_mask)
    decoder_input = torch.cat([decoder_input, output[:, -1]]).to(device)
print(output[:, :, :4])
print(latent[0][0])