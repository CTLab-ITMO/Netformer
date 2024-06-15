import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.diffusion.DDPM import DDPM
from experiments.diffusion.MatrixConverter import matrix_converter
from lib.netformer.dataset import Net

device = 'cuda' if torch.cuda.is_available() else 'cpu'

time_steps_number = 1000
model = DDPM(time_steps_number, device)

n_epochs = 20
lr = 0.001
n_steps = model.n_steps
n_samples = 25

optimm = Adam(model.noise_predictor.parameters(), lr)

losses = [[], []]

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
    },
}


train_data = torch.stack([matrix_converter(Net(10,1, act)) for i in range(1000)])
test_data = torch.stack([matrix_converter(Net(10,1, act)) for i in range(1000)])


train_dataloader = DataLoader(train_data, batch_size=20, shuffle=True)#, transform=transform)
val_dataloader = DataLoader(test_data, batch_size=20, shuffle=True)


for epoch in range(1, n_epochs + 1):
    train_loss = 0.0
    model.train()
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}/{n_epochs}"):
        batch = batch.to(device)
        # x0_class = x0_class.to(device)

        batch_size = len(batch)

        time_step = torch.randint(0, n_steps, (batch_size,)).to(device)

        loss = model.loss(batch, time_step)
        optimm.zero_grad()
        loss.backward()
        optimm.step()

        train_loss += loss.item() * len(batch) / len(train_dataloader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = batch.to(device)
            batch_size = len(batch)
            time_step = torch.randint(0, n_steps, (batch_size,)).to(device)

            loss = model.loss(batch, time_step)
            val_loss += loss.item() * len(batch) / len(val_dataloader.dataset)

    print(f"Epoch {epoch}/{n_epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    losses[0].append(train_loss)
    losses[1].append(val_loss)
