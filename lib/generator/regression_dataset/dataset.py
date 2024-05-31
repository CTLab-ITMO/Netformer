import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class RegDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.features = torch.tensor(df.drop(columns=['reg_id', 'y']).values, dtype=torch.float32)
        self.targets = torch.tensor(df['y'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def get_dataloaders_and_datasets(reg_data):
    reg_datasets = []
    reg_dataloaders = []
    for reg in reg_data:
        train_data, valid_data = train_test_split(reg, test_size=0.1, random_state=42)
        train_dataset = RegDataset(train_data)
        valid_dataset = RegDataset(valid_data)
        reg_datasets += [[train_dataset, valid_dataset]]
        train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=50, shuffle=False)
        reg_dataloaders += [[train_loader, valid_loader]]
    return reg_dataloaders, reg_datasets
