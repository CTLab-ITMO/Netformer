import torch
from torch.utils.data import Dataset, DataLoader


class ModelsDataset(Dataset):
    def __init__(self, models: torch.tensor):
        super().__init__()
        self.features = models
        self.targets = torch.arange(0, models.shape[0])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

