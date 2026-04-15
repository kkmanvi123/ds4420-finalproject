import torch
from torch.utils.data import Dataset, DataLoader


class PCADataset(Dataset):
    """Simple dataset for PCA-transformed features and labels."""

    def __init__(self, X, y, task="regression"):
        self.X = torch.tensor(X, dtype=torch.float32)

        if task == "classification":
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloaders(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    batch_size=32,
    task="regression"
):
    """
    Create train/val/test dataloaders from already-split arrays.
    No splitting is done here.
    """
    train_ds = PCADataset(X_train, y_train, task=task)
    val_ds = PCADataset(X_val, y_val, task=task)
    test_ds = PCADataset(X_test, y_test, task=task)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader