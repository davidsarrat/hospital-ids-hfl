from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class IDSFlowDataset(Dataset):
    """Parquet-backed CIC-IDS2017 binary classification dataset."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset split not found: {path}")

        df = pd.read_parquet(path)
        if "label" not in df.columns:
            raise ValueError(f"'label' column missing in {path}")

        self.y = df["label"].astype("float32").to_numpy(copy=True)
        self.x = df.drop(columns=["label"]).astype("float32").to_numpy(copy=True)

        if len(self.x) == 0:
            raise ValueError(f"Dataset split is empty: {path}")
        if not np.isfinite(self.x).all():
            raise ValueError(f"Non-finite feature values found in {path}")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.x[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class TabularIDSNet(nn.Module):
    """Compact MLP for binary intrusion detection."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def infer_input_dim(data_dir: str | Path) -> int:
    sample = pd.read_parquet(Path(data_dir) / "train.parquet")
    return len([c for c in sample.columns if c != "label"])


def load_dataloaders(
    data_dir: str | Path,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_dir = Path(data_dir)

    train_ds = IDSFlowDataset(data_dir / "train.parquet")
    val_ds = IDSFlowDataset(data_dir / "val.parquet")
    test_ds = IDSFlowDataset(data_dir / "test.parquet")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def compute_pos_weight(train_loader: DataLoader) -> torch.Tensor:
    labels = []
    for _, y in train_loader:
        labels.append(y)
    y_all = torch.cat(labels)
    positives = torch.sum(y_all == 1).item()
    negatives = torch.sum(y_all == 0).item()
    if positives == 0:
        return torch.tensor(1.0)
    return torch.tensor(max(1.0, negatives / positives))


def train_one_client(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    model.to(device)
    model.train()

    pos_weight = compute_pos_weight(train_loader).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_loss = 0.0
    total_batches = 0

    for _ in range(epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

    return total_loss / max(1, total_batches)
