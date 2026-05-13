from __future__ import annotations

import json
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ModuleNotFoundError:  # pragma: no cover - used when run as python scripts/foo.py
    from _bootstrap import bootstrap

bootstrap()

import pandas as pd
import torch
import typer
from rich.console import Console
from torch.utils.data import ConcatDataset, DataLoader

from hfl_cicids.checkpointing import save_torch_checkpoint
from hfl_cicids.config import DEFAULT_PARTITIONS_DIR, HOSPITALS
from hfl_cicids.metrics import evaluate_binary_classifier
from hfl_cicids.task import IDSFlowDataset, TabularIDSNet, infer_input_dim, train_one_client

app = typer.Typer(add_completion=False)
console = Console()


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@app.command()
def main(
    partitions_dir: Path = typer.Option(DEFAULT_PARTITIONS_DIR, "--partitions-dir"),
    output_checkpoint: Path = typer.Option(
        Path("shared/checkpoints/centralized_mlp.pt"),
        "--output-checkpoint",
    ),
    output_metrics: Path = typer.Option(
        Path("reports/centralized_mlp_metrics.csv"),
        "--output-metrics",
    ),
    epochs: int = typer.Option(3, "--epochs"),
    batch_size: int = typer.Option(1024, "--batch-size"),
    learning_rate: float = typer.Option(0.001, "--learning-rate"),
) -> None:
    """Train a centralized MLP baseline over all simulated hospital train splits."""

    first_dir = partitions_dir / HOSPITALS[0].hospital_id
    input_dim = infer_input_dim(first_dir)
    model = TabularIDSNet(input_dim=input_dim)

    train_datasets = [
        IDSFlowDataset(partitions_dir / hospital.hospital_id / "train.parquet") for hospital in HOSPITALS
    ]
    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=batch_size,
        shuffle=True,
    )
    loss = train_one_client(model, train_loader, epochs, learning_rate, _device())
    save_torch_checkpoint(
        model,
        output_checkpoint,
        {
            "model_kind": "centralized_mlp",
            "input_dim": input_dim,
            "epochs": epochs,
            "train_loss": float(loss),
        },
    )

    rows = []
    for hospital in HOSPITALS:
        data_dir = partitions_dir / hospital.hospital_id
        test_loader = DataLoader(
            IDSFlowDataset(data_dir / "test.parquet"),
            batch_size=batch_size,
            shuffle=False,
        )
        metrics = evaluate_binary_classifier(model, test_loader, _device())
        rows.append({"hospital_id": hospital.hospital_id, "region": hospital.region, **metrics})

    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_metrics, index=False)
    output_metrics.with_suffix(".metadata.json").write_text(
        json.dumps({"checkpoint": str(output_checkpoint), "train_loss": float(loss)}, indent=2)
    )
    console.print(f"[green]Wrote centralized checkpoint to {output_checkpoint}[/green]")
    console.print(f"[green]Wrote metrics to {output_metrics}[/green]")


if __name__ == "__main__":
    app()
