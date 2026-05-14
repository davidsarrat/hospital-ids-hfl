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

from hfl_cicids.checkpointing import save_torch_checkpoint
from hfl_cicids.config import DEFAULT_PARTITIONS_DIR, HOSPITALS
from hfl_cicids.metrics import evaluate_binary_classifier
from hfl_cicids.task import TabularIDSNet, infer_input_dim, load_dataloaders, train_one_client

app = typer.Typer(add_completion=False)
console = Console()


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@app.command()
def main(
    partitions_dir: Path = typer.Option(DEFAULT_PARTITIONS_DIR, "--partitions-dir"),
    output_dir: Path = typer.Option(Path("shared/checkpoints/local_only"), "--output-dir"),
    output_metrics: Path = typer.Option(Path("reports/local_only_metrics.csv"), "--output-metrics"),
    epochs: int = typer.Option(3, "--epochs"),
    batch_size: int = typer.Option(1024, "--batch-size"),
    learning_rate: float = typer.Option(0.001, "--learning-rate"),
) -> None:
    """Train one independent local-only model per healthcare-network site."""

    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for hospital in HOSPITALS:
        data_dir = partitions_dir / hospital.hospital_id
        input_dim = infer_input_dim(data_dir)
        model = TabularIDSNet(input_dim=input_dim)
        train_loader, _, test_loader = load_dataloaders(data_dir, batch_size)
        loss = train_one_client(model, train_loader, epochs, learning_rate, _device())
        metrics = evaluate_binary_classifier(model, test_loader, _device())
        checkpoint = output_dir / f"{hospital.hospital_id}.pt"
        save_torch_checkpoint(
            model,
            checkpoint,
            {
                "model_kind": "local_only",
                "hospital_id": hospital.hospital_id,
                "region": hospital.region,
                "input_dim": input_dim,
                "epochs": epochs,
                "train_loss": float(loss),
            },
        )
        rows.append(
            {
                "hospital_id": hospital.hospital_id,
                "region": hospital.region,
                "checkpoint": str(checkpoint),
                "train_loss": float(loss),
                **metrics,
            }
        )
        console.print(f"{hospital.hospital_id}: loss={loss:.4f} f1={metrics['eval_f1']:.4f}")

    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_metrics, index=False)
    output_metrics.with_suffix(".metadata.json").write_text(
        json.dumps({"checkpoint_dir": str(output_dir)}, indent=2)
    )
    console.print(f"[green]Wrote local-only checkpoints to {output_dir}[/green]")
    console.print(f"[green]Wrote metrics to {output_metrics}[/green]")


if __name__ == "__main__":
    app()
