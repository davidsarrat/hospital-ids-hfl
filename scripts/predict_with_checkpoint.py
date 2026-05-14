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

from hfl_cicids.config import DEFAULT_PARTITIONS_DIR
from hfl_cicids.task import TabularIDSNet, infer_input_dim

app = typer.Typer(add_completion=False)
console = Console()


@torch.no_grad()
def predict_split(
    checkpoint: Path,
    hospital_id: str,
    partitions_dir: Path,
    split: str,
    threshold: float,
    limit: int,
) -> pd.DataFrame:
    """Load a global checkpoint and produce row-level predictions for one site split."""

    data_dir = partitions_dir / hospital_id
    split_path = data_dir / f"{split}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"Split not found: {split_path}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    df = pd.read_parquet(split_path)
    if "label" not in df.columns:
        raise ValueError(f"'label' column missing from {split_path}")

    input_dim = infer_input_dim(data_dir)
    model = TabularIDSNet(input_dim=input_dim)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    sample = df.head(limit).copy()
    x = torch.tensor(sample.drop(columns=["label"]).to_numpy(dtype="float32"))
    probs = torch.sigmoid(model(x)).cpu().numpy()
    labels = sample["label"].astype(int).to_numpy()
    preds = (probs >= threshold).astype(int)

    return pd.DataFrame(
        {
            "row_index": list(range(len(sample))),
            "hospital_id": hospital_id,
            "split": split,
            "label": labels,
            "prob_attack": probs,
            "prediction": preds,
            "correct": preds == labels,
        }
    )


@app.command()
def main(
    checkpoint: Path = typer.Option(
        Path("shared/checkpoints/global/round_1.pt"),
        "--checkpoint",
        "-c",
        help="Checkpoint to load for prediction.",
    ),
    hospital_id: str = typer.Option("hospital_eu_01", "--hospital-id"),
    partitions_dir: Path = typer.Option(DEFAULT_PARTITIONS_DIR, "--partitions-dir"),
    split: str = typer.Option("test", "--split", help="train, val, or test."),
    threshold: float = typer.Option(0.5, "--threshold"),
    limit: int = typer.Option(50, "--limit", help="Number of rows to predict."),
    output: Path = typer.Option(Path("reports/predictions_hospital_eu_01.csv"), "--output", "-o"),
) -> None:
    """Generate a small prediction table from a trained checkpoint."""

    predictions = predict_split(
        checkpoint=checkpoint,
        hospital_id=hospital_id,
        partitions_dir=partitions_dir,
        split=split,
        threshold=threshold,
        limit=limit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    metadata = {
        "checkpoint": str(checkpoint),
        "hospital_id": hospital_id,
        "split": split,
        "threshold": threshold,
        "rows": int(len(predictions)),
        "accuracy": float(predictions["correct"].mean()) if len(predictions) else 0.0,
        "output": str(output),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    console.print(predictions.head(12).to_string(index=False))
    console.print(f"[green]Wrote predictions to {output}[/green]")


if __name__ == "__main__":
    app()
