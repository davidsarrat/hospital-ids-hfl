from __future__ import annotations

import json
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ModuleNotFoundError:  # pragma: no cover - used when run as python scripts/foo.py
    from _bootstrap import bootstrap

bootstrap()

import numpy as np
import pandas as pd
import torch
import typer
from rich.console import Console

from hfl_cicids.config import DEFAULT_PARTITIONS_DIR, HOSPITALS
from hfl_cicids.metrics import evaluate_binary_classifier
from hfl_cicids.task import TabularIDSNet, infer_input_dim, load_dataloaders

app = typer.Typer(add_completion=False)
console = Console()


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _evaluate_checkpoint(
    checkpoint: Path,
    partitions_dir: Path,
    batch_size: int,
    split: str,
    run_name: str,
    model_kind: str,
) -> list[dict]:
    if not checkpoint.exists():
        console.print(f"[yellow]Skipping missing checkpoint: {checkpoint}[/yellow]")
        return []

    first_dir = partitions_dir / HOSPITALS[0].hospital_id
    input_dim = infer_input_dim(first_dir)
    model = TabularIDSNet(input_dim=input_dim)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))

    rows = []
    for hospital in HOSPITALS:
        data_dir = partitions_dir / hospital.hospital_id
        if not data_dir.exists():
            console.print(f"[yellow]Skipping missing partition: {data_dir}[/yellow]")
            continue
        _, val_loader, test_loader = load_dataloaders(data_dir, batch_size)
        loader = test_loader if split == "test" else val_loader
        metrics = evaluate_binary_classifier(model, loader, _device())
        rows.append(
            {
                "run": run_name,
                "model_kind": model_kind,
                "checkpoint": str(checkpoint),
                "hospital_id": hospital.hospital_id,
                "region": hospital.region,
                **metrics,
            }
        )
    return rows


def _summary(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    summaries = []
    for (run, model_kind), group in df.groupby(["run", "model_kind"]):
        weights = group["num-examples"].to_numpy(dtype=float)
        summaries.append(
            {
                "run": run,
                "model_kind": model_kind,
                "scope": "global_summary",
                "weighted_f1": float(np.average(group["eval_f1"], weights=weights)),
                "macro_f1": float(group["eval_f1"].mean()),
                "weighted_roc_auc": float(np.average(group["eval_roc_auc"], weights=weights)),
                "weighted_auprc": float(np.average(group["eval_auprc"], weights=weights)),
                "mean_false_positive_rate": float(group["eval_false_positive_rate"].mean()),
                "worst_false_negative_rate": float(group["eval_false_negative_rate"].max()),
                "num_examples": int(group["num-examples"].sum()),
            }
        )
    return pd.DataFrame(summaries)


@app.command()
def main(
    checkpoint: Path = typer.Option(
        Path("shared/checkpoints/global/round_3.pt"),
        "--checkpoint",
        "-c",
        help="Hierarchical global checkpoint to evaluate.",
    ),
    partitions_dir: Path = typer.Option(DEFAULT_PARTITIONS_DIR, "--partitions-dir"),
    output: Path = typer.Option(
        Path("reports/metrics_summary.csv"),
        "--output",
        "-o",
        help="Metrics CSV output.",
    ),
    batch_size: int = typer.Option(4096, "--batch-size"),
    split: str = typer.Option("test", "--split", help="val or test."),
    flat_checkpoint: Path | None = typer.Option(
        None,
        "--flat-checkpoint",
        help="Optional flat FL checkpoint for comparison.",
    ),
    local_checkpoints_dir: Path | None = typer.Option(
        None,
        "--local-checkpoints-dir",
        help="Optional directory with <hospital_id>.pt local-only checkpoints.",
    ),
) -> None:
    """Evaluate global, optional flat, and optional local-only checkpoints."""

    rows = _evaluate_checkpoint(
        checkpoint=checkpoint,
        partitions_dir=partitions_dir,
        batch_size=batch_size,
        split=split,
        run_name="hierarchical_fl",
        model_kind="hierarchical_global",
    )

    if flat_checkpoint is not None:
        rows.extend(
            _evaluate_checkpoint(
                checkpoint=flat_checkpoint,
                partitions_dir=partitions_dir,
                batch_size=batch_size,
                split=split,
                run_name="flat_fl",
                model_kind="flat_global",
            )
        )

    if local_checkpoints_dir is not None:
        for hospital in HOSPITALS:
            local_checkpoint = local_checkpoints_dir / f"{hospital.hospital_id}.pt"
            local_rows = _evaluate_checkpoint(
                checkpoint=local_checkpoint,
                partitions_dir=partitions_dir,
                batch_size=batch_size,
                split=split,
                run_name="local_only",
                model_kind=hospital.hospital_id,
            )
            rows.extend([row for row in local_rows if row["hospital_id"] == hospital.hospital_id])

    if not rows:
        raise FileNotFoundError("No checkpoints could be evaluated")

    detail = pd.DataFrame(rows)
    summary = _summary(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output, index=False)
    summary_output = output.with_name(f"{output.stem}_global{output.suffix}")
    summary.to_csv(summary_output, index=False)
    metadata = {
        "detail_rows": int(len(detail)),
        "summary_rows": int(len(summary)),
        "split": split,
        "outputs": [str(output), str(summary_output)],
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    console.print(f"[green]Wrote site metrics to {output}[/green]")
    console.print(f"[green]Wrote global summary to {summary_output}[/green]")


if __name__ == "__main__":
    app()
