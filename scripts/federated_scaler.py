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
import typer
from rich.console import Console

from hfl_cicids.config import DEFAULT_PARTITIONS_DIR, DEFAULT_SCALER_PATH

app = typer.Typer(add_completion=False)
console = Console()


def local_feature_stats(train_path: Path) -> dict:
    df = pd.read_parquet(train_path)
    feature_cols = [c for c in df.columns if c != "label"]
    x = df[feature_cols].astype("float64").replace([np.inf, -np.inf], np.nan)
    return {
        "n": int(len(x)),
        "features": feature_cols,
        "sum": x.fillna(0.0).sum().to_dict(),
        "sumsq": x.fillna(0.0).pow(2).sum().to_dict(),
        "non_missing": x.notna().sum().to_dict(),
    }


def aggregate_stats(stats: list[dict]) -> dict:
    if not stats:
        raise ValueError("No local scaler stats supplied")
    features = stats[0]["features"]
    total_non_missing = {feature: 0 for feature in features}
    total_sum = {feature: 0.0 for feature in features}
    total_sumsq = {feature: 0.0 for feature in features}

    for item in stats:
        if item["features"] != features:
            raise ValueError("All local stats must use the same feature order")
        for feature in features:
            total_non_missing[feature] += int(item["non_missing"][feature])
            total_sum[feature] += float(item["sum"][feature])
            total_sumsq[feature] += float(item["sumsq"][feature])

    mean = {}
    std = {}
    for feature in features:
        n = max(1, total_non_missing[feature])
        mu = total_sum[feature] / n
        variance = max(0.0, total_sumsq[feature] / n - mu**2)
        mean[feature] = float(mu)
        std[feature] = float(np.sqrt(variance) or 1.0)

    return {
        "features": features,
        "mean": mean,
        "std": std,
        "fitted_on": "federated aggregate train statistics",
    }


@app.command()
def main(
    partitions_dir: Path = typer.Option(DEFAULT_PARTITIONS_DIR, "--partitions-dir"),
    output: Path = typer.Option(DEFAULT_SCALER_PATH, "--output", "-o"),
) -> None:
    """Aggregate per-site n/sum/sumsq feature statistics into scaler metadata."""

    stats = []
    for train_path in sorted(partitions_dir.glob("*/train.parquet")):
        stats.append(local_feature_stats(train_path))
        console.print(f"Collected stats from {train_path}")

    scaler = aggregate_stats(stats)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scaler, indent=2, sort_keys=True))
    console.print(f"[green]Wrote federated scaler metadata to {output}[/green]")


if __name__ == "__main__":
    app()
