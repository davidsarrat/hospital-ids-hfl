from __future__ import annotations

import json
from collections import Counter
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

from hfl_cicids.config import DEFAULT_PROCESSED_PATH, DEFAULT_RAW_DIR

app = typer.Typer(add_completion=False)
console = Console()

LEAKAGE_COLUMNS = {
    "flow id",
    "source ip",
    "src ip",
    "destination ip",
    "dst ip",
    "timestamp",
}


def _unique_columns(columns: list[str]) -> list[str]:
    counts: Counter[str] = Counter()
    out: list[str] = []
    for col in columns:
        base = col.strip()
        counts[base] += 1
        if counts[base] == 1:
            out.append(base)
        else:
            out.append(f"{base}__{counts[base]}")
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin1")


def clean_cicids_csvs(
    raw_dir: Path,
    max_missing_fraction: float = 0.40,
    max_rows: int | None = None,
    global_impute: bool = False,
) -> tuple[pd.DataFrame, dict]:
    csvs = sorted(raw_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CIC-IDS2017 CSV files found under {raw_dir}")

    frames = []
    for csv_path in csvs:
        console.print(f"Reading {csv_path}")
        frame = _read_csv(csv_path)
        frame.columns = _unique_columns([str(c) for c in frame.columns])
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    label_cols = [c for c in df.columns if c.strip().lower() == "label"]
    if not label_cols:
        raise ValueError("Could not find a CIC-IDS2017 label column")
    label_col = label_cols[0]

    raw_labels = df[label_col].astype(str).str.strip()
    attack_type = raw_labels.str.upper().replace({"": "UNKNOWN"})
    labels = (attack_type != "BENIGN").astype("int64")

    drop_cols = {label_col}
    for col in df.columns:
        base_col = col.strip().lower().split("__", maxsplit=1)[0]
        if base_col in LEAKAGE_COLUMNS:
            drop_cols.add(col)

    features = df.drop(columns=list(drop_cols), errors="ignore")
    numeric = features.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    missing_fraction = numeric.isna().mean()
    kept_cols = missing_fraction[missing_fraction <= max_missing_fraction].index.tolist()
    dropped_cols = sorted(set(numeric.columns) - set(kept_cols))
    numeric = numeric[kept_cols]

    if global_impute:
        medians = numeric.median(numeric_only=True).fillna(0.0)
        numeric = numeric.fillna(medians)

    cleaned = numeric.copy()
    cleaned["label"] = labels
    cleaned["attack_type"] = attack_type

    metadata = {
        "source_files": [str(path) for path in csvs],
        "rows": int(len(cleaned)),
        "feature_columns": kept_cols,
        "num_features": int(len(kept_cols)),
        "label_mapping": {"BENIGN": 0, "ATTACK": 1},
        "attack_distribution": attack_type.value_counts().to_dict(),
        "dropped_columns": dropped_cols,
        "max_missing_fraction": max_missing_fraction,
        "global_impute": global_impute,
        "note": (
            "Final train-only imputation and standardization are performed by "
            "scripts/make_partitions.py."
        ),
    }
    return cleaned, metadata


@app.command()
def main(
    raw_dir: Path = typer.Option(DEFAULT_RAW_DIR, "--raw-dir", help="Raw CIC-IDS2017 directory."),
    output: Path = typer.Option(
        DEFAULT_PROCESSED_PATH,
        "--output",
        "-o",
        help="Processed parquet output path.",
    ),
    max_missing_fraction: float = typer.Option(
        0.40,
        "--max-missing-fraction",
        help="Drop feature columns above this NaN fraction.",
    ),
    max_rows: int | None = typer.Option(
        None,
        "--max-rows",
        help="Optional sample size for fast smoke tests.",
    ),
    global_impute: bool = typer.Option(
        False,
        "--global-impute/--defer-imputation",
        help="Impute in this step. Default defers to train-only partition statistics.",
    ),
) -> None:
    """Clean CIC-IDS2017 CSVs into a binary-classification parquet file."""

    cleaned, metadata = clean_cicids_csvs(
        raw_dir=raw_dir,
        max_missing_fraction=max_missing_fraction,
        max_rows=max_rows,
        global_impute=global_impute,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output, index=False)
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    console.print(f"[green]Wrote {len(cleaned):,} rows to {output}[/green]")


if __name__ == "__main__":
    app()
