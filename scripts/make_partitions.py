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
from sklearn.model_selection import train_test_split

from hfl_cicids.config import (
    DEFAULT_PARTITIONS_DIR,
    DEFAULT_PROCESSED_PATH,
    DEFAULT_SCALER_PATH,
    HOSPITALS,
    Hospital,
)

app = typer.Typer(add_completion=False)
console = Console()


def attack_group(label: str) -> str:
    value = str(label).strip().upper()
    if value == "BENIGN":
        return "BENIGN"
    if "DDOS" in value:
        return "DDOS"
    if "HEARTBLEED" in value:
        return "HEARTBLEED"
    if "DOS" in value:
        return "DOS"
    if "FTP" in value or "SSH" in value or "BRUTE" in value or "PATATOR" in value:
        return "BRUTE_FORCE"
    if "WEB" in value:
        return "WEB"
    if "PORTSCAN" in value or "PORT SCAN" in value:
        return "PORTSCAN"
    if "BOT" in value:
        return "BOTNET"
    if "INFILTRATION" in value:
        return "INFILTRATION"
    return "OTHER_ATTACK"


def _allocation_weights(
    group: str,
    hospitals: tuple[Hospital, ...],
    rng: np.random.Generator,
    alpha: float,
) -> np.ndarray:
    base = rng.dirichlet(np.full(len(hospitals), alpha))
    preference = np.array(
        [3.0 if group in hospital.preferred_attack_groups else 0.25 for hospital in hospitals],
        dtype=float,
    )
    if group != "BENIGN":
        preference += np.array(
            [0.50 if hospital.hospital_id == "hospital_na_03" else 0.0 for hospital in hospitals]
        )
    weights = base * preference
    if weights.sum() <= 0:
        weights = np.ones(len(hospitals), dtype=float)
    return weights / weights.sum()


def _repair_min_group_rows(
    assignments: pd.Series,
    group_mask: pd.Series,
    min_rows: int,
    rng: np.random.Generator,
) -> None:
    """Move already assigned rows so each site has a minimum group count."""

    if min_rows <= 0:
        return

    hospital_ids = [hospital.hospital_id for hospital in HOSPITALS]
    eligible_idx = assignments.index[group_mask]
    if len(eligible_idx) < min_rows * len(hospital_ids):
        return

    for hospital_id in hospital_ids:
        current = assignments.index[(assignments == hospital_id) & group_mask]
        deficit = min_rows - len(current)
        if deficit <= 0:
            continue

        donors = []
        for donor_id in hospital_ids:
            if donor_id == hospital_id:
                continue
            donor_idx = assignments.index[(assignments == donor_id) & group_mask]
            extra = len(donor_idx) - min_rows
            if extra > 0:
                donors.extend(rng.choice(np.array(donor_idx), size=extra, replace=False).tolist())

        if len(donors) < deficit:
            continue
        moved = rng.choice(np.array(donors), size=deficit, replace=False)
        assignments.loc[moved] = hospital_id


def allocate_hospitals(df: pd.DataFrame, alpha: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    hospital_ids = np.array([hospital.hospital_id for hospital in HOSPITALS])
    groups = df["attack_type"].map(attack_group) if "attack_type" in df.columns else df["label"].map(str)
    assignments = pd.Series(index=df.index, dtype="object")

    for group, group_index in groups.groupby(groups).groups.items():
        idx = np.array(list(group_index))
        weights = _allocation_weights(str(group), HOSPITALS, rng, alpha)
        assignments.loc[idx] = rng.choice(hospital_ids, size=len(idx), p=weights)

    benign_mask = groups == "BENIGN"
    attack_mask = groups != "BENIGN"
    _repair_min_group_rows(assignments, benign_mask, min_rows=10, rng=rng)
    _repair_min_group_rows(assignments, attack_mask, min_rows=1, rng=rng)

    return assignments


def _stratify_or_none(labels: pd.Series) -> pd.Series | None:
    counts = labels.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return labels


def split_hospital(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(df) < 3:
        raise ValueError(
            "A site partition has fewer than three rows. Use a larger sample or the full dataset."
        )

    temp_size = val_size + test_size
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=seed,
        stratify=_stratify_or_none(df["label"]),
    )

    relative_test_size = test_size / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=seed + 1,
        stratify=_stratify_or_none(temp_df["label"]),
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _fit_scaler(train_frames: list[pd.DataFrame], feature_cols: list[str]) -> dict:
    train_features = pd.concat([frame[feature_cols] for frame in train_frames], ignore_index=True)
    train_features = train_features.replace([np.inf, -np.inf], np.nan)
    impute_values = train_features.median(numeric_only=True).fillna(0.0)
    imputed = train_features.fillna(impute_values)
    means = imputed.mean(numeric_only=True)
    stds = imputed.std(numeric_only=True, ddof=0).replace(0.0, 1.0).fillna(1.0)
    return {
        "features": feature_cols,
        "impute_values": {col: float(impute_values[col]) for col in feature_cols},
        "mean": {col: float(means[col]) for col in feature_cols},
        "std": {col: float(stds[col]) for col in feature_cols},
        "fitted_on": "concatenated simulated healthcare-site train splits",
    }


def _transform_split(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    feature_cols = list(scaler["features"])
    x = df[feature_cols].astype("float64").replace([np.inf, -np.inf], np.nan)
    impute = pd.Series(scaler["impute_values"])
    mean = pd.Series(scaler["mean"])
    std = pd.Series(scaler["std"])
    x = x.fillna(impute)
    x = (x - mean) / std
    out = x.astype("float32")
    out["label"] = df["label"].astype("int64").to_numpy()
    return out


def _metadata(
    hospital: Hospital,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    source = pd.concat([train_df, val_df, test_df], ignore_index=True)
    attack_counts = (
        source["attack_type"].map(attack_group).value_counts().sort_index().to_dict()
        if "attack_type" in source.columns
        else {}
    )
    return {
        "hospital_id": hospital.hospital_id,
        "region": hospital.region,
        "num_train": int(len(train_df)),
        "num_val": int(len(val_df)),
        "num_test": int(len(test_df)),
        "num_features": int(len(feature_cols)),
        "label_mapping": {"BENIGN": 0, "ATTACK": 1},
        "train_label_counts": train_df["label"].value_counts().sort_index().to_dict(),
        "val_label_counts": val_df["label"].value_counts().sort_index().to_dict(),
        "test_label_counts": test_df["label"].value_counts().sort_index().to_dict(),
        "attack_group_counts": attack_counts,
    }


@app.command()
def main(
    input_path: Path = typer.Option(
        DEFAULT_PROCESSED_PATH,
        "--input",
        "-i",
        help="Processed CIC-IDS2017 parquet from prepare_cicids.py.",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_PARTITIONS_DIR,
        "--output-dir",
        "-o",
        help="Healthcare-site partition output directory.",
    ),
    scaler_path: Path = typer.Option(
        DEFAULT_SCALER_PATH,
        "--scaler-path",
        help="Shared scaler metadata path.",
    ),
    alpha: float = typer.Option(0.30, "--alpha", help="Dirichlet concentration."),
    val_size: float = typer.Option(0.15, "--val-size", help="Validation fraction."),
    test_size: float = typer.Option(0.15, "--test-size", help="Test fraction."),
    seed: int = typer.Option(123, "--seed", help="Random seed."),
) -> None:
    """Create six non-IID healthcare-site partitions and train-only scaler metadata."""

    df = pd.read_parquet(input_path)
    if "label" not in df.columns:
        raise ValueError(f"'label' column missing from {input_path}")
    if "attack_type" not in df.columns:
        df["attack_type"] = df["label"].map(lambda x: "BENIGN" if int(x) == 0 else "OTHER_ATTACK")

    feature_cols = [c for c in df.columns if c not in {"label", "attack_type"}]
    assignments = allocate_hospitals(df, alpha=alpha, seed=seed)
    df = df.assign(hospital_id=assignments)

    splits: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    train_frames: list[pd.DataFrame] = []
    for hospital in HOSPITALS:
        hospital_df = df[df["hospital_id"] == hospital.hospital_id].drop(columns=["hospital_id"])
        train_df, val_df, test_df = split_hospital(
            hospital_df,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
        )
        splits[hospital.hospital_id] = (train_df, val_df, test_df)
        train_frames.append(train_df)

    scaler = _fit_scaler(train_frames, feature_cols)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.write_text(json.dumps(scaler, indent=2, sort_keys=True))

    output_dir.mkdir(parents=True, exist_ok=True)
    for hospital in HOSPITALS:
        train_df, val_df, test_df = splits[hospital.hospital_id]
        hospital_dir = output_dir / hospital.hospital_id
        hospital_dir.mkdir(parents=True, exist_ok=True)

        _transform_split(train_df, scaler).to_parquet(hospital_dir / "train.parquet", index=False)
        _transform_split(val_df, scaler).to_parquet(hospital_dir / "val.parquet", index=False)
        _transform_split(test_df, scaler).to_parquet(hospital_dir / "test.parquet", index=False)

        meta = _metadata(hospital, train_df, val_df, test_df, feature_cols)
        (hospital_dir / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
        console.print(
            f"{hospital.hospital_id}: train={meta['num_train']:,} "
            f"val={meta['num_val']:,} test={meta['num_test']:,}"
        )

    console.print(f"[green]Wrote partitions to {output_dir}[/green]")
    console.print(f"[green]Wrote train-only scaler to {scaler_path}[/green]")


if __name__ == "__main__":
    app()
