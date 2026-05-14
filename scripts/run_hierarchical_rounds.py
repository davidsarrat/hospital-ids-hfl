from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ModuleNotFoundError:  # pragma: no cover - used when run as python scripts/foo.py
    from _bootstrap import bootstrap

bootstrap()

import pandas as pd
import typer
from rich.console import Console

from hfl_cicids.checkpointing import ensure_initial_checkpoint
from hfl_cicids.config import (
    DEFAULT_PARTITIONS_DIR,
    DEFAULT_SHARED_DIR,
    REGIONS,
    global_checkpoint,
    hospitals_by_region,
    parse_regions,
    region_checkpoint,
)

app = typer.Typer(add_completion=False)
console = Console(no_color=True, highlight=False, soft_wrap=True, width=220)


def _no_color_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "NO_COLOR": "1",
            "CLICOLOR": "0",
            "FORCE_COLOR": "0",
            "RICH_NO_COLOR": "1",
            "RICH_FORCE_TERMINAL": "0",
            "TERM": "dumb",
        }
    )
    return env


def _metadata(partition_dir: Path) -> dict:
    path = partition_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing partition metadata: {path}")
    return json.loads(path.read_text())


def infer_input_dim(partitions_dir: Path) -> int:
    train_paths = sorted(partitions_dir.glob("*/train.parquet"))
    if not train_paths:
        raise FileNotFoundError(f"No site train.parquet files found under {partitions_dir}")
    sample = pd.read_parquet(train_paths[0])
    return len([c for c in sample.columns if c != "label"])


def region_train_examples(region: str, partitions_dir: Path, allow_missing: bool = False) -> int:
    total = 0
    for hospital in hospitals_by_region(region):
        try:
            metadata = _metadata(partitions_dir / hospital.hospital_id)
        except FileNotFoundError:
            if allow_missing:
                total += 1
                continue
            raise
        total += int(metadata["num_train"])
    return total


def global_train_examples(regions: list[str], partitions_dir: Path, allow_missing: bool = False) -> int:
    return sum(
        region_train_examples(region, partitions_dir, allow_missing=allow_missing)
        for region in regions
    )


def _run_config(items: dict[str, object]) -> str:
    formatted = []
    for key, value in items.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, int | float):
            rendered = str(value)
        else:
            rendered = json.dumps(str(value))
        formatted.append(f"{key}={rendered}")
    return " ".join(formatted)


def _run_flower(profile: str, run_config: dict[str, object], dry_run: bool) -> None:
    command = [
        "flwr",
        "run",
        ".",
        profile,
        "--stream",
        "--run-config",
        _run_config(run_config),
    ]
    console.print()
    console.print(f"[bold]Submitting Flower run to {profile}[/bold]")
    console.print(f"$ {shlex.join(command)}")
    if not dry_run:
        subprocess.run(command, check=True, env=_no_color_env())


def _checkpoint_summary(checkpoint: Path) -> str:
    metadata_path = checkpoint.with_suffix(".metadata.json")
    if not checkpoint.exists():
        return f"Checkpoint missing: {checkpoint}"
    if not metadata_path.exists():
        return f"Checkpoint written: {checkpoint}"

    metadata = json.loads(metadata_path.read_text())
    parts = [f"Checkpoint written: {checkpoint}"]
    if "num_examples" in metadata:
        parts.append(f"num_examples={int(metadata['num_examples']):,}")
    if metadata.get("level") == "regional":
        parts.extend(
            [
                f"val_f1={float(metadata.get('val_f1', 0.0)):.4f}",
                f"val_roc_auc={float(metadata.get('val_roc_auc', 0.0)):.4f}",
                f"val_auprc={float(metadata.get('val_auprc', 0.0)):.4f}",
            ]
        )
    return " | ".join(parts)


def _regional_run_config(
    region: str,
    round_number: int,
    regional_rounds: int,
    input_dim: int,
    partitions_dir: Path,
    shared_dir: Path,
    runtime_shared_dir: Path,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    allow_missing_metadata: bool,
) -> dict[str, object]:
    min_nodes = len(hospitals_by_region(region))
    return {
        "level": "regional",
        "region": region,
        "global-round": round_number,
        "init-checkpoint": global_checkpoint(round_number - 1, runtime_shared_dir),
        "output-checkpoint": region_checkpoint(region, round_number, runtime_shared_dir),
        "num-server-rounds": regional_rounds,
        "local-epochs": local_epochs,
        "batch-size": batch_size,
        "learning-rate": learning_rate,
        "input-dim": input_dim,
        "region-num-examples": region_train_examples(
            region,
            partitions_dir,
            allow_missing=allow_missing_metadata,
        ),
        "fraction-train": 1.0,
        "fraction-evaluate": 1.0,
        "min-train-nodes": min_nodes,
        "min-evaluate-nodes": min_nodes,
        "min-available-nodes": min_nodes,
    }


def _global_run_config(
    round_number: int,
    input_dim: int,
    regions: list[str],
    partitions_dir: Path,
    shared_dir: Path,
    runtime_shared_dir: Path,
    batch_size: int,
    allow_missing_metadata: bool,
) -> dict[str, object]:
    return {
        "level": "global",
        "global-round": round_number,
        "init-checkpoint": global_checkpoint(round_number - 1, runtime_shared_dir),
        "output-checkpoint": global_checkpoint(round_number, runtime_shared_dir),
        "num-server-rounds": 1,
        "local-epochs": 1,
        "batch-size": batch_size,
        "learning-rate": 0.0,
        "input-dim": input_dim,
        "global-num-examples": global_train_examples(
            regions,
            partitions_dir,
            allow_missing=allow_missing_metadata,
        ),
        "fraction-train": 1.0,
        "fraction-evaluate": 0.0,
        "min-train-nodes": len(regions),
        "min-evaluate-nodes": 0,
        "min-available-nodes": len(regions),
    }


@app.command()
def main(
    global_rounds: int = typer.Option(3, "--global-rounds", help="Global HFL rounds."),
    regional_rounds: int = typer.Option(2, "--regional-rounds", help="Regional FedAvg rounds."),
    regions: str = typer.Option("region_eu,region_na", "--regions", help="Comma-separated regions."),
    partitions_dir: Path = typer.Option(DEFAULT_PARTITIONS_DIR, "--partitions-dir"),
    shared_dir: Path = typer.Option(DEFAULT_SHARED_DIR, "--shared-dir"),
    runtime_shared_dir: Path = typer.Option(
        Path("/shared"),
        "--runtime-shared-dir",
        help="Shared directory as seen by Flower ServerApp/ClientApp containers.",
    ),
    local_epochs: int = typer.Option(1, "--local-epochs"),
    batch_size: int = typer.Option(1024, "--batch-size"),
    learning_rate: float = typer.Option(0.001, "--learning-rate"),
    input_dim: int = typer.Option(
        0,
        "--input-dim",
        help="Override input dimension. Useful for dry-runs before data exists.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print flwr commands without running them."),
) -> None:
    """Run repeated regional Flower federations followed by global gateway aggregation."""

    selected_regions = parse_regions(regions)
    if not set(selected_regions).issubset(set(REGIONS)):
        raise ValueError(f"Unsupported regions: {selected_regions}")

    if input_dim <= 0:
        try:
            input_dim = infer_input_dim(partitions_dir)
        except FileNotFoundError:
            if not dry_run:
                raise
            input_dim = 1
    shared_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        ensure_initial_checkpoint(global_checkpoint(0, shared_dir), input_dim=input_dim)

    for round_number in range(1, global_rounds + 1):
        console.rule(f"Global round {round_number}")
        console.print(
            f"Starting from global checkpoint: {global_checkpoint(round_number - 1, shared_dir)}"
        )
        for region in selected_regions:
            profile = region.replace("_", "-")
            console.print()
            console.print(
                f"Regional phase: {region} | site_nodes={len(hospitals_by_region(region))} | "
                f"train_examples={region_train_examples(region, partitions_dir, allow_missing=dry_run):,} | "
                f"regional_rounds={regional_rounds}"
            )
            _run_flower(
                profile=profile,
                run_config=_regional_run_config(
                    region=region,
                    round_number=round_number,
                    regional_rounds=regional_rounds,
                    input_dim=input_dim,
                    partitions_dir=partitions_dir,
                    shared_dir=shared_dir,
                    runtime_shared_dir=runtime_shared_dir,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    allow_missing_metadata=dry_run,
                ),
                dry_run=dry_run,
            )
            if not dry_run:
                console.print(_checkpoint_summary(region_checkpoint(region, round_number, shared_dir)))

        console.print()
        console.print(
            f"Global phase: gateway_nodes={len(selected_regions)} | "
            f"train_examples={global_train_examples(selected_regions, partitions_dir, allow_missing=dry_run):,}"
        )
        _run_flower(
            profile="global",
            run_config=_global_run_config(
                round_number=round_number,
                input_dim=input_dim,
                regions=selected_regions,
                partitions_dir=partitions_dir,
                shared_dir=shared_dir,
                runtime_shared_dir=runtime_shared_dir,
                batch_size=batch_size,
                allow_missing_metadata=dry_run,
            ),
            dry_run=dry_run,
        )
        if not dry_run:
            console.print(_checkpoint_summary(global_checkpoint(round_number, shared_dir)))

    console.print(f"[green]Latest global checkpoint: {global_checkpoint(global_rounds, shared_dir)}[/green]")


if __name__ == "__main__":
    app()
