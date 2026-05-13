from __future__ import annotations

import shlex
import subprocess
import json
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ModuleNotFoundError:  # pragma: no cover - used when run as python scripts/foo.py
    from _bootstrap import bootstrap

bootstrap()

import typer
from rich.console import Console

from hfl_cicids.checkpointing import ensure_initial_checkpoint
from hfl_cicids.config import DEFAULT_PARTITIONS_DIR, DEFAULT_SHARED_DIR, HOSPITALS
from scripts.run_hierarchical_rounds import _run_config, infer_input_dim

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    rounds: int = typer.Option(3, "--rounds", help="Flat FedAvg server rounds."),
    partitions_dir: Path = typer.Option(DEFAULT_PARTITIONS_DIR, "--partitions-dir"),
    shared_dir: Path = typer.Option(DEFAULT_SHARED_DIR, "--shared-dir"),
    runtime_shared_dir: Path = typer.Option(
        Path("/shared"),
        "--runtime-shared-dir",
        help="Shared directory as seen by Flower ServerApp containers.",
    ),
    local_epochs: int = typer.Option(1, "--local-epochs"),
    batch_size: int = typer.Option(1024, "--batch-size"),
    learning_rate: float = typer.Option(0.001, "--learning-rate"),
    input_dim: int = typer.Option(
        0,
        "--input-dim",
        help="Override input dimension. Useful for dry-runs before data exists.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run a one-layer flat Flower FedAvg baseline against six hospital SuperNodes."""

    if input_dim <= 0:
        try:
            input_dim = infer_input_dim(partitions_dir)
        except FileNotFoundError:
            if not dry_run:
                raise
            input_dim = 1
    init_checkpoint = shared_dir / "checkpoints" / "flat" / "round_0.pt"
    runtime_init_checkpoint = runtime_shared_dir / "checkpoints" / "flat" / "round_0.pt"
    runtime_output_checkpoint = runtime_shared_dir / "checkpoints" / "flat" / f"round_{rounds}.pt"
    output_checkpoint = shared_dir / "checkpoints" / "flat" / f"round_{rounds}.pt"
    if not dry_run:
        ensure_initial_checkpoint(init_checkpoint, input_dim=input_dim)
    num_train = 0
    for hospital in HOSPITALS:
        metadata_path = partitions_dir / hospital.hospital_id / "metadata.json"
        if metadata_path.exists():
            num_train += int(json.loads(metadata_path.read_text())["num_train"])
        elif dry_run:
            num_train += 1
        else:
            raise FileNotFoundError(f"Missing partition metadata: {metadata_path}")

    run_config = {
        "level": "regional",
        "region": "flat",
        "global-round": 1,
        "init-checkpoint": runtime_init_checkpoint,
        "output-checkpoint": runtime_output_checkpoint,
        "num-server-rounds": rounds,
        "local-epochs": local_epochs,
        "batch-size": batch_size,
        "learning-rate": learning_rate,
        "input-dim": input_dim,
        "region-num-examples": num_train,
        "fraction-train": 1.0,
        "fraction-evaluate": 1.0,
        "min-train-nodes": len(HOSPITALS),
        "min-evaluate-nodes": len(HOSPITALS),
        "min-available-nodes": len(HOSPITALS),
    }
    command = ["flwr", "run", ".", "flat", "--stream", "--run-config", _run_config(run_config)]
    console.print(f"[bold]Running:[/bold] {shlex.join(command)}")
    if not dry_run:
        subprocess.run(command, check=True)
    console.print(f"[green]Flat FL checkpoint: {output_checkpoint}[/green]")


if __name__ == "__main__":
    app()
