from __future__ import annotations

import subprocess
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ModuleNotFoundError:  # pragma: no cover - used when run as python scripts/foo.py
    from _bootstrap import bootstrap

bootstrap()

import typer
from rich.console import Console

from hfl_cicids.config import DEFAULT_RAW_DIR

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    dataset: str = typer.Option(
        "chethuhn/network-intrusion-dataset",
        "--dataset",
        help="Kaggle dataset slug.",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_RAW_DIR,
        "--output-dir",
        "-o",
        help="Directory for downloaded raw files.",
    ),
    unzip: bool = typer.Option(True, "--unzip/--no-unzip", help="Unzip after download."),
) -> None:
    """Download the Kaggle CIC-IDS2017 mirror."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(output_dir)]
    if unzip:
        cmd.append("--unzip")

    console.print(f"[bold]Running:[/bold] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    console.print(f"[green]Downloaded dataset into {output_dir}[/green]")


if __name__ == "__main__":
    app()
