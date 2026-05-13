from __future__ import annotations

from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ModuleNotFoundError:  # pragma: no cover - used when run as python scripts/foo.py
    from _bootstrap import bootstrap

bootstrap()

import typer
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()


FLOWER_CONFIG = """[superlink]
default = "region-eu"

[superlink.region-eu]
address = "127.0.0.1:19093"
insecure = true

[superlink.region-na]
address = "127.0.0.1:29093"
insecure = true

[superlink.global]
address = "127.0.0.1:39093"
insecure = true

[superlink.flat]
address = "127.0.0.1:49093"
insecure = true
"""


@app.command()
def main(
    output: Path = typer.Option(
        Path.home() / ".flwr" / "config.toml",
        "--output",
        "-o",
        help="Flower CLI config path.",
    )
) -> None:
    """Write local Flower SuperLink profiles for the Docker Compose topology."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(FLOWER_CONFIG)
    console.print(f"[green]Wrote Flower profiles to {output}[/green]")


if __name__ == "__main__":
    app()
