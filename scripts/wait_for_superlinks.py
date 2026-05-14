from __future__ import annotations

import socket
import time

import typer
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console(no_color=True, highlight=False, soft_wrap=True, width=160)


def _parse_ports(value: str) -> list[int]:
    ports = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ports:
        raise typer.BadParameter("At least one port is required.")
    return ports


def _can_connect(host: str, port: int, connect_timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=connect_timeout):
            return True
    except OSError:
        return False


@app.command()
def main(
    host: str = typer.Option("127.0.0.1", "--host"),
    ports: str = typer.Option("19093,29093,39093", "--ports"),
    timeout: float = typer.Option(120.0, "--timeout"),
    poll_interval: float = typer.Option(2.0, "--poll-interval"),
    settle_seconds: float = typer.Option(8.0, "--settle-seconds"),
) -> None:
    """Wait until the Flower SuperLink control ports accept TCP connections."""

    target_ports = _parse_ports(ports)
    deadline = time.monotonic() + timeout
    pending = set(target_ports)

    while pending and time.monotonic() < deadline:
        for port in list(pending):
            if _can_connect(host, port, connect_timeout=min(1.0, poll_interval)):
                pending.remove(port)
                console.print(f"[green]SuperLink port ready:[/green] {host}:{port}")
        if pending:
            console.print(f"[yellow]Waiting for SuperLink ports:[/yellow] {sorted(pending)}")
            time.sleep(poll_interval)

    if pending:
        raise TimeoutError(f"Timed out waiting for SuperLink ports: {sorted(pending)}")

    if settle_seconds > 0:
        console.print(f"Waiting {settle_seconds:g}s for Flower APIs to settle...")
        time.sleep(settle_seconds)


if __name__ == "__main__":
    app()
