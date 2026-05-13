from __future__ import annotations

from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ModuleNotFoundError:  # pragma: no cover - used when run as python scripts/foo.py
    from _bootstrap import bootstrap

bootstrap()

import typer
from rich.console import Console

from hfl_cicids.config import HOSPITALS, REGIONS, hospitals_by_region

app = typer.Typer(add_completion=False)
console = Console()

SERVERAPP_BUILD = """\
      context: .
      dockerfile_inline: |
        FROM flwr/superexec:${FLWR_VERSION:-1.29.0}
        USER root
        RUN apt-get update \\
          && apt-get -y --no-install-recommends install build-essential \\
          && rm -rf /var/lib/apt/lists/*
        USER app
        WORKDIR /app
        COPY --chown=app:app pyproject.toml README.md ./
        COPY --chown=app:app hfl_cicids ./hfl_cicids
        COPY --chown=app:app scripts ./scripts
        RUN python -m pip install -U --no-cache-dir \\
          --index-url https://download.pytorch.org/whl/cpu \\
          "torch>=2.3,<3.0"
        RUN python -m pip install -U --no-cache-dir .
        ENTRYPOINT [\"flower-superexec\"]
"""

CLIENTAPP_BUILD = SERVERAPP_BUILD


def _superlink(name: str, port: int) -> str:
    return f"""\
  {name}-superlink:
    image: flwr/superlink:${{FLWR_VERSION:-1.29.0}}
    command:
      - --insecure
      - --isolation
      - process
    ports:
      - \"{port}:9093\"
"""


def _serverexec(name: str) -> str:
    return f"""\
  {name}-superexec-serverapp:
    build:
{SERVERAPP_BUILD.rstrip()}
    command:
      - --insecure
      - --plugin-type
      - serverapp
      - --appio-api-address
      - {name}-superlink:9091
    volumes:
      - ./shared:/shared
    depends_on:
      - {name}-superlink
"""


def _hospital_supernode(hospital_id: str, region: str) -> str:
    service = hospital_id.replace("_", "-")
    region_service = region.replace("_", "-")
    return f"""\
  {service}-supernode:
    image: flwr/supernode:${{FLWR_VERSION:-1.29.0}}
    command:
      - --insecure
      - --superlink
      - {region_service}-superlink:9092
      - --clientappio-api-address
      - 0.0.0.0:9094
      - --isolation
      - process
      - --node-config
      - 'role=\"hospital\" region=\"{region}\" hospital-id=\"{hospital_id}\" data-dir=\"/data\"'
    depends_on:
      - {region_service}-superlink

  {service}-superexec-clientapp:
    build:
{CLIENTAPP_BUILD.rstrip()}
    command:
      - --insecure
      - --plugin-type
      - clientapp
      - --appio-api-address
      - {service}-supernode:9094
    volumes:
      - ./data/partitions/{hospital_id}:/data:ro
      - ./shared:/shared
    depends_on:
      - {service}-supernode
"""


def _gateway_supernode(region: str) -> str:
    service = region.replace("_", "-")
    return f"""\
  {service}-gateway-supernode:
    image: flwr/supernode:${{FLWR_VERSION:-1.29.0}}
    command:
      - --insecure
      - --superlink
      - global-superlink:9092
      - --clientappio-api-address
      - 0.0.0.0:9094
      - --isolation
      - process
      - --node-config
      - 'role=\"region-gateway\" region=\"{region}\" checkpoint-root=\"/shared/checkpoints\"'
    volumes:
      - ./shared:/shared
    depends_on:
      - global-superlink

  {service}-gateway-superexec-clientapp:
    build:
{CLIENTAPP_BUILD.rstrip()}
    command:
      - --insecure
      - --plugin-type
      - clientapp
      - --appio-api-address
      - {service}-gateway-supernode:9094
    volumes:
      - ./shared:/shared
    depends_on:
      - {service}-gateway-supernode
"""


def _flat_supernodes() -> str:
    chunks = []
    for hospital in HOSPITALS:
        service = f"flat-{hospital.hospital_id.replace('_', '-')}"
        chunks.append(
            f"""\
  {service}-supernode:
    image: flwr/supernode:${{FLWR_VERSION:-1.29.0}}
    command:
      - --insecure
      - --superlink
      - flat-superlink:9092
      - --clientappio-api-address
      - 0.0.0.0:9094
      - --isolation
      - process
      - --node-config
      - 'role=\"hospital\" region=\"flat\" hospital-id=\"{hospital.hospital_id}\" data-dir=\"/data\"'
    depends_on:
      - flat-superlink

  {service}-superexec-clientapp:
    build:
{CLIENTAPP_BUILD.rstrip()}
    command:
      - --insecure
      - --plugin-type
      - clientapp
      - --appio-api-address
      - {service}-supernode:9094
    volumes:
      - ./data/partitions/{hospital.hospital_id}:/data:ro
      - ./shared:/shared
    depends_on:
      - {service}-supernode
"""
        )
    return "\n".join(chunks)


def render_compose(include_flat: bool = True) -> str:
    chunks = [
        "services:\n",
        _superlink("global", 39093),
        _serverexec("global"),
    ]
    for region in REGIONS:
        service = region.replace("_", "-")
        port = 19093 if region == "region_eu" else 29093
        chunks.extend([_superlink(service, port), _serverexec(service)])
        for hospital in hospitals_by_region(region):
            chunks.append(_hospital_supernode(hospital.hospital_id, hospital.region))
        chunks.append(_gateway_supernode(region))

    if include_flat:
        chunks.extend([_superlink("flat", 49093), _serverexec("flat"), _flat_supernodes()])

    return "\n".join(chunks)


@app.command()
def main(
    output: Path = typer.Option(Path("docker-compose.yml"), "--output", "-o"),
    include_flat: bool = typer.Option(True, "--include-flat/--hierarchical-only"),
) -> None:
    """Generate the Docker Compose topology for regional, global, and flat Flower runs."""

    output.write_text(render_compose(include_flat=include_flat))
    console.print(f"[green]Wrote {output}[/green]")


if __name__ == "__main__":
    app()
