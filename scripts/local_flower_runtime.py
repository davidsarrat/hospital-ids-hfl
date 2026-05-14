from __future__ import annotations

import os
import shlex
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ModuleNotFoundError:  # pragma: no cover - used when run as python scripts/foo.py
    from _bootstrap import bootstrap

bootstrap()

from hfl_cicids.config import REGIONS, hospitals_by_region


@dataclass(frozen=True)
class SuperLinkPorts:
    serverappio: int
    fleet: int
    control: int


@dataclass(frozen=True)
class RuntimeProcess:
    name: str
    command: list[str]
    log_path: Path
    process: subprocess.Popen[bytes]


SUPERLINK_PORTS = {
    "region_eu": SuperLinkPorts(serverappio=19091, fleet=19092, control=19093),
    "region_na": SuperLinkPorts(serverappio=29091, fleet=29092, control=29093),
    "global": SuperLinkPorts(serverappio=39091, fleet=39092, control=39093),
}

HOSPITAL_CLIENTAPP_PORTS = {
    "hospital_eu_01": 19101,
    "hospital_eu_02": 19102,
    "hospital_eu_03": 19103,
    "hospital_na_01": 29101,
    "hospital_na_02": 29102,
    "hospital_na_03": 29103,
}

GATEWAY_CLIENTAPP_PORTS = {
    "region_eu": 39101,
    "region_na": 39102,
}


def no_color_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "NO_COLOR": "1",
            "CLICOLOR": "0",
            "FORCE_COLOR": "0",
            "RICH_NO_COLOR": "1",
            "RICH_FORCE_TERMINAL": "0",
            "TERM": "dumb",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _wait_for_port(port: int, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for 127.0.0.1:{port}")


def _tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    content = path.read_text(errors="replace").splitlines()
    return "\n".join(content[-lines:])


class LocalFlowerRuntime:
    """Manage a local process-based Flower Deployment Runtime for the demo.

    Docker Compose remains useful for showing container boundaries, but a local runtime is
    much better for an executable notebook/demo loop: it starts the same Flower roles on
    localhost without rebuilding images or downloading dependencies into containers.
    """

    def __init__(self, root: Path, logs_dir: Path) -> None:
        self.root = root.resolve()
        self.logs_dir = logs_dir.resolve()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.processes: list[RuntimeProcess] = []

    def __enter__(self) -> LocalFlowerRuntime:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()

    def start(self) -> None:
        self._start_superlinks()
        self._start_serverexecs()
        self._start_hospital_nodes()
        self._start_gateway_nodes()
        # Give SuperNodes a short window to register with their parent SuperLinks before
        # flwr run submits the first ServerApp.
        time.sleep(3.0)

    def stop(self) -> None:
        for item in reversed(self.processes):
            if item.process.poll() is not None:
                continue
            item.process.terminate()

        deadline = time.monotonic() + 10.0
        for item in reversed(self.processes):
            if item.process.poll() is not None:
                continue
            remaining = max(0.1, deadline - time.monotonic())
            try:
                item.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(item.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def summary_lines(self) -> list[str]:
        lines = ["Local Flower runtime started:"]
        for region in REGIONS:
            ports = SUPERLINK_PORTS[region]
            profile = region.replace("_", "-")
            lines.append(
                f"- {profile} SuperLink: control=127.0.0.1:{ports.control}, "
                f"fleet=127.0.0.1:{ports.fleet}, serverappio=127.0.0.1:{ports.serverappio}"
            )
            for hospital in hospitals_by_region(region):
                port = HOSPITAL_CLIENTAPP_PORTS[hospital.hospital_id]
                lines.append(
                    f"  - {hospital.hospital_id} SuperNode -> {profile} fleet, "
                    f"ClientAppIo=127.0.0.1:{port}"
                )
        global_ports = SUPERLINK_PORTS["global"]
        lines.append(
            f"- global SuperLink: control=127.0.0.1:{global_ports.control}, "
            f"fleet=127.0.0.1:{global_ports.fleet}, "
            f"serverappio=127.0.0.1:{global_ports.serverappio}"
        )
        for region in REGIONS:
            port = GATEWAY_CLIENTAPP_PORTS[region]
            lines.append(
                f"  - {region} RegionGateway SuperNode -> global fleet, "
                f"ClientAppIo=127.0.0.1:{port}"
            )
        lines.append(f"Runtime logs: {self.logs_dir.relative_to(self.root)}")
        return lines

    def command_lines(self) -> list[str]:
        return [f"{item.name}: {shlex.join(item.command)}" for item in self.processes]

    def _launch(self, name: str, command: list[str]) -> None:
        log_path = self.logs_dir / f"{name}.log"
        log_handle = log_path.open("ab")
        process = subprocess.Popen(
            command,
            cwd=self.root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=no_color_env(),
            start_new_session=True,
        )
        log_handle.close()
        item = RuntimeProcess(name=name, command=command, log_path=log_path, process=process)
        self.processes.append(item)
        time.sleep(0.25)
        if process.poll() is not None:
            raise RuntimeError(
                f"{name} exited immediately with code {process.returncode}\n{_tail(log_path)}"
            )

    def _start_superlinks(self) -> None:
        for name, ports in SUPERLINK_PORTS.items():
            service = name.replace("_", "-")
            self._launch(
                f"{service}-superlink",
                [
                    "flower-superlink",
                    "--insecure",
                    "--isolation",
                    "process",
                    "--serverappio-api-address",
                    f"127.0.0.1:{ports.serverappio}",
                    "--fleet-api-address",
                    f"127.0.0.1:{ports.fleet}",
                    "--control-api-address",
                    f"127.0.0.1:{ports.control}",
                ],
            )
            _wait_for_port(ports.control)
            _wait_for_port(ports.fleet)
            _wait_for_port(ports.serverappio)

    def _start_serverexecs(self) -> None:
        for name, ports in SUPERLINK_PORTS.items():
            service = name.replace("_", "-")
            self._launch(
                f"{service}-superexec-serverapp",
                [
                    "flower-superexec",
                    "--insecure",
                    "--plugin-type",
                    "serverapp",
                    "--appio-api-address",
                    f"127.0.0.1:{ports.serverappio}",
                ],
            )

    def _start_hospital_nodes(self) -> None:
        for region in REGIONS:
            fleet_port = SUPERLINK_PORTS[region].fleet
            for hospital in hospitals_by_region(region):
                port = HOSPITAL_CLIENTAPP_PORTS[hospital.hospital_id]
                data_dir = self.root / "data" / "partitions" / hospital.hospital_id
                service = hospital.hospital_id.replace("_", "-")
                self._launch(
                    f"{service}-supernode",
                    [
                        "flower-supernode",
                        "--insecure",
                        "--superlink",
                        f"127.0.0.1:{fleet_port}",
                        "--clientappio-api-address",
                        f"127.0.0.1:{port}",
                        "--isolation",
                        "process",
                        "--node-config",
                        (
                            f'role="hospital" region="{region}" '
                            f'hospital-id="{hospital.hospital_id}" data-dir="{data_dir}"'
                        ),
                    ],
                )
                _wait_for_port(port)
                self._launch(
                    f"{service}-superexec-clientapp",
                    [
                        "flower-superexec",
                        "--insecure",
                        "--plugin-type",
                        "clientapp",
                        "--appio-api-address",
                        f"127.0.0.1:{port}",
                    ],
                )

    def _start_gateway_nodes(self) -> None:
        fleet_port = SUPERLINK_PORTS["global"].fleet
        checkpoint_root = self.root / "shared" / "checkpoints"
        for region in REGIONS:
            port = GATEWAY_CLIENTAPP_PORTS[region]
            service = region.replace("_", "-")
            self._launch(
                f"{service}-gateway-supernode",
                [
                    "flower-supernode",
                    "--insecure",
                    "--superlink",
                    f"127.0.0.1:{fleet_port}",
                    "--clientappio-api-address",
                    f"127.0.0.1:{port}",
                    "--isolation",
                    "process",
                    "--node-config",
                    f'role="region-gateway" region="{region}" checkpoint-root="{checkpoint_root}"',
                ],
            )
            _wait_for_port(port)
            self._launch(
                f"{service}-gateway-superexec-clientapp",
                [
                    "flower-superexec",
                    "--insecure",
                    "--plugin-type",
                    "clientapp",
                    "--appio-api-address",
                    f"127.0.0.1:{port}",
                ],
            )
