from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import nbformat as nbf
import pandas as pd
import yaml

try:
    from scripts.ansi import strip_ansi
except ModuleNotFoundError:  # pragma: no cover - used when run as python scripts/foo.py
    from ansi import strip_ansi

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "04_flower_runtime_orchestration.ipynb"


def sanitize(text: str) -> str:
    return text.replace(str(Path.home()), "~").replace(str(ROOT), ".")


def run(command: list[str], timeout: int = 30) -> str:
    proc = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    text = proc.stdout
    if proc.stderr:
        text = f"{text}\n{proc.stderr}" if text else proc.stderr
    return sanitize(strip_ansi(text)).strip()


def stream_cell(source: str, output: str, execution_count: int) -> nbf.NotebookNode:
    cell = nbf.v4.new_code_cell(source)
    cell["execution_count"] = execution_count
    if output:
        cell["outputs"] = [
            nbf.v4.new_output(
                output_type="stream",
                name="stdout",
                text=output if output.endswith("\n") else f"{output}\n",
            )
        ]
    return cell


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def service_role(name: str) -> str:
    if name.endswith("-superlink"):
        return "SuperLink"
    if name.endswith("-superexec-serverapp"):
        return "SuperExec ServerApp"
    if name.endswith("-supernode"):
        return "SuperNode"
    if name.endswith("-superexec-clientapp"):
        return "SuperExec ClientApp"
    return "service"


def list_arg(args: list[str], flag: str) -> str:
    try:
        idx = args.index(flag)
    except ValueError:
        return ""
    if idx + 1 >= len(args):
        return ""
    return args[idx + 1]


def compose_table() -> str:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    rows: list[dict[str, Any]] = []
    for name, service in sorted(compose["services"].items()):
        if name.startswith("flat-"):
            continue
        command = service.get("command", [])
        if not isinstance(command, list):
            command = str(command).split()
        rows.append(
            {
                "service": name,
                "role": service_role(name),
                "superlink_target": list_arg(command, "--superlink"),
                "appio_target": list_arg(command, "--appio-api-address"),
                "node_config": list_arg(command, "--node-config"),
                "volumes": ", ".join(service.get("volumes", [])),
            }
        )
    df = pd.DataFrame(rows)
    return df.to_string(index=False, max_colwidth=80)


def checkpoint_metadata() -> str:
    lines: list[str] = []
    for path in sorted((ROOT / "shared" / "checkpoints").rglob("*.metadata.json")):
        rel = path.relative_to(ROOT)
        lines.append(str(rel))
        lines.append(json.dumps(json.loads(path.read_text()), indent=2, sort_keys=True))
    return "\n".join(lines) if lines else "No checkpoint metadata found."


def global_metrics() -> str:
    summary = ROOT / "reports" / "metrics_summary_global.csv"
    detail = ROOT / "reports" / "metrics_summary.csv"
    if summary.exists():
        return pd.read_csv(summary).to_string(index=False)
    if detail.exists():
        return pd.read_csv(detail).head(12).to_string(index=False)
    return "No evaluation report found. Run make eval after training."


def shared_raw_files() -> str:
    files = sorted(
        list((ROOT / "shared").rglob("*.csv"))
        + list((ROOT / "shared").rglob("*.parquet"))
    )
    if not files:
        return "[]"
    return "\n".join(str(path.relative_to(ROOT)) for path in files)


def logs_snapshot() -> str:
    services = [
        "hospital-eu-01-supernode",
        "region-eu-gateway-supernode",
        "region-eu-superlink",
        "global-superlink",
    ]
    sections = []
    for service in services:
        output = run(["docker", "compose", "logs", "--tail=8", service], timeout=20)
        lines = output.splitlines()
        if service.endswith("superlink"):
            lines = [line for line in lines if "PullMessages" in line][-5:]
        sections.append(
            f"$ docker compose logs --tail=8 {service}\n" + "\n".join(lines)
        )
    return "\n\n".join(sections)


def build_notebook() -> nbf.NotebookNode:
    now = datetime.now().isoformat(timespec="seconds")
    dry_run = run(
        [
            "python",
            "scripts/run_hierarchical_rounds.py",
            "--global-rounds",
            "1",
            "--regional-rounds",
            "1",
            "--batch-size",
            "8192",
            "--dry-run",
        ],
        timeout=60,
    )
    docker_ps = run(
        [
            "docker",
            "compose",
            "ps",
            "--format",
            "table {{.Service}}\\t{{.State}}\\t{{.Ports}}",
        ],
        timeout=20,
    )
    flower_config = run(["flwr", "config", "list"], timeout=20)

    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb["cells"] = [
        md(
            "# 04 - Flower Runtime Orchestration\n\n"
            "Rendered snapshot of how the deployment runtime is actuated in this demo. "
            f"Captured at `{now}` from the local Docker/Flower topology.\n\n"
            "The goal of this notebook is not to retrain the model. It shows which "
            "SuperLinks, SuperNodes and SuperExec services are running, what each one "
            "connects to, and which `flwr run` calls drive the regional and global layers."
        ),
        md(
            "## 1. Running Services\n\n"
            "There are three Flower federations running at the same time: "
            "`region-eu`, `region-na`, and `global`. Healthcare-site SuperNodes connect to "
            "regional SuperLinks. Region gateway SuperNodes connect only to the global "
            "SuperLink."
        ),
        stream_cell(
            "!docker compose ps --format 'table {{.Service}}\\t{{.State}}\\t{{.Ports}}'",
            docker_ps,
            1,
        ),
        md(
            "## 2. Compose Roles\n\n"
            "`SuperLink` services expose the Flower APIs. `SuperExec ServerApp` services "
            "execute the server application submitted through `flwr run`. Healthcare-site "
            "`SuperNode` services receive tasks from their regional SuperLink, while "
            "gateway SuperNodes receive tasks from the global SuperLink. ClientApp "
            "SuperExec services execute `hfl_cicids.client_app` next to each SuperNode."
        ),
        stream_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import yaml\n\n"
            "def service_role(name):\n"
            "    if name.endswith('-superlink'):\n"
            "        return 'SuperLink'\n"
            "    if name.endswith('-superexec-serverapp'):\n"
            "        return 'SuperExec ServerApp'\n"
            "    if name.endswith('-supernode'):\n"
            "        return 'SuperNode'\n"
            "    if name.endswith('-superexec-clientapp'):\n"
            "        return 'SuperExec ClientApp'\n"
            "    return 'service'\n\n"
            "def list_arg(args, flag):\n"
            "    return args[args.index(flag) + 1] if flag in args else ''\n\n"
            "compose = yaml.safe_load(Path('docker-compose.yml').read_text())\n"
            "rows = []\n"
            "for name, service in sorted(compose['services'].items()):\n"
            "    if name.startswith('flat-'):\n"
            "        continue\n"
            "    command = service.get('command', [])\n"
            "    rows.append({\n"
            "        'service': name,\n"
            "        'role': service_role(name),\n"
            "        'superlink_target': list_arg(command, '--superlink'),\n"
            "        'appio_target': list_arg(command, '--appio-api-address'),\n"
            "        'node_config': list_arg(command, '--node-config'),\n"
            "        'volumes': ', '.join(service.get('volumes', [])),\n"
            "    })\n"
            "pd.DataFrame(rows)",
            compose_table(),
            2,
        ),
        md(
            "## 3. Flower Profiles\n\n"
            "The profile names are what the orchestrator passes to `flwr run . <profile>`. "
            "Each profile points to one SuperLink control endpoint."
        ),
        stream_cell("!flwr config list", flower_config, 3),
        md(
            "## 4. Actuating the Federations\n\n"
            "`scripts/run_hierarchical_rounds.py` is the control plane for the demo. "
            "For each global round it submits one regional Flower run per region, then "
            "one global Flower run where the clients are the gateway SuperNodes."
        ),
        stream_cell(
            "!python scripts/run_hierarchical_rounds.py --global-rounds 1 "
            "--regional-rounds 1 --batch-size 8192 --dry-run",
            dry_run,
            4,
        ),
        md(
            "## 5. Runtime Logs\n\n"
            "The healthcare-site SuperNode receives train/evaluate messages from its regional "
            "SuperLink. The gateway SuperNode receives a train message from the global "
            "SuperLink and returns the regional checkpoint as a model update."
        ),
        stream_cell(
            "!docker compose logs --tail=12 hospital-eu-01-supernode "
            "region-eu-gateway-supernode region-eu-superlink global-superlink",
            logs_snapshot(),
            5,
        ),
        md(
            "## 6. Checkpoint Metadata\n\n"
            "Regional checkpoints carry the regional training-example count used by "
            "global weighted FedAvg. The global checkpoint is produced by the global "
            "Flower federation."
        ),
        stream_cell(
            "from pathlib import Path\n"
            "import json\n\n"
            "for path in sorted(Path('shared/checkpoints').rglob('*.metadata.json')):\n"
            "    print(path)\n"
            "    print(json.dumps(json.loads(path.read_text()), indent=2, sort_keys=True))",
            checkpoint_metadata(),
            6,
        ),
        md(
            "## 7. Evaluation Snapshot\n\n"
            "These metrics are from the latest rendered local run. They are evidence that "
            "the global checkpoint can be loaded and evaluated across the site test splits."
        ),
        stream_cell(
            "import pandas as pd\n"
            "pd.read_csv('reports/metrics_summary_global.csv')",
            global_metrics(),
            7,
        ),
        md(
            "## 8. Privacy Boundary Check\n\n"
            "`shared/` is allowed to contain checkpoints, metrics and preprocessing "
            "metadata. It should not contain site CSV/parquet network-flow rows."
        ),
        stream_cell(
            "from pathlib import Path\n"
            "sorted(list(Path('shared').rglob('*.csv')) + list(Path('shared').rglob('*.parquet')))",
            shared_raw_files(),
            8,
        ),
    ]
    return nb


def main() -> None:
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbf.validate(notebook)
    NOTEBOOK.write_text(nbf.writes(notebook))
    print(f"Wrote {NOTEBOOK.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
