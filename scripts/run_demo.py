from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import typer

try:
    from scripts.ansi import strip_ansi
    from scripts.local_flower_runtime import LocalFlowerRuntime
except ModuleNotFoundError:  # pragma: no cover - used when run as python scripts/foo.py
    from ansi import strip_ansi
    from local_flower_runtime import LocalFlowerRuntime


ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPT = ROOT / "reports" / "demo_transcript.txt"

app = typer.Typer(add_completion=False)


def _demo_env() -> dict[str, str]:
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


def _write(text: str) -> None:
    clean = strip_ansi(text).replace("\r", "\n")
    print(clean, end="", flush=True)
    with TRANSCRIPT.open("a", encoding="utf-8") as handle:
        handle.write(clean)


def _section(title: str, detail: str | None = None) -> None:
    line = "=" * 88
    body = f"\n{line}\n{title}\n{line}\n"
    if detail:
        body += f"{detail}\n"
    _write(body + "\n")


def _run(title: str, command: list[str], detail: str | None = None) -> None:
    _section(title, detail)
    _write(f"$ {shlex.join(command)}\n\n")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=_demo_env(),
    )
    assert process.stdout is not None
    for line in process.stdout:
        _write(line)
    return_code = process.wait()
    _write(f"\n[exit code: {return_code}]\n")
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _reset_demo_checkpoints() -> None:
    for relative in (
        "shared/checkpoints/global",
        "shared/checkpoints/region_eu",
        "shared/checkpoints/region_na",
    ):
        path = ROOT / relative
        if path.exists():
            shutil.rmtree(path)


@app.command()
def main(
    global_rounds: int = typer.Option(1, "--global-rounds"),
    regional_rounds: int = typer.Option(1, "--regional-rounds"),
    batch_size: int = typer.Option(8192, "--batch-size"),
    prediction_site: str = typer.Option("hospital_eu_01", "--prediction-site"),
    runtime: str = typer.Option(
        "local",
        "--runtime",
        help="'local' starts localhost Flower SuperLinks/SuperNodes; 'existing' uses running services.",
    ),
    runtime_shared_dir: Path | None = typer.Option(
        None,
        "--runtime-shared-dir",
        help="Shared directory path as seen by ServerApp/ClientApp processes.",
    ),
) -> None:
    """Run the visible end-to-end demo and write a clean transcript."""

    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    TRANSCRIPT.write_text("", encoding="utf-8")

    _section(
        "Demo scope",
        "CIC-IDS2017 is network-flow telemetry. The local nodes are simulated "
        "healthcare-network sites, not clinical or patient-data systems.",
    )
    _reset_demo_checkpoints()
    _write("Reset demo checkpoints under shared/checkpoints/{global,region_eu,region_na}.\n")
    if runtime not in {"local", "existing"}:
        raise typer.BadParameter("runtime must be 'local' or 'existing'")

    logs_dir = ROOT / "reports" / "runtime_logs"
    if runtime == "local" and logs_dir.exists():
        shutil.rmtree(logs_dir)

    effective_runtime_shared_dir = runtime_shared_dir
    if effective_runtime_shared_dir is None:
        effective_runtime_shared_dir = ROOT / "shared" if runtime == "local" else Path("/shared")

    runtime_context = (
        LocalFlowerRuntime(ROOT, logs_dir) if runtime == "local" else nullcontext(None)
    )
    with runtime_context as flower_runtime:
        if isinstance(flower_runtime, LocalFlowerRuntime):
            _section(
                "Step 1/6 - Start local Flower deployment runtime",
                "This starts real SuperLinks, SuperNodes, and SuperExec processes on localhost.",
            )
            _write("\n".join(flower_runtime.summary_lines()) + "\n\n")
            _write("Launched commands:\n")
            for line in flower_runtime.command_lines():
                _write(f"- {line}\n")
            _write("\n")
        else:
            _section(
                "Step 1/6 - Use existing Flower deployment runtime",
                "The demo assumes SuperLinks, SuperNodes, and SuperExec services are already running.",
            )

        _run(
            "Step 2/6 - Hierarchical Flower training",
            [
                sys.executable,
                "scripts/run_hierarchical_rounds.py",
                "--global-rounds",
                str(global_rounds),
                "--regional-rounds",
                str(regional_rounds),
                "--batch-size",
                str(batch_size),
                "--runtime-shared-dir",
                str(effective_runtime_shared_dir),
            ],
            "Regional Flower federations train first; the global Flower federation then "
            "aggregates RegionGateway SuperNodes.",
        )

        checkpoint = f"shared/checkpoints/global/round_{global_rounds}.pt"
        _run(
            "Step 3/6 - Evaluate the global checkpoint",
            [
                sys.executable,
                "scripts/evaluate_global_model.py",
                "--checkpoint",
                checkpoint,
                "--batch-size",
                str(batch_size),
            ],
            "The evaluator loads the produced global model and scores it against each site test split.",
        )

        _run(
            "Step 4/6 - Generate row-level predictions",
            [
                sys.executable,
                "scripts/predict_with_checkpoint.py",
                "--checkpoint",
                checkpoint,
                "--hospital-id",
                prediction_site,
                "--output",
                f"reports/predictions_{prediction_site}.csv",
            ],
            "This proves the checkpoint is usable for ordinary benign-vs-attack inference.",
        )

        _run(
            "Step 5/6 - Execute educational notebooks",
            [sys.executable, "scripts/render_vignettes.py"],
            "The notebooks execute real repository code and save their outputs in notebooks/.",
        )

        _section("Global metrics summary")
        metrics_path = ROOT / "reports" / "metrics_summary_global.csv"
        _write(
            metrics_path.read_text(encoding="utf-8")
            if metrics_path.exists()
            else "No summary found.\n"
        )

        _run(
            "Step 6/6 - Rebuild GitHub Pages handbook",
            [sys.executable, "scripts/build_pages.py"],
            "The handbook renders snippets, notebook outputs, charts, logs, metrics, and predictions.",
        )


if __name__ == "__main__":
    app()
