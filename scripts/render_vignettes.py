from __future__ import annotations

import subprocess
import sys
from textwrap import dedent
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip())


def notebook(cells: list[nbf.NotebookNode]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook(cells=cells)
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    return nb


SETUP_CELL = """
from pathlib import Path
import os

ROOT = Path.cwd()
if not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
os.chdir(ROOT)
ROOT
"""


def write_and_execute(path: Path, nb: nbf.NotebookNode, timeout: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    executed = client.execute()
    path.write_text(nbf.writes(executed))
    print(f"Wrote executed notebook {path.relative_to(ROOT)}")


def build_dataset_notebook() -> nbf.NotebookNode:
    return notebook(
        [
            md(
                """
                # 00 - Dataset and Partitioning

                This vignette executes repository code to inspect the CIC-IDS2017 data pipeline.

                The dataset is **network-flow telemetry**, not medical or clinical data. The hospital
                naming convention represents healthcare organizations operating local networks. It does
                not represent patient records, diagnoses, imaging, lab results, prescriptions, or clinical
                decision support.

                Binary labels mean:

                - `0`: benign network flow
                - `1`: malicious / attack network flow
                """
            ),
            code(SETUP_CELL),
            md(
                """
                ## 1. Source files and processed metadata

                The download step is intentionally not executed from the notebook because it requires
                Kaggle credentials and a large external download. The code below still executes against
                the local repository and reports whether the raw and processed artifacts are present.
                """
            ),
            code(
                """
                import json
                from pathlib import Path
                import pandas as pd

                raw_csvs = sorted(Path("data/raw").glob("*.csv"))
                processed_meta = Path("data/processed/cicids_clean.metadata.json")

                print(f"raw CSV files: {len(raw_csvs)}")
                print("first raw files:", [path.name for path in raw_csvs[:4]])
                print(f"processed metadata exists: {processed_meta.exists()}")

                if processed_meta.exists():
                    meta = json.loads(processed_meta.read_text())
                    print(f"processed rows: {meta['rows']:,}")
                    print(f"numeric features: {meta['num_features']}")
                    attack_distribution = (
                        pd.Series(meta["attack_distribution"], name="rows")
                        .sort_values(ascending=False)
                        .rename_axis("original_label")
                        .reset_index()
                    )
                    attack_distribution.head(12)
                """
            ),
            md(
                """
                ## 2. Non-IID healthcare-network partitions

                `scripts/make_partitions.py` creates six simulated healthcare-network sites with seed
                `123`. The site names remain `hospital_*` because the demo is about hospital network
                operators, but the rows are still CIC-IDS2017 network-flow records.
                """
            ),
            code(
                """
                rows = []
                for meta_path in sorted(Path("data/partitions").glob("*/metadata.json")):
                    item = json.loads(meta_path.read_text())
                    rows.append({
                        "site_id": item["hospital_id"],
                        "region": item["region"],
                        "num_train": item["num_train"],
                        "num_val": item["num_val"],
                        "num_test": item["num_test"],
                        "num_features": item["num_features"],
                        "train_benign": item["train_label_counts"].get("0", 0),
                        "train_attack": item["train_label_counts"].get("1", 0),
                    })

                partition_df = pd.DataFrame(rows)
                partition_df
                """
            ),
            code(
                """
                import matplotlib.pyplot as plt

                if partition_df.empty:
                    print("No partition metadata found. Run: make partition SEED=123")
                else:
                    ax = (
                        partition_df
                        .set_index("site_id")[["train_benign", "train_attack"]]
                        .plot(kind="bar", stacked=True, figsize=(10, 4), color=["#0f766e", "#dc2626"])
                    )
                    ax.set_title("Training network-flow rows per simulated healthcare site")
                    ax.set_ylabel("rows")
                    ax.set_xlabel("")
                    ax.legend(["benign", "attack"], loc="upper right")
                    plt.tight_layout()
                """
            ),
            md(
                """
                ## 3. Raw-data boundary check

                After partitioning, `shared/` may contain checkpoints, metrics, and preprocessing
                metadata. It should not contain raw CSV or parquet network-flow rows.
                """
            ),
            code(
                """
                shared_data_files = sorted(
                    list(Path("shared").rglob("*.csv"))
                    + list(Path("shared").rglob("*.parquet"))
                )
                [str(path) for path in shared_data_files]
                """
            ),
        ]
    )


def build_baseline_notebook() -> nbf.NotebookNode:
    return notebook(
        [
            md(
                """
                # 01 - Baselines and Evaluation Tables

                This vignette executes lightweight inspection code for the available metric reports.
                Full centralized/local baseline training can be run from the Makefile, but the notebook
                does not silently launch expensive training. It reports which artifacts exist and plots
                the current hierarchical evaluation output.
                """
            ),
            code(SETUP_CELL),
            md("## 1. Baseline artifact availability"),
            code(
                """
                from pathlib import Path
                import pandas as pd

                reports = [
                    ("centralized_mlp", Path("reports/centralized_mlp_metrics.csv"), "make centralized"),
                    ("local_only", Path("reports/local_only_metrics.csv"), "make local"),
                    ("flat_fl", Path("reports/flat_metrics_summary.csv"), "make flat"),
                    ("hierarchical_fl", Path("reports/metrics_summary.csv"), "make demo"),
                ]
                availability = pd.DataFrame(
                    {
                        "report": name,
                        "path": str(path),
                        "exists": path.exists(),
                        "command": command,
                    }
                    for name, path, command in reports
                )
                availability
                """
            ),
            md("## 2. Current hierarchical global summary"),
            code(
                """
                summary_path = Path("reports/metrics_summary_global.csv")
                if summary_path.exists():
                    summary = pd.read_csv(summary_path)
                    summary
                else:
                    print("No global summary found. Run: make demo")
                """
            ),
            md("## 3. Per-site evaluation"),
            code(
                """
                detail_path = Path("reports/metrics_summary.csv")
                if detail_path.exists():
                    detail = pd.read_csv(detail_path)
                    display(detail[[
                        "hospital_id",
                        "region",
                        "eval_f1",
                        "eval_roc_auc",
                        "eval_auprc",
                        "eval_false_positive_rate",
                        "eval_false_negative_rate",
                        "num-examples",
                    ]])
                else:
                    print("No per-site metrics found. Run: make demo")
                """
            ),
            code(
                """
                import matplotlib.pyplot as plt

                if detail_path.exists():
                    plot_df = detail.set_index("hospital_id")[["eval_f1", "eval_roc_auc", "eval_auprc"]]
                    ax = plot_df.plot(kind="bar", figsize=(10, 4), color=["#0f766e", "#2563eb", "#9333ea"])
                    ax.set_title("Global checkpoint metrics by healthcare-network site")
                    ax.set_ylim(0, 1.05)
                    ax.set_xlabel("")
                    ax.set_ylabel("score")
                    plt.tight_layout()
                """
            ),
        ]
    )


def build_flat_notebook() -> nbf.NotebookNode:
    return notebook(
        [
            md(
                """
                # 02 - Flat Flower Baseline

                This vignette executes safe control-plane code for the one-layer Flower baseline.
                The flat baseline is a comparison point: all six healthcare-network sites connect to
                one `flat` SuperLink, instead of using regional hubs plus gateway SuperNodes.
                """
            ),
            code(SETUP_CELL),
            md("## 1. Regenerate Compose and inspect flat services"),
            code(
                """
                import subprocess
                import sys
                from pathlib import Path
                import pandas as pd
                import yaml

                flat_compose = Path("reports/docker-compose-flat-preview.yml")
                flat_compose.parent.mkdir(parents=True, exist_ok=True)
                proc = subprocess.run(
                    [sys.executable, "scripts/generate_compose.py", "--output", str(flat_compose)],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                print(proc.stdout.strip())

                compose = yaml.safe_load(flat_compose.read_text())
                flat_services = sorted(name for name in compose["services"] if name.startswith("flat-"))
                pd.DataFrame({"flat_service": flat_services})
                """
            ),
            md("## 2. Execute the flat FL dry-run command"),
            code(
                """
                cmd = [
                    sys.executable,
                    "scripts/flat_fl_baseline.py",
                    "--rounds",
                    "1",
                    "--batch-size",
                    "8192",
                    "--dry-run",
                ]
                proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
                print(proc.stdout)
                if proc.stderr:
                    print(proc.stderr)
                """
            ),
            md("## 3. Optional flat metrics"),
            code(
                """
                flat_metrics = Path("reports/flat_metrics_summary.csv")
                if flat_metrics.exists():
                    pd.read_csv(flat_metrics).head(12)
                else:
                    print("Flat metrics are not present. Run `make flat` and then evaluate the flat checkpoint to populate them.")
                """
            ),
        ]
    )


def build_hierarchical_notebook() -> nbf.NotebookNode:
    return notebook(
        [
            md(
                """
                # 03 - Hierarchical Flower Demo

                This vignette executes control-plane code and inspects the artifacts produced by
                `make demo`. The global model is produced by the global Flower SuperLink, whose clients
                are regional gateway SuperNodes. No clinical data exists in this workflow; the local rows
                are network-flow telemetry.
                """
            ),
            code(SETUP_CELL),
            md("## 1. Hierarchical dry-run commands"),
            code(
                """
                import subprocess
                import sys
                from pathlib import Path
                import json
                import pandas as pd

                cmd = [
                    sys.executable,
                    "scripts/run_hierarchical_rounds.py",
                    "--global-rounds",
                    "1",
                    "--regional-rounds",
                    "1",
                    "--batch-size",
                    "8192",
                    "--dry-run",
                ]
                proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
                print(proc.stdout)
                if proc.stderr:
                    print(proc.stderr)
                """
            ),
            md(
                """
                ## 2. Verbose training transcript

                `make demo` writes a cleaned transcript to `reports/demo_transcript.txt`. The excerpt
                below keeps the process lines: section headers, submitted Flower runs, regional/global
                phases, checkpoint writes, evaluation, predictions, and the final metrics summary.
                """
            ),
            code(
                """
                import re

                transcript = Path("reports/demo_transcript.txt")
                ansi = re.compile(r"\\x1B(?:\\[[0-?]*[ -/]*[@-~]|\\][^\\x07]*(?:\\x07|\\x1B\\\\)|[@-Z\\\\-_])")
                if transcript.exists():
                    lines = ansi.sub("", transcript.read_text(errors="replace")).splitlines()
                    interesting = [
                        line for line in lines
                        if line.startswith("=")
                        or line.startswith("Step ")
                        or line.startswith("Demo scope")
                        or line.startswith("$ ")
                        or "Global round" in line
                        or "Regional phase:" in line
                        or "Global phase:" in line
                        or "Submitting Flower run" in line
                        or "Checkpoint written:" in line
                        or "Latest global checkpoint" in line
                        or "weighted_f1" in line
                        or "hierarchical_fl" in line
                        or "Wrote site metrics" in line
                        or "Wrote predictions" in line
                    ]
                    print("\\n".join(interesting[-160:]))
                else:
                    print("No demo transcript found. Run: make demo")
                """
            ),
            md("## 3. Checkpoints and metadata from the latest demo run"),
            code(
                """
                rows = []
                for meta_path in sorted(Path("shared/checkpoints").rglob("*.metadata.json")):
                    item = json.loads(meta_path.read_text())
                    rows.append({
                        "metadata": str(meta_path),
                        "level": item.get("level", "initial"),
                        "region": item.get("region", ""),
                        "global_round": item.get("global_round", ""),
                        "num_examples": item.get("num_examples", ""),
                        "val_f1": item.get("val_f1", ""),
                        "val_roc_auc": item.get("val_roc_auc", ""),
                    })
                pd.DataFrame(rows)
                """
            ),
            md("## 4. Global evaluation and prediction sample"),
            code(
                """
                summary = Path("reports/metrics_summary_global.csv")
                predictions = Path("reports/predictions_hospital_eu_01.csv")

                if summary.exists():
                    display(pd.read_csv(summary))
                else:
                    print("No global summary found. Run: make demo")

                if predictions.exists():
                    display(pd.read_csv(predictions).head(15))
                else:
                    print("No prediction sample found. Run: make predict GLOBAL_ROUNDS=1")
                """
            ),
            code(
                """
                import matplotlib.pyplot as plt

                if summary.exists():
                    metric_cols = ["weighted_f1", "macro_f1", "weighted_roc_auc", "weighted_auprc"]
                    values = pd.read_csv(summary).iloc[0][metric_cols].astype(float)
                    ax = values.plot(kind="barh", figsize=(8, 3), color="#0f766e")
                    ax.set_xlim(0, 1.05)
                    ax.set_title("Global checkpoint summary metrics")
                    ax.set_xlabel("score")
                    plt.tight_layout()
                """
            ),
            md("## 5. Raw-data boundary check"),
            code(
                """
                data_like_files = sorted(
                    list(Path("shared").rglob("*.csv"))
                    + list(Path("shared").rglob("*.parquet"))
                )
                [str(path) for path in data_like_files]
                """
            ),
        ]
    )


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)

    notebooks = [
        (NOTEBOOKS / "00_dataset_and_partitioning.ipynb", build_dataset_notebook(), 180),
        (NOTEBOOKS / "01_centralized_baseline.ipynb", build_baseline_notebook(), 180),
        (NOTEBOOKS / "02_flat_fl_baseline.ipynb", build_flat_notebook(), 180),
        (NOTEBOOKS / "03_hierarchical_flower_demo.ipynb", build_hierarchical_notebook(), 180),
    ]
    for path, nb, timeout in notebooks:
        write_and_execute(path, nb, timeout=timeout)

    subprocess.run([sys.executable, "scripts/render_runtime_notebook.py"], cwd=ROOT, check=True)
    runtime_path = NOTEBOOKS / "04_flower_runtime_orchestration.ipynb"
    runtime_nb = nbf.read(runtime_path, as_version=4)
    write_and_execute(runtime_path, runtime_nb, timeout=180)


if __name__ == "__main__":
    main()
