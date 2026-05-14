from __future__ import annotations

import html
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import nbformat
from nbformat.validator import MissingIDFieldWarning


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EPISODES_DIR = DOCS / "episodes"
RENDERS_DIR = DOCS / "renders"
REPO_URL = "https://github.com/davidsarrat/hospital-ids-hfl"

warnings.filterwarnings("ignore", category=MissingIDFieldWarning)


@dataclass(frozen=True)
class Page:
    slug: str
    title: str
    summary: str
    body: str


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def p(text: str) -> str:
    return f"<p>{text}</p>"


def h2(text: str) -> str:
    return f"<h2>{esc(text)}</h2>"


def h3(text: str) -> str:
    return f"<h3>{esc(text)}</h3>"


def ul(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def ol(items: list[str]) -> str:
    return "<ol>" + "".join(f"<li>{item}</li>" for item in items) + "</ol>"


def cmd(text: str) -> str:
    return f"<pre class=\"command\"><code>{esc(text.strip())}</code></pre>"


def code(text: str, caption: str = "") -> str:
    cap = f"<figcaption>{esc(caption)}</figcaption>" if caption else ""
    return f"<figure>{cap}<pre><code>{esc(text.rstrip())}</code></pre></figure>"


def callout(title: str, body: str) -> str:
    return f"<aside class=\"callout\"><strong>{esc(title)}</strong>{p(body)}</aside>"


def table(headers: list[str], rows: list[list[str]]) -> str:
    header_html = "".join(f"<th>{esc(header)}</th>" for header in headers)
    row_html = ""
    for row in rows:
        row_html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{row_html}</tbody></table>"


def snippet(path: str, start: int, end: int, caption: str | None = None) -> str:
    file_path = ROOT / path
    lines = file_path.read_text().splitlines()
    selected = lines[start - 1 : end]
    numbered = "\n".join(f"{idx:>4}  {line}" for idx, line in enumerate(selected, start=start))
    label = caption or f"{path}:{start}-{end}"
    source_link = f"{REPO_URL}/blob/main/{path}#L{start}-L{end}"
    return (
        f"<figure><figcaption><a href=\"{esc(source_link)}\">{esc(label)}</a></figcaption>"
        f"<pre><code>{esc(numbered)}</code></pre></figure>"
    )


def rel(prefix: str, path: str) -> str:
    return f"{prefix}{path}"


def nav_html(active: str, prefix: str) -> str:
    links = [f"<a class=\"{'active' if active == 'index' else ''}\" href=\"{rel(prefix, 'index.html')}\">Home</a>"]
    for page in EPISODES:
        cls = "active" if page.slug == active else ""
        links.append(
            f"<a class=\"{cls}\" href=\"{rel(prefix, f'episodes/{page.slug}.html')}\">"
            f"{esc(page.title)}</a>"
        )
    links.append(
        f"<a href=\"{rel(prefix, 'renders/04_flower_runtime_orchestration.html')}\">Render notebook 04</a>"
    )
    return "\n".join(links)


def render_shell(title: str, body: str, active: str, prefix: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)} - hospital-ids-hfl</title>
  <link rel="stylesheet" href="{rel(prefix, 'styles.css')}">
</head>
<body>
  <header class="topbar">
    <a class="brand" href="{rel(prefix, 'index.html')}">hospital-ids-hfl handbook</a>
    <a class="repo" href="{REPO_URL}">GitHub</a>
  </header>
  <div class="layout">
    <nav class="sidebar" aria-label="Book index">
      {nav_html(active, prefix)}
    </nav>
    <main class="content">
      {body}
    </main>
  </div>
</body>
</html>
"""


def hero(title: str, subtitle: str) -> str:
    return f"<section class=\"hero\"><p class=\"eyebrow\">Handbook</p><h1>{esc(title)}</h1><p>{subtitle}</p></section>"


def page_intro(page: Page) -> str:
    return hero(page.title, page.summary)


def write_page(page: Page, index: int) -> None:
    prev_link = ""
    next_link = ""
    if index > 0:
        prev_page = EPISODES[index - 1]
        prev_link = f"<a href=\"{prev_page.slug}.html\">Previous: {esc(prev_page.title)}</a>"
    if index < len(EPISODES) - 1:
        next_page = EPISODES[index + 1]
        next_link = f"<a href=\"{next_page.slug}.html\">Next: {esc(next_page.title)}</a>"
    pager = f"<div class=\"pager\">{prev_link}<span></span>{next_link}</div>"
    body = page_intro(page) + page.body + pager
    (EPISODES_DIR / f"{page.slug}.html").write_text(render_shell(page.title, body, page.slug, "../"))


def make_index() -> str:
    rows = []
    for idx, page in enumerate(EPISODES):
        rows.append(
            [
                esc(f"{idx:02d}"),
                f"<a href=\"episodes/{page.slug}.html\">{esc(page.title)}</a>",
                esc(page.summary),
            ]
        )
    body = hero(
        "Deployment handbook: Hierarchical Flower with CIC-IDS2017",
        "A from-scratch guide to understand and run the demo: data, Flower nodes, gateways, "
        "orchestration, evaluation, and privacy boundaries.",
    )
    body += callout(
        "Fast path",
        "If you only want to inspect the infrastructure, start with episode 00 and open the "
        "rendered notebook 04. If you want to reproduce everything, follow the episodes in order.",
    )
    body += h2("Episode roadmap")
    body += table(["#", "Episode", "What you learn"], rows)
    body += h2("HTML Renders")
    body += ul(
        [
            '<a href="renders/04_flower_runtime_orchestration.html">Rendered notebook 04: '
            "Flower Runtime Orchestration</a>",
            '<a href="renders/03_hierarchical_flower_demo.html">Notebook 03: hierarchical demo</a>',
            '<a href="renders/00_dataset_and_partitioning.html">Notebook 00: dataset and partitioning</a>',
        ]
    )
    body += h2("Full command path")
    body += cmd(
        """
python -m pip install -e .
make data
make partition SEED=123
make compose
make flower-config
make up
make train GLOBAL_ROUNDS=3 REGIONAL_ROUNDS=2
make eval GLOBAL_ROUNDS=3
make render-runtime-notebook
make pages
"""
    )
    body += h2("Direct demo")
    body += p(
        "For a compact run that starts the topology, launches one hierarchical round, "
        "evaluates the checkpoint, and prints the metrics summary at the end:"
    )
    body += cmd("make demo")
    return render_shell("Home", body, "index", "")


EPISODES: list[Page] = [
    Page(
        slug="00-roadmap",
        title="00 - Mental model and topology",
        summary="The complete architecture before touching commands: hospitals, regions, gateways, and the global hub.",
        body=(
            h2("Demo objective")
            + p(
                "This project simulates hospitals collaborating to train an intrusion detector "
                "on CIC-IDS2017 network-flow data. Each hospital keeps its local partition. "
                "Regions aggregate hospital models, and the global layer aggregates regional models."
            )
            + code(
                """
Hospital SuperNodes
  -> region-eu / region-na SuperLinks
    -> regional checkpoints
      -> RegionGateway SuperNodes
        -> global SuperLink
          -> global checkpoint
""",
                "Three-layer topology",
            )
            + h2("Roadmap")
            + ol(
                [
                    "Prepare the environment, Kaggle CLI, and dependencies.",
                    "Download and clean CIC-IDS2017 without committing raw data.",
                    "Create six non-IID partitions with seed 123.",
                    "Start SuperLinks, SuperNodes, and SuperExecs with Docker Compose.",
                    "Run regional federations with hospitals as clients.",
                    "Run the global federation with regional gateways as clients.",
                    "Evaluate the global checkpoint and review privacy boundaries.",
                ]
            )
            + h2("Code map")
            + table(
                ["Path", "Responsibility"],
                [
                    ["<code>hfl_cicids/config.py</code>", "Simulated topology: hospitals, regions, paths, and checkpoints."],
                    ["<code>hfl_cicids/task.py</code>", "PyTorch dataset, MLP, and local training loop."],
                    ["<code>hfl_cicids/client_app.py</code>", "ClientApp for hospital nodes and regional gateway nodes."],
                    ["<code>hfl_cicids/server_app.py</code>", "Regional/global ServerApp with FedAvg."],
                    ["<code>hfl_cicids/checkpointing.py</code>", "Loads and saves Flower/PyTorch checkpoints."],
                    ["<code>hfl_cicids/metrics.py</code>", "Classification metrics and error rates."],
                    ["<code>scripts/download_kaggle.py</code>", "Reproducible download from the Kaggle mirror."],
                    ["<code>scripts/prepare_cicids.py</code>", "Centralized CSV cleaning into processed parquet."],
                    ["<code>scripts/make_partitions.py</code>", "Non-IID partitioning with seed and train-only scaler."],
                    ["<code>scripts/generate_compose.py</code>", "Generates SuperLinks, SuperNodes, and SuperExecs."],
                    ["<code>scripts/configure_flower_profiles.py</code>", "Writes local Flower CLI profiles."],
                    ["<code>scripts/run_hierarchical_rounds.py</code>", "Orchestrator that calls <code>flwr run</code>."],
                    ["<code>scripts/evaluate_global_model.py</code>", "Evaluates checkpoints against each hospital test split."],
                    ["<code>scripts/centralized_mlp_baseline.py</code>", "Non-federated centralized baseline."],
                    ["<code>scripts/local_only_baseline.py</code>", "Per-hospital baseline without collaboration."],
                    ["<code>scripts/flat_fl_baseline.py</code>", "Flat Flower baseline with all six hospitals."],
                    ["<code>scripts/render_runtime_notebook.py</code>", "Regenerates the rendered runtime notebook."],
                    ["<code>scripts/build_pages.py</code>", "Builds this handbook and HTML renders in <code>docs/</code>."],
                    ["<code>docs/</code>", "Static GitHub Pages book."],
                ],
            )
        ),
    ),
    Page(
        slug="01-setup",
        title="01 - Setup from scratch",
        summary="How to prepare Python, dependencies, Kaggle credentials, and the main targets.",
        body=(
            h2("Prerequisites")
            + ul(
                [
                    "Python 3.11 or another version compatible with Flower 1.29.",
                    "Docker Desktop or Docker Engine with Compose.",
                    "Kaggle CLI credentials stored outside the repository.",
                    "GitHub CLI only if you want to publish or configure Pages from the terminal.",
                ]
            )
            + h2("Installation")
            + cmd(
                """
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
"""
            )
            + callout(
                "Credentials",
                "Do not put real tokens in Git. Use environment variables or ~/.kaggle/kaggle.json. "
                "The repository ignores data, checkpoints, and common credential files.",
            )
            + cmd(
                """
export KAGGLE_API_TOKEN="<kaggle_token>"
make data
make partition SEED=123
"""
            )
            + h2("Main targets")
            + snippet("Makefile", 1, 55)
            + h2("Direct demo")
            + p(
                "Once data and partitions exist, <code>make demo</code> starts Docker, runs one "
                "global round with one regional round, evaluates, and refreshes the renders."
            )
            + cmd("make demo")
            + h2("Package dependencies")
            + snippet("pyproject.toml", 15, 28)
        ),
    ),
    Page(
        slug="02-data-pipeline",
        title="02 - Reproducible data pipeline",
        summary="How CIC-IDS2017 CSVs become six simulated hospital partitions with a non-IID split.",
        body=(
            h2("Cleaning")
            + p(
                "The cleaning step concatenates the CSVs, normalizes column names, finds the "
                "Label column, maps BENIGN to 0 and every attack to 1, drops metadata/leakage "
                "columns, and keeps numeric features."
            )
            + snippet("scripts/prepare_cicids.py", 54, 122)
            + h2("Non-IID split with seed 123")
            + p(
                "Partitioning assigns attack groups to hospitals with biased weights. This makes "
                "each hospital see a different distribution, which is the interesting case for FL."
            )
            + snippet("scripts/make_partitions.py", 31, 72, "Attack grouping and hospital preferences")
            + snippet("scripts/make_partitions.py", 112, 128, "Dirichlet assignment by group")
            + h2("Train-only scaling")
            + p(
                "The MVP demo fits imputation and standardization from the simulated train splits. "
                "It then transforms train, validation, and test for every hospital with those values."
            )
            + snippet("scripts/make_partitions.py", 167, 193)
            + h2("Expected output")
            + code(
                """
data/partitions/hospital_eu_01/train.parquet
data/partitions/hospital_eu_01/val.parquet
data/partitions/hospital_eu_01/test.parquet
data/partitions/hospital_eu_01/metadata.json
...
shared/preprocessing/scaler.json
""",
                "Local artifacts ignored by Git",
            )
        ),
    ),
    Page(
        slug="03-flower-runtime",
        title="03 - Flower Runtime: SuperLink, SuperNode, and SuperExec",
        summary="What each Flower Deployment Runtime component does and how it connects in Docker.",
        body=(
            h2("Roles")
            + table(
                ["Component", "In this demo"],
                [
                    ["SuperLink", "The hub of one federation. There is one per region and one global hub."],
                    ["SuperNode", "A client connected to a SuperLink. It can be a hospital or a gateway."],
                    ["SuperExec ServerApp", "The process that executes <code>hfl_cicids.server_app</code>."],
                    ["SuperExec ClientApp", "The process that executes <code>hfl_cicids.client_app</code> next to a SuperNode."],
                ],
            )
            + h2("SuperLink and ServerApp")
            + snippet("scripts/generate_compose.py", 43, 71)
            + h2("Hospital SuperNode")
            + p(
                "The hospital connects to its regional SuperLink, and its ClientApp mounts only "
                "its own local folder at <code>/data:ro</code>."
            )
            + snippet("scripts/generate_compose.py", 74, 107)
            + h2("RegionGateway SuperNode")
            + p(
                "The gateway connects to the global SuperLink and mounts <code>/shared</code>, not "
                "<code>/data</code>. This lets it return regional checkpoints without reading raw rows."
            )
            + snippet("scripts/generate_compose.py", 110, 143)
            + h2("Real render")
            + p(
                'The rendered notebook shows running containers and message logs. '
                '<a href="../renders/04_flower_runtime_orchestration.html">Open notebook 04 HTML render</a>.'
            )
        ),
    ),
    Page(
        slug="04-model-clientapp",
        title="04 - Model and hospital ClientApp",
        summary="How local parquet is loaded, the MLP is trained, and an update is returned to Flower.",
        body=(
            h2("Dataset and MLP")
            + p(
                "The main model is intentionally simple so FedAvg remains transparent: "
                "a tabular MLP with one logit output and BCEWithLogitsLoss."
            )
            + snippet("hfl_cicids/task.py", 12, 63)
            + h2("Local training")
            + snippet("hfl_cicids/task.py", 88, 131)
            + h2("Hospital ClientApp")
            + p(
                "When the node role is hospital, Flower sends initial arrays. The ClientApp loads "
                "those weights, trains on <code>/data/train.parquet</code>, evaluates on validation, "
                "and returns arrays plus metrics. The <code>num-examples</code> key enables weighted FedAvg."
            )
            + snippet("hfl_cicids/client_app.py", 32, 84)
        ),
    ),
    Page(
        slug="05-serverapp-fedavg",
        title="05 - Regional ServerApp and FedAvg",
        summary="How the regional server aggregates hospitals and writes regional checkpoints.",
        body=(
            h2("Shared entry point")
            + p(
                "The ServerApp reads <code>level</code> from the run config. This lets the same "
                "component serve a regional federation or the global federation."
            )
            + snippet("hfl_cicids/server_app.py", 16, 41)
            + h2("Regional federation")
            + p(
                "Each region uses FedAvg with <code>weighted_by_key='num-examples'</code>. "
                "The clients are the hospitals connected to the regional SuperLink."
            )
            + snippet("hfl_cicids/server_app.py", 44, 85)
            + h2("Formula")
            + code(
                """
N_r = sum_h n_h
theta_r = sum_h (n_h / N_r) * theta_h
""",
                "Regional FedAvg weighted by examples",
            )
        ),
    ),
    Page(
        slug="06-gateway-global",
        title="06 - Regional gateway and global aggregation",
        summary="The key piece: the global layer sees regions as clients, not hospitals.",
        body=(
            h2("ClientApp in gateway mode")
            + p(
                "The gateway does not train. It loads the already aggregated regional checkpoint "
                "and its metadata. It returns that model to the global SuperLink as a client update."
            )
            + snippet("hfl_cicids/client_app.py", 87, 130)
            + h2("ServerApp global")
            + p(
                "The global layer uses FedAvg again, but now its clients are gateways. Each "
                "gateway is weighted by the total number of training examples in its region."
            )
            + snippet("hfl_cicids/server_app.py", 88, 124)
            + h2("Formula")
            + code(
                """
N_global = sum_r N_r
theta_global = sum_r (N_r / N_global) * theta_r
""",
                "Global FedAvg weighted by region",
            )
            + callout(
                "Why this matters",
                "This is the difference between a real hierarchical demo and a Python script that "
                "averages files. The global checkpoint comes from a Flower federation with a global SuperLink.",
            )
        ),
    ),
    Page(
        slug="07-orchestration",
        title="07 - Orchestration, Docker, and logs",
        summary="How federations are actuated from a script and what to inspect when something fails.",
        body=(
            h2("Profiles Flower")
            + p(
                "Profiles connect names like <code>region-eu</code> or <code>global</code> "
                "to each SuperLink endpoint."
            )
            + cmd("make flower-config\nflwr config list")
            + h2("Run config")
            + p(
                "The orchestrator builds TOML strings for <code>--run-config</code>. Each run "
                "receives the level, region, initial checkpoint, output checkpoint, and weights."
            )
            + snippet("scripts/run_hierarchical_rounds.py", 70, 95)
            + snippet("scripts/run_hierarchical_rounds.py", 98, 166)
            + h2("Hierarchical loop")
            + snippet("scripts/run_hierarchical_rounds.py", 208, 245)
            + h2("Diagnostic commands")
            + cmd(
                """
docker compose ps
docker compose logs --tail=80 region-eu-superlink
docker compose logs --tail=80 hospital-eu-01-supernode
docker compose logs --tail=80 region-eu-gateway-supernode
docker compose logs --tail=80 global-superlink
"""
            )
        ),
    ),
    Page(
        slug="08-evaluation-privacy",
        title="08 - Evaluation, privacy, and next steps",
        summary="How the result is measured and which limitations should remain explicit.",
        body=(
            h2("Metrics")
            + p(
                "Accuracy is not enough because intrusion detection datasets are often imbalanced. "
                "Evaluation reports F1, ROC-AUC, AUPRC, false-positive rate, and false-negative rate."
            )
            + snippet("hfl_cicids/metrics.py", 17, 74)
            + h2("Evaluate checkpoints")
            + snippet("scripts/evaluate_global_model.py", 31, 91)
            + snippet("scripts/evaluate_global_model.py", 94, 176)
            + h2("Privacy")
            + ul(
                [
                    "Raw data is not uploaded to <code>shared/</code> or to Git.",
                    "Hospitals mount <code>/data</code> read-only.",
                    "The gateway mounts checkpoints, not hospital partitions.",
                    "Model updates can still leak information. This is not differential privacy.",
                ]
            )
            + h2("Reasonable extensions")
            + ul(
                [
                    "TLS and SuperNode authentication.",
                    "Secure aggregation so the server cannot inspect individual updates.",
                    "Centralized XGBoost/LightGBM baseline for performance comparison.",
                    "Federated scaling statistics instead of the simulation's global scaler.",
                ]
            )
        ),
    ),
]


CSS = """
:root {
  --bg: #f7f7f4;
  --panel: #ffffff;
  --ink: #202124;
  --muted: #5f6368;
  --line: #d9d8d1;
  --accent: #0f766e;
  --accent-dark: #115e59;
  --code-bg: #111827;
  --code-ink: #f9fafb;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  color: var(--ink);
  background: var(--bg);
  font: 16px/1.55 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

a { color: var(--accent-dark); }

.topbar {
  position: sticky;
  top: 0;
  z-index: 5;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  min-height: 58px;
  padding: 0 24px;
  background: rgba(255, 255, 255, 0.94);
  border-bottom: 1px solid var(--line);
}

.brand {
  color: var(--ink);
  font-weight: 700;
  text-decoration: none;
}

.repo {
  font-size: 14px;
  text-decoration: none;
}

.layout {
  display: grid;
  grid-template-columns: 280px minmax(0, 1fr);
  min-height: calc(100vh - 58px);
}

.sidebar {
  position: sticky;
  top: 58px;
  height: calc(100vh - 58px);
  overflow: auto;
  padding: 20px 14px;
  border-right: 1px solid var(--line);
}

.sidebar a {
  display: block;
  padding: 9px 10px;
  border-radius: 6px;
  color: var(--ink);
  text-decoration: none;
  font-size: 14px;
}

.sidebar a.active,
.sidebar a:hover {
  background: #e7f2f0;
  color: var(--accent-dark);
}

.content {
  width: min(1040px, 100%);
  padding: 34px 32px 80px;
}

.hero {
  padding: 34px 0 26px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 28px;
}

.hero h1 {
  margin: 0;
  max-width: 880px;
  font-size: clamp(34px, 5vw, 60px);
  line-height: 1.02;
  letter-spacing: 0;
}

.hero p {
  max-width: 760px;
  color: var(--muted);
  font-size: 18px;
}

.eyebrow {
  margin: 0 0 10px;
  color: var(--accent-dark);
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
}

h2 {
  margin-top: 34px;
  padding-top: 8px;
  font-size: 26px;
  letter-spacing: 0;
}

h3 {
  margin-top: 28px;
  font-size: 20px;
}

p, li { max-width: 860px; }

table {
  width: 100%;
  border-collapse: collapse;
  margin: 18px 0 28px;
  background: var(--panel);
  border: 1px solid var(--line);
}

th, td {
  padding: 11px 12px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}

th {
  background: #eef5f4;
  font-size: 13px;
  text-transform: uppercase;
}

code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.92em;
}

figure {
  margin: 20px 0 28px;
  border-radius: 8px;
  overflow: hidden;
  background: var(--code-bg);
}

figcaption {
  padding: 10px 12px;
  color: #d1d5db;
  background: #1f2937;
  font-size: 13px;
}

figcaption a { color: #99f6e4; }

pre {
  margin: 0;
  padding: 16px;
  overflow: auto;
  background: var(--code-bg);
  color: var(--code-ink);
}

pre.command {
  border-radius: 8px;
  border: 1px solid #233044;
}

.callout {
  margin: 22px 0;
  padding: 16px 18px;
  border-left: 4px solid var(--accent);
  background: #edf7f5;
}

.callout p { margin-bottom: 0; }

.notebook-cell {
  margin: 24px 0;
}

.cell-label {
  margin-bottom: 8px;
  color: var(--muted);
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
}

.notebook-output-html {
  margin: 12px 0 20px;
  padding: 14px;
  overflow: auto;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}

.pager {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 12px;
  margin-top: 48px;
  padding-top: 24px;
  border-top: 1px solid var(--line);
}

.pager a:last-child {
  text-align: right;
}

@media (max-width: 860px) {
  .layout { grid-template-columns: 1fr; }
  .sidebar {
    position: static;
    height: auto;
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }
  .content { padding: 24px 18px 56px; }
  .topbar { padding: 0 16px; }
}
"""


def convert_notebooks() -> None:
    for notebook in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(notebook, as_version=4)
        body = hero(f"Render: {notebook.stem}", f"HTML render generated from {notebook.name}.")
        body += p('<a href="../index.html">Back to handbook</a>')
        for idx, cell in enumerate(nb.cells, start=1):
            if cell.cell_type == "markdown":
                body += markdown_cell(cell.source)
            elif cell.cell_type == "code":
                body += f"<section class=\"notebook-cell\"><p class=\"cell-label\">Code cell {idx}</p>"
                body += code(cell.source, "")
                for output in cell.get("outputs", []):
                    body += render_output(output)
                body += "</section>"
        rendered = render_shell(f"Render {notebook.stem}", body, "index", "../")
        (RENDERS_DIR / f"{notebook.stem}.html").write_text(rendered)


def markdown_cell(source: str) -> str:
    blocks: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(p(" ".join(esc(line) for line in paragraph)))
            paragraph.clear()

    def flush_list() -> None:
        if list_items:
            blocks.append(ul([esc(item) for item in list_items]))
            list_items.clear()

    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            flush_list()
            continue
        if line.startswith("### "):
            flush_paragraph()
            flush_list()
            blocks.append(h3(line[4:]))
        elif line.startswith("## "):
            flush_paragraph()
            flush_list()
            blocks.append(h2(line[3:]))
        elif line.startswith("# "):
            flush_paragraph()
            flush_list()
            blocks.append(f"<h1>{esc(line[2:])}</h1>")
        elif line.startswith("- "):
            flush_paragraph()
            list_items.append(line[2:])
        elif len(line) > 2 and line[0].isdigit() and line[1:3] == ". ":
            flush_paragraph()
            list_items.append(line[3:])
        else:
            flush_list()
            paragraph.append(line)
    flush_paragraph()
    flush_list()
    return "\n".join(blocks)


def output_text(value: object) -> str:
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    return str(value)


def render_output(output: nbformat.NotebookNode) -> str:
    output_type = output.get("output_type", "")
    if output_type == "stream":
        return code(output_text(output.get("text", "")), "Output")
    if output_type == "error":
        traceback = output.get("traceback", [])
        return code("\n".join(str(line) for line in traceback), "Error")
    data = output.get("data", {})
    if "text/html" in data:
        return f"<div class=\"notebook-output-html\">{output_text(data['text/html'])}</div>"
    if "text/plain" in data:
        return code(output_text(data["text/plain"]), "Output")
    return ""


def main() -> None:
    if DOCS.exists():
        shutil.rmtree(DOCS)
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS / ".nojekyll").write_text("")
    (DOCS / "styles.css").write_text(CSS.strip() + "\n")
    (DOCS / "index.html").write_text(make_index())
    for index, page in enumerate(EPISODES):
        write_page(page, index)
    convert_notebooks()
    print(f"Wrote GitHub Pages site to {DOCS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
