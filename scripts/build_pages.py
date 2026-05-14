from __future__ import annotations

import csv
import html
import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import nbformat
from nbformat.validator import MissingIDFieldWarning
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name, get_lexer_for_filename
from pygments.util import ClassNotFound


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EPISODES_DIR = DOCS / "episodes"
RENDERS_DIR = DOCS / "renders"
REPORTS_DIR = ROOT / "reports"
REPO_URL = "https://github.com/davidsarrat/hospital-ids-hfl"

warnings.filterwarnings("ignore", category=MissingIDFieldWarning)

PYGMENTS_FORMATTER = HtmlFormatter(
    style="friendly",
    linenos="table",
    nowrap=False,
    cssclass="highlight",
)
PYGMENTS_INLINE_FORMATTER = HtmlFormatter(style="friendly", nowrap=False, cssclass="highlight")


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


def _highlight_text(text: str, language: str | None = None, filename: str | None = None) -> str:
    try:
        if language:
            lexer = get_lexer_by_name(language)
        elif filename:
            lexer = get_lexer_for_filename(filename)
        else:
            lexer = TextLexer()
    except ClassNotFound:
        lexer = TextLexer()
    return highlight(text.rstrip(), lexer, PYGMENTS_INLINE_FORMATTER)


def cmd(text: str) -> str:
    return f"<figure class=\"command\">{_highlight_text(text.strip(), 'bash')}</figure>"


def code(text: str, caption: str = "", language: str = "text") -> str:
    cap = f"<figcaption>{esc(caption)}</figcaption>" if caption else ""
    return f"<figure>{cap}{_highlight_text(text.rstrip(), language)}</figure>"


def callout(title: str, body: str) -> str:
    return f"<aside class=\"callout\"><strong>{esc(title)}</strong>{p(body)}</aside>"


def table(headers: list[str], rows: list[list[str]]) -> str:
    header_html = "".join(f"<th>{esc(header)}</th>" for header in headers)
    row_html = ""
    for row in rows:
        row_html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{row_html}</tbody></table>"


def notes(items: list[str], title: str = "What to notice") -> str:
    return f"<div class=\"notes\"><strong>{esc(title)}</strong>{ul(items)}</div>"


def line_ref(path: str, start: int, end: int | None = None, label: str | None = None) -> str:
    end = start if end is None else end
    text = label or (f"{path}:{start}" if start == end else f"{path}:{start}-{end}")
    suffix = f"#L{start}" if start == end else f"#L{start}-L{end}"
    return f"<a href=\"{REPO_URL}/blob/main/{esc(path)}{suffix}\">{esc(text)}</a>"


def lab_card(title: str, body: str, href: str) -> str:
    return (
        "<aside class=\"lab-card\">"
        f"<strong>{esc(title)}</strong>"
        f"<p>{body}</p>"
        f"<a href=\"{esc(href)}\">Open lab render</a>"
        "</aside>"
    )


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def topology_svg() -> str:
    return """
<figure class="diagram">
<figcaption>Three-layer Flower deployment topology</figcaption>
<svg viewBox="0 0 1060 560" role="img" aria-label="Three-layer hierarchical Flower topology">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#0f766e" />
    </marker>
  </defs>
  <rect x="30" y="40" width="1000" height="120" rx="8" fill="#eef5f4" stroke="#b7d6d0" />
  <text x="55" y="78" class="svg-title">Layer 3: global federation</text>
  <rect x="410" y="92" width="240" height="46" rx="6" fill="#ffffff" stroke="#0f766e" />
  <text x="530" y="121" text-anchor="middle">global-superlink</text>

  <rect x="30" y="210" width="1000" height="140" rx="8" fill="#fff7ed" stroke="#fed7aa" />
  <text x="55" y="248" class="svg-title">Layer 2: regional federations and gateways</text>
  <rect x="140" y="275" width="210" height="46" rx="6" fill="#ffffff" stroke="#c2410c" />
  <text x="245" y="304" text-anchor="middle">region-eu-superlink</text>
  <rect x="710" y="275" width="210" height="46" rx="6" fill="#ffffff" stroke="#c2410c" />
  <text x="815" y="304" text-anchor="middle">region-na-superlink</text>
  <rect x="410" y="222" width="190" height="42" rx="6" fill="#ffffff" stroke="#0f766e" />
  <text x="505" y="248" text-anchor="middle">EU gateway</text>
  <rect x="610" y="222" width="190" height="42" rx="6" fill="#ffffff" stroke="#0f766e" />
  <text x="705" y="248" text-anchor="middle">NA gateway</text>

  <rect x="30" y="400" width="1000" height="120" rx="8" fill="#f8fafc" stroke="#cbd5e1" />
  <text x="55" y="438" class="svg-title">Layer 1: local hospital SuperNodes</text>
  <rect x="90" y="462" width="130" height="36" rx="6" fill="#ffffff" stroke="#64748b" />
  <text x="155" y="486" text-anchor="middle">EU 01</text>
  <rect x="235" y="462" width="130" height="36" rx="6" fill="#ffffff" stroke="#64748b" />
  <text x="300" y="486" text-anchor="middle">EU 02</text>
  <rect x="380" y="462" width="130" height="36" rx="6" fill="#ffffff" stroke="#64748b" />
  <text x="445" y="486" text-anchor="middle">EU 03</text>
  <rect x="550" y="462" width="130" height="36" rx="6" fill="#ffffff" stroke="#64748b" />
  <text x="615" y="486" text-anchor="middle">NA 01</text>
  <rect x="695" y="462" width="130" height="36" rx="6" fill="#ffffff" stroke="#64748b" />
  <text x="760" y="486" text-anchor="middle">NA 02</text>
  <rect x="840" y="462" width="130" height="36" rx="6" fill="#ffffff" stroke="#64748b" />
  <text x="905" y="486" text-anchor="middle">NA 03</text>

  <path d="M155 462 L215 321" stroke="#0f766e" stroke-width="3" marker-end="url(#arrow)" fill="none" />
  <path d="M300 462 L245 321" stroke="#0f766e" stroke-width="3" marker-end="url(#arrow)" fill="none" />
  <path d="M445 462 L275 321" stroke="#0f766e" stroke-width="3" marker-end="url(#arrow)" fill="none" />
  <path d="M615 462 L785 321" stroke="#0f766e" stroke-width="3" marker-end="url(#arrow)" fill="none" />
  <path d="M760 462 L815 321" stroke="#0f766e" stroke-width="3" marker-end="url(#arrow)" fill="none" />
  <path d="M905 462 L845 321" stroke="#0f766e" stroke-width="3" marker-end="url(#arrow)" fill="none" />
  <path d="M350 298 C430 298, 430 250, 410 244" stroke="#0f766e" stroke-width="3" marker-end="url(#arrow)" fill="none" />
  <path d="M710 298 C645 298, 645 250, 610 244" stroke="#0f766e" stroke-width="3" marker-end="url(#arrow)" fill="none" />
  <path d="M505 222 C505 178, 500 160, 500 139" stroke="#0f766e" stroke-width="3" marker-end="url(#arrow)" fill="none" />
  <path d="M705 222 C705 178, 580 160, 570 139" stroke="#0f766e" stroke-width="3" marker-end="url(#arrow)" fill="none" />
  <text x="55" y="548" class="svg-note">Raw parquet rows stay mounted only inside each hospital ClientApp container. The global layer receives regional checkpoints through gateway SuperNodes.</text>
</svg>
</figure>
"""


def partition_chart() -> str:
    metadata_paths = sorted((ROOT / "data" / "partitions").glob("*/metadata.json"))
    if not metadata_paths:
        return callout("Partition chart unavailable", "Run <code>make partition SEED=123</code> to generate hospital metadata.")
    rows = []
    max_total = 1
    for path in metadata_paths:
        meta = read_json(path)
        train_counts = meta.get("train_label_counts", {})
        benign = int(train_counts.get("0", 0))
        attack = int(train_counts.get("1", 0))
        total = benign + attack
        max_total = max(max_total, total)
        rows.append((meta.get("hospital_id", path.parent.name), meta.get("region", ""), benign, attack, total))
    svg_rows = []
    y = 55
    for hospital_id, region, benign, attack, total in rows:
        benign_w = int(620 * benign / max_total)
        attack_w = int(620 * attack / max_total)
        svg_rows.append(f'<text x="20" y="{y + 18}" class="chart-label">{esc(hospital_id)}</text>')
        svg_rows.append(f'<rect x="190" y="{y}" width="{benign_w}" height="24" fill="#0f766e" />')
        svg_rows.append(f'<rect x="{190 + benign_w}" y="{y}" width="{attack_w}" height="24" fill="#dc2626" />')
        svg_rows.append(f'<text x="830" y="{y + 18}" class="chart-label">{esc(region)} | train={total:,} | attack={attack:,}</text>')
        y += 42
    height = y + 35
    return (
        '<figure class="chart"><figcaption>Training rows per hospital after non-IID partitioning</figcaption>'
        f'<svg viewBox="0 0 1040 {height}" role="img" aria-label="Hospital partition chart">'
        '<text x="190" y="28" class="chart-label">BENIGN</text><rect x="250" y="15" width="20" height="14" fill="#0f766e" />'
        '<text x="295" y="28" class="chart-label">ATTACK</text><rect x="355" y="15" width="20" height="14" fill="#dc2626" />'
        + "".join(svg_rows)
        + "</svg></figure>"
    )


def metrics_chart() -> str:
    rows = read_csv_rows(REPORTS_DIR / "metrics_summary_global.csv")
    if not rows:
        return callout("Metrics chart unavailable", "Run <code>make eval</code> or <code>make demo</code> to produce reports/metrics_summary_global.csv.")
    row = rows[0]
    metrics = [
        ("weighted_f1", float(row.get("weighted_f1", 0.0))),
        ("macro_f1", float(row.get("macro_f1", 0.0))),
        ("weighted_roc_auc", float(row.get("weighted_roc_auc", 0.0))),
        ("weighted_auprc", float(row.get("weighted_auprc", 0.0))),
    ]
    bars = []
    y = 55
    for name, value in metrics:
        width = int(650 * max(0.0, min(1.0, value)))
        bars.append(f'<text x="20" y="{y + 18}" class="chart-label">{esc(name)}</text>')
        bars.append(f'<rect x="220" y="{y}" width="{width}" height="24" fill="#0f766e" />')
        bars.append(f'<text x="{235 + width}" y="{y + 18}" class="chart-label">{value:.3f}</text>')
        y += 46
    return (
        '<figure class="chart"><figcaption>Latest global checkpoint evaluation summary</figcaption>'
        '<svg viewBox="0 0 1040 260" role="img" aria-label="Global evaluation metrics">'
        + "".join(bars)
        + "</svg></figure>"
    )


def predictions_table(limit: int = 12) -> str:
    rows = read_csv_rows(REPORTS_DIR / "predictions_hospital_eu_01.csv")
    if not rows:
        return callout("Prediction table unavailable", "Run <code>make predict</code> or <code>make demo</code> to generate row-level predictions.")
    display_rows = []
    for row in rows[:limit]:
        display_rows.append(
            [
                esc(row.get("row_index", "")),
                esc(row.get("hospital_id", "")),
                esc(row.get("label", "")),
                f"{float(row.get('prob_attack', 0.0)):.4f}",
                esc(row.get("prediction", "")),
                esc(row.get("correct", "")),
            ]
        )
    return table(["row", "hospital", "label", "prob_attack", "prediction", "correct"], display_rows)


def demo_transcript(max_lines: int = 90) -> str:
    path = REPORTS_DIR / "demo_transcript.txt"
    if not path.exists():
        return callout("Demo transcript unavailable", "Run <code>make demo</code> to capture the training/evaluation transcript.")
    lines = path.read_text(errors="replace").splitlines()
    interesting = [
        line
        for line in lines
        if line.strip()
        and (
            "Running:" in line
            or "Global round" in line
            or "Wrote" in line
            or "Latest global checkpoint" in line
            or "weighted_f1" in line
            or "hierarchical_fl" in line
            or "train_loss" in line
            or "Received" in line
            or "Sending" in line
            or "Wrote predictions" in line
        )
    ]
    if not interesting:
        interesting = lines
    return code("\n".join(interesting[-max_lines:]), "Captured make demo transcript excerpt", "text")


def snippet(path: str, start: int, end: int, caption: str | None = None, language: str | None = None) -> str:
    file_path = ROOT / path
    lines = file_path.read_text().splitlines()
    selected = lines[start - 1 : end]
    label = caption or f"{path}:{start}-{end}"
    source_link = f"{REPO_URL}/blob/main/{path}#L{start}-L{end}"
    try:
        lexer = get_lexer_by_name(language) if language else get_lexer_for_filename(path)
    except ClassNotFound:
        lexer = TextLexer()
    formatter = HtmlFormatter(
        style="friendly",
        linenos="table",
        linenostart=start,
        nowrap=False,
        cssclass="highlight",
    )
    highlighted = highlight("\n".join(selected), lexer, formatter)
    return (
        f"<figure><figcaption><a href=\"{esc(source_link)}\">{esc(label)}</a></figcaption>"
        f"{highlighted}</figure>"
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
        "runtime logs lab. If you want to reproduce everything, follow the episodes in order.",
    )
    body += h2("Episode roadmap")
    body += table(["#", "Episode", "What you learn"], rows)
    body += h2("Demo topology at a glance")
    body += topology_svg()
    body += h2("Hands-on labs")
    body += p(
        "The rendered notebooks are treated as labs attached to the chapters, not as a separate "
        "parallel story. Start with the book chapters for the explanation, then open the matching "
        "lab when you want to inspect the actual rendered output."
    )
    body += table(
        ["Lab", "Use it after", "Render"],
        [
            ["Dataset and partitioning", "Episode 02", '<a href="renders/00_dataset_and_partitioning.html">Open lab 00</a>'],
            ["Baselines", "Episodes 04 and 08", '<a href="renders/01_centralized_baseline.html">Open lab 01</a>'],
            ["Flat FL baseline", "Episode 05", '<a href="renders/02_flat_fl_baseline.html">Open lab 02</a>'],
            ["Hierarchical demo commands", "Episode 07", '<a href="renders/03_hierarchical_flower_demo.html">Open lab 03</a>'],
            ["Runtime logs and outputs", "Episode 09", '<a href="renders/04_flower_runtime_orchestration.html">Open lab 04</a>'],
        ],
    )
    body += h2("How to read this handbook")
    body += p(
        "Every code block is generated from the repository at build time. The line numbers on the "
        "left match the current source files, and each caption links to the exact GitHub line range. "
        "The prose below the snippets calls out the specific lines that matter, so you can connect "
        "the architecture to the implementation without guessing."
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
make predict GLOBAL_ROUNDS=3
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
            + topology_svg()
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
            + h2("The invariant to keep in mind")
            + p(
                "The architecture is only hierarchical if every layer exposes an aggregate to the "
                "layer above it. Hospitals expose model updates to a regional SuperLink; a region "
                "exposes one regional checkpoint through a gateway SuperNode; the global SuperLink "
                "never sees hospital partitions or hospital SuperNodes directly."
            )
            + notes(
                [
                    "The region is a Flower federation in its own right, not just a Python function.",
                    "The global layer is another Flower federation whose clients are gateways.",
                    "Weighted averaging works because hospitals and gateways both return <code>num-examples</code>.",
                    "The <code>shared/</code> directory is for model/checkpoint artifacts, not raw rows.",
                ],
                "Core rules",
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
            + snippet("Makefile", 1, 74)
            + notes(
                [
                    f"{line_ref('Makefile', 18, 23, 'Lines 18-23')} separate download/cleaning from partitioning, so data preparation can be rerun without rebuilding Docker.",
                    f"{line_ref('Makefile', 37, 40, 'Lines 37-40')} call the hierarchical orchestrator for normal training runs.",
                    f"{line_ref('Makefile', 52, 74, 'Lines 52-74')} define the compact demo: reset the runtime, train, evaluate, generate predictions, rebuild educational output, then print the CSV summary.",
                    "The demo target assumes the dataset and partitions already exist. This avoids downloading a large dataset every time someone wants to show the algorithm running.",
                ]
            )
            + h2("Direct demo")
            + p(
                "Once data and partitions exist, <code>make demo</code> starts Docker, runs one "
                "global round with one regional round, evaluates, and refreshes the renders."
            )
            + cmd("make demo")
            + h2("Package dependencies")
            + snippet("pyproject.toml", 15, 28)
            + notes(
                [
                    f"{line_ref('pyproject.toml', 16, label='Line 16')} pins Flower to the version used for the deployment runtime demo.",
                    f"{line_ref('pyproject.toml', 17, label='Line 17')} keeps the model in PyTorch so FedAvg can average tensors directly.",
                    f"{line_ref('pyproject.toml', 20, 21, 'Lines 20-21')} provide preprocessing and parquet IO.",
                    f"{line_ref('pyproject.toml', 22, label='Line 22')} keeps the dataset download reproducible through Kaggle.",
                ]
            )
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
            + notes(
                [
                    f"{line_ref('scripts/prepare_cicids.py', 60, 62, 'Lines 60-62')} fail early if the raw CSVs are missing. This is better than silently creating empty partitions.",
                    f"{line_ref('scripts/prepare_cicids.py', 64, 71, 'Lines 64-71')} read every CIC-IDS2017 CSV and concatenate them into one simulation input.",
                    f"{line_ref('scripts/prepare_cicids.py', 75, 83, 'Lines 75-83')} implement the binary label correction: BENIGN is 0 and every attack label is 1.",
                    f"{line_ref('scripts/prepare_cicids.py', 84, 97, 'Lines 84-97')} drop metadata/leakage columns and keep numeric features only.",
                    f"{line_ref('scripts/prepare_cicids.py', 103, 105, 'Lines 103-105')} keep <code>attack_type</code> only for partitioning; training splits later contain numeric features plus <code>label</code>.",
                ]
            )
            + h2("Non-IID split with seed 123")
            + p(
                "Partitioning assigns attack groups to hospitals with biased weights. This makes "
                "each hospital see a different distribution, which is the interesting case for FL."
            )
            + snippet("scripts/make_partitions.py", 31, 72, "Attack grouping and hospital preferences")
            + snippet("scripts/make_partitions.py", 112, 128, "Dirichlet assignment by group")
            + notes(
                [
                    f"{line_ref('scripts/make_partitions.py', 31, 51, 'Lines 31-51')} compress many CIC-IDS2017 attack strings into stable attack groups.",
                    f"{line_ref('scripts/make_partitions.py', 60, 72, 'Lines 60-72')} combine a Dirichlet draw with each hospital's preferred attack groups. This creates non-IID partitions without hand-picking every row.",
                    f"{line_ref('scripts/make_partitions.py', 112, 128, 'Lines 112-128')} assigns rows group by group and repairs minimum benign/attack coverage where possible.",
                    "The seed controls the random generator, so the same processed parquet produces the same hospital partitions when <code>SEED=123</code> is used.",
                ],
                "Why this split matters",
            )
            + h2("Train-only scaling")
            + p(
                "The MVP demo fits imputation and standardization from the simulated train splits. "
                "It then transforms train, validation, and test for every hospital with those values."
            )
            + snippet("scripts/make_partitions.py", 167, 193)
            + notes(
                [
                    f"{line_ref('scripts/make_partitions.py', 167, 180, 'Lines 167-180')} fit imputation and standardization from train splits only.",
                    f"{line_ref('scripts/make_partitions.py', 183, 193, 'Lines 183-193')} apply the stored impute/mean/std values to every split and write a compact training table.",
                    "This is still a simulation-level scaler. A more privacy-preserving version would aggregate local count/sum/sumsq statistics instead of fitting centrally.",
                ]
            )
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
            + h2("What the generated partitions look like")
            + p(
                "The chart below is generated from the local <code>metadata.json</code> files created "
                "by the partitioning script. The uneven bar sizes and different attack ratios are "
                "intentional: they show that hospitals do not receive identical data distributions."
            )
            + partition_chart()
            + lab_card(
                "Hands-on lab: dataset and partitioning",
                "Open the rendered notebook when you want to see the dataset inspection workflow, "
                "partition metadata tables, and the commands used to build the local hospital folders.",
                "../renders/00_dataset_and_partitioning.html",
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
            + notes(
                [
                    f"{line_ref('scripts/generate_compose.py', 43, 52, 'Lines 43-52')} create one SuperLink service and expose the Control API port to the host.",
                    f"{line_ref('scripts/generate_compose.py', 56, 70, 'Lines 56-70')} create the ServerApp SuperExec. It connects to the generated SuperLink AppIO API, for example <code>region-eu-superlink:9091</code>.",
                    "The SuperLink is the coordination hub; the ServerApp SuperExec is the worker process that actually runs the application code submitted by <code>flwr run</code>.",
                ]
            )
            + h2("Hospital SuperNode")
            + p(
                "The hospital connects to its regional SuperLink, and its ClientApp mounts only "
                "its own local folder at <code>/data:ro</code>."
            )
            + snippet("scripts/generate_compose.py", 74, 107)
            + notes(
                [
                    f"{line_ref('scripts/generate_compose.py', 78, 89, 'Lines 78-89')} start a hospital SuperNode and point it to the regional SuperLink Fleet API.",
                    f"{line_ref('scripts/generate_compose.py', 88, 89, 'Lines 88-89')} pass node config into Flower. The ClientApp later reads <code>role</code>, <code>region</code>, <code>hospital-id</code>, and <code>data-dir</code> from this config.",
                    f"{line_ref('scripts/generate_compose.py', 93, 104, 'Lines 93-104')} start the ClientApp SuperExec and mount only this hospital's partition at <code>/data:ro</code>.",
                    "Because each hospital service gets a different host folder mounted into <code>/data</code>, the ClientApp code can be shared while the data boundary remains per-hospital.",
                ]
            )
            + h2("RegionGateway SuperNode")
            + p(
                "The gateway connects to the global SuperLink and mounts <code>/shared</code>, not "
                "<code>/data</code>. This lets it return regional checkpoints without reading raw rows."
            )
            + snippet("scripts/generate_compose.py", 110, 143)
            + notes(
                [
                    f"{line_ref('scripts/generate_compose.py', 113, 124, 'Lines 113-124')} define a gateway SuperNode connected to <code>global-superlink:9092</code>, not to a regional SuperLink.",
                    f"{line_ref('scripts/generate_compose.py', 123, 124, 'Lines 123-124')} mark the node as <code>role=\"region-gateway\"</code>, which switches the ClientApp into gateway behavior.",
                    f"{line_ref('scripts/generate_compose.py', 125, 140, 'Lines 125-140')} mount <code>./shared:/shared</code> only. There is no hospital <code>/data</code> mount for a gateway.",
                    "This is the privacy boundary that makes the global layer see regional models rather than raw local rows.",
                ]
            )
            + h2("Real render")
            + p(
                "The runtime lab is linked from the end-to-end demo chapter. It is useful after you "
                "understand the roles above because it shows the running services, Flower profiles, "
                "checkpoint metadata, and message logs from the actual containers."
            )
            + lab_card(
                "Hands-on lab: runtime logs and container state",
                "Use this lab to inspect <code>docker compose ps</code>, Flower profiles, gateway logs, "
                "regional checkpoint metadata, and global evaluation output.",
                "../renders/04_flower_runtime_orchestration.html",
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
            + notes(
                [
                    f"{line_ref('hfl_cicids/task.py', 12, 30, 'Lines 12-30')} load one parquet split and validate that all features are finite.",
                    f"{line_ref('hfl_cicids/task.py', 24, 25, 'Lines 24-25')} separate labels from features and convert them to float32 arrays for PyTorch.",
                    f"{line_ref('hfl_cicids/task.py', 42, 63, 'Lines 42-63')} define the MLP. The final layer has one output because this is binary benign-vs-attack classification.",
                    "The model is intentionally averageable: its state dict is a set of tensors that Flower can aggregate with FedAvg.",
                ]
            )
            + h2("Local training")
            + snippet("hfl_cicids/task.py", 88, 131)
            + notes(
                [
                    f"{line_ref('hfl_cicids/task.py', 88, 97, 'Lines 88-97')} compute <code>pos_weight</code> to reduce the impact of class imbalance during binary classification.",
                    f"{line_ref('hfl_cicids/task.py', 110, 112, 'Lines 110-112')} create the weighted BCE loss and AdamW optimizer used by every hospital.",
                    f"{line_ref('hfl_cicids/task.py', 117, 129, 'Lines 117-129')} run the local epochs. No server or other hospital data is accessed here.",
                    "This function knows nothing about Flower. That makes the ML task testable on its own and keeps Flower-specific logic in the ClientApp.",
                ]
            )
            + h2("Hospital ClientApp")
            + p(
                "When the node role is hospital, Flower sends initial arrays. The ClientApp loads "
                "those weights, trains on <code>/data/train.parquet</code>, evaluates on validation, "
                "and returns arrays plus metrics. The <code>num-examples</code> key enables weighted FedAvg."
            )
            + snippet("hfl_cicids/client_app.py", 32, 84)
            + notes(
                [
                    f"{line_ref('hfl_cicids/client_app.py', 32, 41, 'Lines 32-41')} route by node role. Hospitals train; gateways return regional checkpoints.",
                    f"{line_ref('hfl_cicids/client_app.py', 44, 55, 'Lines 44-55')} load only the mounted hospital data directory and the arrays sent by Flower.",
                    f"{line_ref('hfl_cicids/client_app.py', 56, 64, 'Lines 56-64')} execute local training and validation.",
                    f"{line_ref('hfl_cicids/client_app.py', 66, 80, 'Lines 66-80')} return two records: <code>arrays</code> for model parameters and <code>metrics</code> for aggregation metadata.",
                    f"{line_ref('hfl_cicids/client_app.py', 68, label='Line 68')} is critical: Flower's FedAvg uses <code>num-examples</code> as the weight for this hospital update.",
                ]
            )
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
            + notes(
                [
                    f"{line_ref('hfl_cicids/server_app.py', 16, 25, 'Lines 16-25')} choose regional or global behavior from <code>context.run_config['level']</code>.",
                    f"{line_ref('hfl_cicids/server_app.py', 28, 41, 'Lines 28-41')} create initial arrays from the configured input dimension and optionally load a previous checkpoint.",
                    "This is how a later global round can seed regional hospital training with the previous global model.",
                ]
            )
            + h2("Regional federation")
            + p(
                "Each region uses FedAvg with <code>weighted_by_key='num-examples'</code>. "
                "The clients are the hospitals connected to the regional SuperLink."
            )
            + snippet("hfl_cicids/server_app.py", 57, 116)
            + notes(
                [
                    f"{line_ref('hfl_cicids/server_app.py', 64, 71, 'Lines 64-71')} instantiate FedAvg and explicitly set <code>weighted_by_key=\"num-examples\"</code>.",
                    f"{line_ref('hfl_cicids/server_app.py', 73, 83, 'Lines 73-83')} start the Flower strategy on the regional grid. The clients sampled here are hospital SuperNodes connected to the regional SuperLink.",
                    f"{line_ref('hfl_cicids/server_app.py', 85, 90, 'Lines 85-90')} capture the final aggregated train/evaluate metrics from Flower's <code>Result</code> object.",
                    f"{line_ref('hfl_cicids/server_app.py', 92, 116, 'Lines 92-116')} save the aggregated regional checkpoint plus metadata needed by the gateway.",
                    f"{line_ref('hfl_cicids/server_app.py', 102, label='Line 102')} records the region's training-example total so the global layer can weight this region correctly.",
                ]
            )
            + h2("Formula")
            + code(
                """
N_r = sum_h n_h
theta_r = sum_h (n_h / N_r) * theta_h
""",
                "Regional FedAvg weighted by examples",
            )
            + lab_card(
                "Hands-on lab: flat FL comparison",
                "The flat baseline lab shows a one-layer Flower federation where all six hospitals "
                "connect to one SuperLink. Compare it with the regional/global split explained here.",
                "../renders/02_flat_fl_baseline.html",
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
            + notes(
                [
                    f"{line_ref('hfl_cicids/client_app.py', 87, 92, 'Lines 87-92')} state the key design contract: the gateway does not train on rows.",
                    f"{line_ref('hfl_cicids/client_app.py', 94, 100, 'Lines 94-100')} compute the expected regional checkpoint path from <code>region</code> and <code>global-round</code>.",
                    f"{line_ref('hfl_cicids/client_app.py', 102, 110, 'Lines 102-110')} fail if either the checkpoint or metadata is missing. This prevents the global layer from silently aggregating stale data.",
                    f"{line_ref('hfl_cicids/client_app.py', 112, 119, 'Lines 112-119')} load the regional model and expose <code>num-examples</code> plus regional validation metrics.",
                    f"{line_ref('hfl_cicids/client_app.py', 122, 130, 'Lines 122-130')} return the regional model as a normal Flower client update.",
                ]
            )
            + h2("ServerApp global")
            + p(
                "The global layer uses FedAvg again, but now its clients are gateways. Each "
                "gateway is weighted by the total number of training examples in its region."
            )
            + snippet("hfl_cicids/server_app.py", 119, 155)
            + notes(
                [
                    f"{line_ref('hfl_cicids/server_app.py', 124, 131, 'Lines 124-131')} configure FedAvg at the global level. Here the clients are gateway SuperNodes, not hospitals.",
                    f"{line_ref('hfl_cicids/server_app.py', 133, 142, 'Lines 133-142')} run exactly one global aggregation round for each outer loop iteration.",
                    f"{line_ref('hfl_cicids/server_app.py', 144, 155, 'Lines 144-155')} save the global checkpoint and metadata after the gateway updates are aggregated.",
                    f"{line_ref('hfl_cicids/server_app.py', 152, label='Line 152')} records the global training-example total, which should match the sum of regional totals.",
                ]
            )
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
            + notes(
                [
                    f"{line_ref('scripts/run_hierarchical_rounds.py', 70, 80, 'Lines 70-80')} serialize Python values into Flower's run-config syntax.",
                    f"{line_ref('scripts/run_hierarchical_rounds.py', 83, 95, 'Lines 83-95')} build and optionally execute the <code>flwr run . &lt;profile&gt; --stream --run-config ...</code> command.",
                    f"{line_ref('scripts/run_hierarchical_rounds.py', 98, 133, 'Lines 98-133')} define the regional run config: level, region, previous global checkpoint, output regional checkpoint, and regional sample count.",
                    f"{line_ref('scripts/run_hierarchical_rounds.py', 136, 166, 'Lines 136-166')} define the global run config: previous global checkpoint, output global checkpoint, and total regional sample count.",
                ]
            )
            + h2("Hierarchical loop")
            + snippet("scripts/run_hierarchical_rounds.py", 208, 245)
            + notes(
                [
                    f"{line_ref('scripts/run_hierarchical_rounds.py', 208, 228, 'Lines 208-228')} run each regional federation first. These runs create the regional checkpoints consumed by the gateways.",
                    f"{line_ref('scripts/run_hierarchical_rounds.py', 230, 243, 'Lines 230-243')} run the global federation after all regional checkpoints for that round exist.",
                    f"{line_ref('scripts/run_hierarchical_rounds.py', 245, label='Line 245')} prints the latest global checkpoint path, which is what evaluation should load next.",
                    "The order is intentional: regional training happens before global aggregation in every outer round.",
                ]
            )
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
            + h2("What progress looks like")
            + p(
                "When <code>make demo</code> runs, the orchestrator prints each <code>flwr run</code> "
                "command before executing it. Flower then streams server/client progress: sampled "
                "nodes, train/evaluate messages, completed rounds, checkpoint writes, evaluation "
                "reports, and finally a compact metrics CSV."
            )
            + demo_transcript(55)
            + lab_card(
                "Hands-on lab: hierarchical runtime",
                "Open this after reading the orchestration code. It shows service state, exact "
                "<code>flwr run</code> commands, logs, checkpoint metadata, and the evaluation summary.",
                "../renders/04_flower_runtime_orchestration.html",
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
            + metrics_chart()
            + snippet("hfl_cicids/metrics.py", 17, 74)
            + h2("Evaluate checkpoints")
            + snippet("scripts/evaluate_global_model.py", 31, 91)
            + snippet("scripts/evaluate_global_model.py", 94, 176)
            + notes(
                [
                    f"{line_ref('scripts/evaluate_global_model.py', 31, 67, 'Lines 31-67')} load one checkpoint and evaluate it separately on every hospital split.",
                    f"{line_ref('scripts/evaluate_global_model.py', 70, 91, 'Lines 70-91')} reduce per-hospital rows into weighted and macro summaries.",
                    f"{line_ref('scripts/evaluate_global_model.py', 161, 166, 'Lines 161-166')} write both detailed hospital metrics and a global summary CSV.",
                ]
            )
            + h2("Prediction from a trained checkpoint")
            + p(
                "Evaluation summarizes a checkpoint over many rows. Prediction is the smaller "
                "inference path: load one checkpoint, take rows from one hospital split, compute "
                "<code>sigmoid(logit)</code>, threshold it, and write row-level predictions."
            )
            + snippet("scripts/predict_with_checkpoint.py", 25, 65)
            + notes(
                [
                    f"{line_ref('scripts/predict_with_checkpoint.py', 31, 38, 'Lines 31-38')} enforce that both the checkpoint and selected hospital split exist.",
                    f"{line_ref('scripts/predict_with_checkpoint.py', 44, 48, 'Lines 44-48')} rebuild the same MLP architecture and load the trained checkpoint weights.",
                    f"{line_ref('scripts/predict_with_checkpoint.py', 50, 53, 'Lines 50-53')} compute attack probabilities with a sigmoid and convert them to binary predictions.",
                    f"{line_ref('scripts/predict_with_checkpoint.py', 55, 65, 'Lines 55-65')} return a human-readable table with probability, prediction, label, and correctness.",
                ]
            )
            + predictions_table()
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
    Page(
        slug="09-end-to-end-demo",
        title="09 - End-to-end demo outputs",
        summary="How to run the complete demo and interpret the visible training, evaluation, and prediction artifacts.",
        body=(
            h2("One command for the live demo")
            + p(
                "The easiest way to present the system is <code>make demo</code>. It assumes the "
                "dataset has already been downloaded and partitioned. It then regenerates Compose, "
                "configures Flower profiles, starts containers, runs one hierarchical training cycle, "
                "evaluates the global checkpoint, produces a small prediction table, refreshes notebook "
                "renders, rebuilds this handbook, and prints the global metrics summary."
            )
            + snippet("Makefile", 52, 74, "make demo target")
            + notes(
                [
                    f"{line_ref('Makefile', 52, 59, 'Lines 52-59')} start from a clean Docker federation, wait for the SuperLinks, reset demo checkpoints, and create a fresh transcript.",
                    f"{line_ref('Makefile', 60, 63, 'Lines 60-63')} run the regional and global Flower federations through the orchestrator while appending logs to <code>reports/demo_transcript.txt</code>.",
                    f"{line_ref('Makefile', 64, 70, 'Lines 64-70')} evaluate the produced global checkpoint and demonstrate inference by writing row-level predictions for one hospital.",
                    f"{line_ref('Makefile', 71, 74, 'Lines 71-74')} refresh the educational artifacts and print the final metrics CSV.",
                ]
            )
            + h2("Captured transcript")
            + p(
                "The transcript below is intentionally not a clean benchmark table. It is a demo "
                "operator view: commands are printed, Flower streams progress, checkpoints and reports "
                "are written, and the final metrics are visible without opening any Python code."
            )
            + demo_transcript(90)
            + h2("Result artifacts")
            + table(
                ["Artifact", "What it proves"],
                [
                    ["<code>shared/checkpoints/region_eu/round_1.pt</code>", "The EU regional Flower federation aggregated hospital updates."],
                    ["<code>shared/checkpoints/region_na/round_1.pt</code>", "The NA regional Flower federation aggregated hospital updates."],
                    ["<code>shared/checkpoints/global/round_1.pt</code>", "The global Flower federation aggregated gateway updates."],
                    ["<code>reports/metrics_summary_global.csv</code>", "The global checkpoint can be evaluated across hospital test splits."],
                    ["<code>reports/predictions_hospital_eu_01.csv</code>", "The trained checkpoint can be used for row-level inference."],
                ],
            )
            + h2("Current global metrics")
            + metrics_chart()
            + h2("Current prediction sample")
            + predictions_table(15)
            + lab_card(
                "Hands-on lab: rendered runtime evidence",
                "This lab is the rendered notebook view of the same runtime evidence: services, "
                "logs, checkpoint metadata, summary metrics, and raw-data boundary checks.",
                "../renders/04_flower_runtime_orchestration.html",
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
  background: var(--panel);
  border: 1px solid var(--line);
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
  background: transparent;
}

figure.command {
  border-radius: 8px;
  border: 1px solid #233044;
  background: var(--code-bg);
}

.callout {
  margin: 22px 0;
  padding: 16px 18px;
  border-left: 4px solid var(--accent);
  background: #edf7f5;
}

.callout p { margin-bottom: 0; }

.notes {
  margin: 18px 0 30px;
  padding: 15px 18px;
  border: 1px solid #b7d6d0;
  border-radius: 8px;
  background: #f1faf8;
}

.notes strong,
.lab-card strong {
  display: block;
  margin-bottom: 8px;
}

.lab-card {
  margin: 22px 0 30px;
  padding: 16px 18px;
  border: 1px solid #fed7aa;
  border-radius: 8px;
  background: #fff7ed;
}

.lab-card p {
  margin: 0 0 10px;
}

.diagram,
.chart {
  background: #ffffff;
}

.diagram svg,
.chart svg {
  display: block;
  width: 100%;
  height: auto;
}

.svg-title {
  font-size: 18px;
  font-weight: 700;
  fill: #202124;
}

.svg-note,
.chart-label {
  font-size: 14px;
  fill: #334155;
}

.highlight {
  margin: 0;
  overflow: auto;
}

.highlight pre {
  padding: 14px;
}

.highlighttable {
  margin: 0;
  border: 0;
  background: transparent;
}

.highlighttable td {
  padding: 0;
  border: 0;
}

.highlighttable .linenos {
  width: 1%;
  min-width: 48px;
  color: #64748b;
  background: #eef2f7;
  border-right: 1px solid #d6dee8;
  text-align: right;
  user-select: none;
}

.highlighttable .code {
  width: 99%;
}

.command .highlight {
  background: var(--code-bg);
}

.command pre {
  color: var(--code-ink);
}

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
        return code(output_text(output.get("text", "")), "Output", "text")
    if output_type == "error":
        traceback = output.get("traceback", [])
        return code("\n".join(str(line) for line in traceback), "Error", "text")
    data = output.get("data", {})
    if "text/html" in data:
        return f"<div class=\"notebook-output-html\">{output_text(data['text/html'])}</div>"
    if "text/plain" in data:
        return code(output_text(data["text/plain"]), "Output", "text")
    return ""


def main() -> None:
    if DOCS.exists():
        shutil.rmtree(DOCS)
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS / ".nojekyll").write_text("")
    (DOCS / "styles.css").write_text(
        CSS.strip() + "\n\n" + PYGMENTS_FORMATTER.get_style_defs(".highlight") + "\n"
    )
    (DOCS / "index.html").write_text(make_index())
    for index, page in enumerate(EPISODES):
        write_page(page, index)
    convert_notebooks()
    print(f"Wrote GitHub Pages site to {DOCS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
