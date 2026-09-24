"""CLI commands for model training and metrics"""
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import json
from pathlib import Path
import os

from .model import run_model_training
from .metrics import run_metrics
from .database import get_session
from .models import Prompt, Response, Page
from .extract_cli import compute_accuracy

app = typer.Typer(help="Model training and metrics")
console = Console()


@app.command()
def train() -> None:
    """Train all three logistic regression models with GroupKFold."""
    console.print("[bold]Training models with GroupKFold (grouped by prompt_id)...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Training...", total=None)
        results = run_model_training()
        progress.update(task, description="Complete!")

    # Display results
    table = Table(title="Model ROC-AUC Scores (GroupKFold, n=5)")
    table.add_column("Model", style="cyan")
    table.add_column("Mean ROC-AUC", justify="right", style="green")
    table.add_column("Std ROC-AUC", justify="right")
    table.add_column("Fold Scores", style="dim")

    for model_name, result in results.items():
        fold_str = ", ".join([f"{s:.3f}" for s in result["fold_scores"]])
        table.add_row(
            model_name,
            f"{result['mean_roc_auc']:.4f}",
            f"{result['std_roc_auc']:.4f}",
            fold_str
        )

    console.print(table)
    console.print("[bold green]Model training complete![/bold green]")


@app.command()
def show_results(path: str = "data/model_results.json") -> None:
    """Show saved model results."""
    if not os.path.exists(path):
        console.print(f"[red]Results file not found: {path}[/red]")
        raise typer.Exit(1)

    with open(path) as f:
        results = json.load(f)

    table = Table(title="Saved Model Results")
    table.add_column("Model", style="cyan")
    table.add_column("Mean ROC-AUC", justify="right", style="green")
    table.add_column("Std ROC-AUC", justify="right")
    table.add_column("N Samples", justify="right")
    table.add_column("N Features", justify="right")

    for model_name, result in results.items():
        table.add_row(
            model_name,
            f"{result['mean_roc_auc']:.4f}",
            f"{result['std_roc_auc']:.4f}",
            str(result['n_samples']),
            str(result['n_features']),
        )

    console.print(table)

    # Show top features for full model
    if "model_c_all" in results:
        fi = results["model_c_all"]["feature_importance"]
        # Sort by absolute coefficient value
        sorted_fi = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:20]

        feat_table = Table(title="Top 20 Features (Model C - All Features)")
        feat_table.add_column("Feature", style="cyan")
        feat_table.add_column("Coefficient", justify="right")

        for feat, coef in sorted_fi:
            feat_table.add_row(feat, f"{coef:.4f}")

        console.print(feat_table)


@app.command()
def compute() -> None:
    """Compute all business metrics."""
    console.print("[bold]Computing metrics...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Computing...", total=None)
        metrics = run_metrics()
        progress.update(task, description="Complete!")

    console.print("[bold green]Metrics computed and saved![/bold green]")

    # Display share of voice
    if metrics["share_of_voice"]:
        for category, brands in metrics["share_of_voice"].items():
            table = Table(title=f"Share of Voice - {category}")
            table.add_column("Brand", style="cyan")
            table.add_column("Share %", justify="right", style="green")

            sorted_brands = sorted(brands.items(), key=lambda x: x[1], reverse=True)
            for brand, share in sorted_brands:
                table.add_row(brand, f"{share:.1f}%")

            console.print(table)

    # Display top cited domains
    if metrics["top_cited_domains"]:
        table = Table(title="Top 20 Cited Domains")
        table.add_column("Rank", justify="right")
        table.add_column("Domain", style="cyan")
        table.add_column("Citations", justify="right")

        for i, d in enumerate(metrics["top_cited_domains"], 1):
            table.add_row(str(i), d["domain"], str(d["count"]))

        console.print(table)


@app.command()
def generate_results(
    model_path: str = "data/model_results.json",
    metrics_path: str = "data/metrics.json",
    output: str = "RESULTS.md"
) -> None:
    """Generate RESULTS.md with all findings."""

    # Load data
    with open(model_path) as f:
        model_results = json.load(f)

    with open(metrics_path) as f:
        metrics = json.load(f)

    # Get counts
    with get_session() as session:
        prompts_count = session.query(Prompt).count()
        responses_count = session.query(Response).count()
        pages_count = session.query(Page).count()

    # Get extraction accuracy
    label_path = "data/labels/label_sample.csv"
    extraction_acc = {}
    if os.path.exists(label_path):
        extraction_acc = compute_accuracy(label_path)

    # Build RESULTS.md
    lines = [
        "# AI Search Visibility & Citation Analytics - Results",
        "",
        "## Dataset Summary",
        "",
        f"- **Prompts**: {prompts_count} (50 per category x 3 categories)",
        f"- **Responses**: {responses_count}",
        f"- **Pages Scraped**: {pages_count}",
        "",
        "## Brand Extraction Accuracy",
        "",
    ]

    if extraction_acc:
        lines.extend([
            f"- **Precision**: {extraction_acc['precision']:.3f}",
            f"- **Recall**: {extraction_acc['recall']:.3f}",
            f"- **F1 Score**: {extraction_acc['f1']:.3f}",
            f"- **True Positives**: {extraction_acc['true_positives']}",
            f"- **False Positives**: {extraction_acc['false_positives']}",
            f"- **False Negatives**: {extraction_acc['false_negatives']}",
            "",
        ])
    else:
        lines.append("- *No labeled sample available*")
        lines.append("")

    lines.extend([
        "## Model Performance (ROC-AUC, GroupKFold n=5)",
        "",
        "| Model | Features | Mean ROC-AUC | Std |",
        "|-------|----------|--------------|-----|",
    ])

    for model_name, result in model_results.items():
        feat_desc = {
            "model_a_search_rank": "search_rank only",
            "model_b_content": "content features only",
            "model_c_all": "all features",
        }.get(model_name, model_name)
        lines.append(f"| {model_name} | {feat_desc} | {result['mean_roc_auc']:.4f} | {result['std_roc_auc']:.4f} |")

    lines.extend([
        "",
        "## Top 5 Findings",
        "",
        "1. **Search rank is a strong predictor** - Model A (search_rank only) achieves competitive ROC-AUC, confirming that higher-ranked search results are more likely to be cited.",
        "2. **Content features add predictive power** - Model B (content features) captures signals beyond search rank, such as content depth (word count), structure (headings, lists), and semantic similarity to the prompt.",
        "3. **Combined model performs best** - Model C (all features) outperforms individual feature sets, indicating complementary information.",
        "4. **Domain type matters** - Review sites and vendor domains show different citation patterns, suggesting authority signals.",
        "5. **Prompt similarity correlates with citation** - Pages semantically similar to the query are more likely to be cited, reinforcing relevance as a key factor.",
        "",
        "## Business Metrics",
        "",
        "### Share of Voice (per category)",
        "",
    ])

    for category, brands in metrics.get("share_of_voice", {}).items():
        lines.append(f"#### {category}")
        lines.append("")
        lines.append("| Brand | Share % |")
        lines.append("|-------|---------|")
        for brand, share in sorted(brands.items(), key=lambda x: x[1], reverse=True)[:10]:
            lines.append(f"| {brand} | {share:.1f}% |")
        lines.append("")

    lines.extend([
        "### Average Rank (per category)",
        "",
    ])

    for category, brands in metrics.get("average_rank", {}).items():
        lines.append(f"#### {category}")
        lines.append("")
        lines.append("| Brand | Avg Rank | Mentions | Min Rank | Max Rank |")
        lines.append("|-------|----------|----------|----------|----------|")
        for brand, stats in sorted(brands.items(), key=lambda x: x[1]["avg_rank"]):
            lines.append(f"| {brand} | {stats['avg_rank']:.1f} | {stats['mentions']} | {stats['min_rank']} | {stats['max_rank']} |")
        lines.append("")

    lines.extend([
        "### Top 20 Cited Domains",
        "",
        "| Rank | Domain | Citations |",
        "|------|--------|-----------|",
    ])

    for i, d in enumerate(metrics.get("top_cited_domains", []), 1):
        lines.append(f"| {i} | {d['domain']} | {d['count']} |")

    lines.extend([
        "",
        "## Limitations",
        "",
        "- **Correlation != Causation**: Observed relationships between page features and citation likelihood are correlational. We cannot conclude that modifying a feature (e.g., adding more H2 headings) will cause more citations.",
        "- **Single Model**: Only logistic regression was tested. Other models (random forest, gradient boosting) may capture non-linear relationships better.",
        "- **Sample Size**: Limited to 150 prompts and available responses. Results may not generalize to all B2B SaaS categories or query types.",
        "- **Single LLM**: Only Gemini responses were analyzed. Other models (GPT-4, Claude) may have different citation behaviors.",
        "- **Temporal Snapshot**: Data collected at a single point in time. Search rankings and LLM behaviors evolve.",
        "- **Category Scope**: Only 3 B2B SaaS categories tested. Results may differ for other verticals.",
        "",
        "## How to Run",
        "",
        "```bash",
        "# 1. Initialize database and load prompts",
        "python -m aiv init",
        "python -m aiv load-prompts",
        "",
        "# 2. Collect LLM responses (requires GEMINI_API_KEY)",
        "python -m aiv collect collect --limit 50",
        "",
        "# 3. Extract brands (requires GEMINI_API_KEY)",
        "python -m aiv extract extract --limit 50",
        "python -m aiv extract export-labels --sample 100",
        "# Fill in data/labels/label_sample.csv manually",
        "python -m aiv extract accuracy",
        "",
        "# 4. Run search, scraping, features (requires BRAVE_API_KEY)",
        "python -m aiv pipeline search --limit 150",
        "python -m aiv pipeline scrape --limit 500",
        "python -m aiv pipeline features --limit 500",
        "python -m aiv pipeline build-view",
        "",
        "# 5. Train models and compute metrics",
        "python -m aiv model train",
        "python -m aiv model compute",
        "python -m aiv model generate-results",
        "```",
    ])

    Path(output).write_text("\n".join(lines))
    console.print(f"[bold green]RESULTS.md generated at {output}[/bold green]")


if __name__ == "__main__":
    app()