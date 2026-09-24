"""CLI commands for brand extraction and labeling"""
import typer
import csv
import random
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from .database import get_session
from .models import Response, Mention
from .extract import extract_brands, save_mentions
from .config import get_settings

app = typer.Typer(help="Brand extraction and labeling")
console = Console()
settings = get_settings()


def extract_all(limit: int = None, model: str = "gemini-2.5-flash") -> dict:
    """Extract brands from responses. Returns statistics."""
    stats = {
        "responses_processed": 0,
        "mentions_saved": 0,
        "errors": 0,
    }
    
    with get_session() as session:
        query = session.query(Response).order_by(Response.id)
        if limit:
            query = query.limit(limit)
        # Load response data before session closes
        responses = [(r.id, r.text) for r in query.all()]
    
    if not responses:
        console.print("[yellow]No responses found in database[/yellow]")
        return stats
    
    console.print(f"[bold]Extracting brands from {len(responses)} responses...[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting...", total=len(responses))
        
        for response_id, response_text in responses:
            progress.update(task, description=f"Response {response_id}...")
            
            try:
                mentions = extract_brands(response_id, response_text, model)
                saved = save_mentions(response_id, mentions)
                
                stats["responses_processed"] += 1
                stats["mentions_saved"] += saved
                
            except Exception as e:
                console.print(f"[red]Error on response {response_id}: {e}[/red]")
                stats["errors"] += 1
            
            progress.advance(task)
    
    return stats


def export_label_sample(sample_size: int = 100, output_path: str = "data/labels/label_sample.csv") -> int:
    """Export random sample of responses for manual labeling."""
    with get_session() as session:
        # Get responses with their mentions - load all data in session
        from sqlalchemy.orm import joinedload
        responses = session.query(Response).options(
            joinedload(Response.mentions)
        ).order_by(Response.id).all()
        
        # Also load prompt categories
        from .models import Prompt
        prompt_categories = {}
        prompts = session.query(Prompt).all()
        for p in prompts:
            prompt_categories[p.id] = p.category
        
        # Extract data we need before session closes
        response_data = []
        for resp in responses:
            mentions = sorted(resp.mentions, key=lambda m: m.rank) if resp.mentions else []
            extracted = "; ".join([f"{m.brand} (rank={m.rank}, sent={m.sentiment})" for m in mentions])
            category = prompt_categories.get(resp.prompt_id, "unknown")
            
            response_data.append({
                'response_id': resp.id,
                'prompt_id': resp.prompt_id,
                'category': category,
                'response_text': resp.text[:500],
                'extracted_brands': extracted,
            })
    
    if not response_data:
        console.print("[yellow]No responses found[/yellow]")
        return 0
    
    # Sample
    sample = random.sample(response_data, min(sample_size, len(response_data)))
    
    # Ensure output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['response_id', 'prompt_id', 'category', 'response_text', 'extracted_brands', 'correct_brands'])
        
        for row in sample:
            writer.writerow([
                row['response_id'],
                row['prompt_id'],
                row['category'],
                row['response_text'],
                row['extracted_brands'],
                ""  # Empty column for manual labeling
            ])
    
    console.print(f"[bold green]Exported {len(sample)} responses to {output_path}[/bold green]")
    return len(sample)


def compute_accuracy(label_path: str = "data/labels/label_sample.csv") -> dict:
    """Compute precision and recall against manual labels."""
    from .categories import normalize_brand
    
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    with open(label_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse extracted brands
            extracted_raw = row['extracted_brands']
            extracted = set()
            if extracted_raw:
                for part in extracted_raw.split("; "):
                    brand = part.split(" (rank=")[0]
                    canonical = normalize_brand(brand)
                    if canonical:
                        extracted.add(canonical)
            
            # Parse correct brands (manual labels)
            correct_raw = row['correct_brands']
            correct = set()
            if correct_raw:
                for brand in correct_raw.split("; "):
                    canonical = normalize_brand(brand.strip())
                    if canonical:
                        correct.add(canonical)
            
            # Compute
            tp += len(extracted & correct)
            fp += len(extracted - correct)
            fn += len(correct - extracted)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@app.command()
def extract(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of responses to process"),
    model: str = typer.Option("gemini-2.5-flash", "--model", "-m", help="Gemini model to use"),
) -> None:
    """Extract brands from LLM responses using structured output."""
    if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
        console.print("[red]Error: GEMINI_API_KEY not set in .env[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold]Starting extraction with model: {model}[/bold]")
    if limit:
        console.print(f"[bold]Limit: {limit} responses[/bold]")
    
    stats = extract_all(limit=limit, model=model)
    
    table = Table(title="Extraction Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="green")
    
    table.add_row("Responses Processed", str(stats["responses_processed"]))
    table.add_row("Mentions Saved", str(stats["mentions_saved"]))
    table.add_row("Errors", str(stats["errors"]))
    
    console.print(table)
    
    if stats["errors"] > 0:
        console.print(f"[yellow]Completed with {stats['errors']} errors[/yellow]")
    else:
        console.print("[bold green]Extraction completed successfully![/bold green]")


@app.command()
def export_labels(
    sample_size: int = typer.Option(100, "--sample", "-s", help="Number of responses to sample"),
    output: str = typer.Option("data/labels/label_sample.csv", "--output", "-o", help="Output CSV path"),
) -> None:
    """Export random sample of responses for manual labeling."""
    export_label_sample(sample_size=sample_size, output_path=output)


@app.command()
def accuracy(
    label_path: str = typer.Option("data/labels/label_sample.csv", "--path", "-p", help="Path to labeled CSV"),
) -> None:
    """Compute precision and recall against manual labels."""
    import os
    if not os.path.exists(label_path):
        console.print(f"[red]Label file not found: {label_path}[/red]")
        raise typer.Exit(1)
    
    stats = compute_accuracy(label_path)
    
    table = Table(title="Brand Extraction Accuracy")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    
    table.add_row("True Positives", str(stats["true_positives"]))
    table.add_row("False Positives", str(stats["false_positives"]))
    table.add_row("False Negatives", str(stats["false_negatives"]))
    table.add_row("Precision", f"{stats['precision']:.3f}")
    table.add_row("Recall", f"{stats['recall']:.3f}")
    table.add_row("F1 Score", f"{stats['f1']:.3f}")
    
    console.print(table)


@app.command()
def show_mentions(limit: int = 20) -> None:
    """Show recently extracted mentions."""
    with get_session() as session:
        mentions = session.query(Mention).order_by(Mention.id.desc()).limit(limit).all()
        
        # Extract data before session closes
        mention_data = [(m.id, m.response_id, m.brand, m.rank, m.sentiment) for m in mentions]
    
    if not mention_data:
        console.print("[yellow]No mentions found[/yellow]")
        return
    
    table = Table(title=f"Recent Mentions (last {limit})")
    table.add_column("ID", justify="right")
    table.add_column("Response ID", justify="right")
    table.add_column("Brand")
    table.add_column("Rank", justify="right")
    table.add_column("Sentiment")
    
    for m_id, resp_id, brand, rank, sentiment in mention_data:
        table.add_row(str(m_id), str(resp_id), brand, str(rank), sentiment)
    
    console.print(table)


if __name__ == "__main__":
    app()