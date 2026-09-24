"""Collect LLM responses with Gemini + Google Search grounding"""
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from .database import get_session
from .models import Prompt, Response, Citation
from .gemini_client import collect_response, CollectResult
from .config import get_settings

app = typer.Typer(help="Collect LLM responses with grounding")
console = Console()
settings = get_settings()


def save_response(result: CollectResult) -> int:
    """Save response and citations to database. Returns response ID."""
    with get_session() as session:
        # Create response
        response = Response(
            prompt_id=result.prompt_id,
            model=result.model,
            text=result.text,
            raw_json=result.raw_json,
        )
        session.add(response)
        session.flush()  # Get the ID
        
        # Create citations
        for i, source in enumerate(result.grounding_sources):
            citation = Citation(
                response_id=response.id,
                url=source.url,
                domain=source.domain,
                position=i + 1,
            )
            session.add(citation)
        
        session.commit()
        return response.id


def collect_prompts(limit: int = None, model: str = "gemini-1.5-pro") -> dict:
    """Collect responses for prompts. Returns statistics."""
    stats = {
        "prompts_processed": 0,
        "responses_saved": 0,
        "citations_saved": 0,
        "errors": 0,
    }
    
    with get_session() as session:
        query = session.query(Prompt).order_by(Prompt.id)
        if limit:
            query = query.limit(limit)
        # Load prompt data before session closes
        prompts = [(p.id, p.text) for p in query.all()]
    
    if not prompts:
        console.print("[yellow]No prompts found in database[/yellow]")
        return stats
    
    console.print(f"[bold]Collecting responses for {len(prompts)} prompts...[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Collecting...", total=len(prompts))
        
        for prompt_id, prompt_text in prompts:
            progress.update(task, description=f"Prompt {prompt_id}: {prompt_text[:50]}...")
            
            try:
                result = collect_response(prompt_id, prompt_text, model)
                response_id = save_response(result)
                
                stats["prompts_processed"] += 1
                stats["responses_saved"] += 1
                stats["citations_saved"] += len(result.grounding_sources)
                
            except Exception as e:
                console.print(f"[red]Error on prompt {prompt_id}: {e}[/red]")
                stats["errors"] += 1
            
            progress.advance(task)
    
    return stats


@app.command()
def collect(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of prompts to process"),
    model: str = typer.Option("gemini-1.5-pro", "--model", "-m", help="Gemini model to use"),
) -> None:
    """Collect LLM responses with Google Search grounding for prompts."""
    if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
        console.print("[red]Error: GEMINI_API_KEY not set in .env[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold]Starting collection with model: {model}[/bold]")
    if limit:
        console.print(f"[bold]Limit: {limit} prompts[/bold]")
    
    stats = collect_prompts(limit=limit, model=model)
    
    # Display results
    table = Table(title="Collection Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="green")
    
    table.add_row("Prompts Processed", str(stats["prompts_processed"]))
    table.add_row("Responses Saved", str(stats["responses_saved"]))
    table.add_row("Citations Saved", str(stats["citations_saved"]))
    table.add_row("Errors", str(stats["errors"]))
    
    console.print(table)
    
    if stats["errors"] > 0:
        console.print(f"[yellow]Completed with {stats['errors']} errors[/yellow]")
    else:
        console.print("[bold green]Collection completed successfully![/bold green]")


@app.command()
def show_responses(limit: int = 10) -> None:
    """Show recently collected responses."""
    with get_session() as session:
        responses = session.query(Response).order_by(Response.id.desc()).limit(limit).all()
    
    if not responses:
        console.print("[yellow]No responses found[/yellow]")
        return
    
    table = Table(title=f"Recent Responses (last {limit})")
    table.add_column("ID", justify="right")
    table.add_column("Prompt ID", justify="right")
    table.add_column("Model")
    table.add_column("Text Preview")
    table.add_column("Citations", justify="right")
    
    for resp in responses:
        with get_session() as session:
            cite_count = session.query(Citation).filter(Citation.response_id == resp.id).count()
        
        preview = resp.text[:100] + "..." if len(resp.text) > 100 else resp.text
        table.add_row(str(resp.id), str(resp.prompt_id), resp.model, preview, str(cite_count))
    
    console.print(table)


if __name__ == "__main__":
    app()