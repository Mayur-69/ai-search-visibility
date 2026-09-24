"""Command-line interface for AI Search Visibility & Citation Analytics"""
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track

from .database import init_db, get_session
from .models import Prompt
from .categories import load_categories, get_all_prompts
from .collect import app as collect_app
from .extract_cli import app as extract_app
from .pipeline_cli import app as pipeline_app
from .model_cli import app as model_app

app = typer.Typer(help="AI Search Visibility & Citation Analytics CLI")
console = Console()

app.add_typer(collect_app, name="collect")
app.add_typer(extract_app, name="extract")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(model_app, name="model")


@app.command()
def init() -> None:
    """Initialize the database (create tables)"""
    console.print("[bold green]Initializing database...[/bold green]")
    init_db()
    console.print("[bold green]Database initialized successfully![/bold green]")


@app.command()
def load_prompts() -> None:
    """Load prompts from categories.yaml into the database"""
    console.print("[bold blue]Loading prompts from config/categories.yaml...[/bold blue]")
    
    categories = load_categories()
    total_prompts = sum(len(cat.prompts) for cat in categories.values())
    
    with get_session() as session:
        # Check existing prompts
        existing_count = session.query(Prompt).count()
        if existing_count > 0:
            console.print(f"[yellow]Database already has {existing_count} prompts. Skipping...[/yellow]")
            return
        
        loaded = 0
        for category in categories.values():
            for prompt_data in track(category.prompts, description=f"Loading {category.name}..."):
                prompt = Prompt(
                    category=prompt_data.category,
                    text=prompt_data.text,
                    intent=prompt_data.intent,
                )
                session.add(prompt)
                loaded += 1
        
        session.commit()
    
    console.print(f"[bold green]Loaded {loaded} prompts into database![/bold green]")


@app.command()
def list_categories() -> None:
    """List all categories and their prompt counts"""
    categories = load_categories()
    
    table = Table(title="Categories")
    table.add_column("Category", style="cyan")
    table.add_column("Brands", justify="right")
    table.add_column("Prompts", justify="right")
    table.add_column("Intents", style="dim")
    
    for cat in categories.values():
        intents = set(p.intent for p in cat.prompts)
        table.add_row(cat.name, str(len(cat.brands)), str(len(cat.prompts)), ", ".join(sorted(intents)))
    
    console.print(table)


@app.command()
def list_prompts(category: str = typer.Option(None, "--category", "-c", help="Filter by category")) -> None:
    """List all prompts"""
    prompts = get_all_prompts()
    
    if category:
        prompts = [p for p in prompts if p.category == category]
    
    table = Table(title="Prompts")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Category", style="cyan")
    table.add_column("Intent", style="green")
    table.add_column("Text")
    
    for i, prompt in enumerate(prompts, 1):
        table.add_row(str(i), prompt.category, prompt.intent, prompt.text[:80] + "..." if len(prompt.text) > 80 else prompt.text)
    
    console.print(table)


@app.command()
def stats() -> None:
    """Show database statistics"""
    with get_session() as session:
        prompts_count = session.query(Prompt).count()
        
        # Count by category
        from sqlalchemy import func
        category_counts = session.query(Prompt.category, func.count(Prompt.id)).group_by(Prompt.category).all()
        intent_counts = session.query(Prompt.intent, func.count(Prompt.id)).group_by(Prompt.intent).all()
    
    console.print(f"[bold]Total Prompts:[/bold] {prompts_count}")
    
    cat_table = Table(title="By Category")
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Count", justify="right")
    for cat, count in category_counts:
        cat_table.add_row(cat, str(count))
    console.print(cat_table)
    
    intent_table = Table(title="By Intent")
    intent_table.add_column("Intent", style="green")
    intent_table.add_column("Count", justify="right")
    for intent, count in intent_counts:
        intent_table.add_row(intent, str(count))
    console.print(intent_table)


if __name__ == "__main__":
    app()