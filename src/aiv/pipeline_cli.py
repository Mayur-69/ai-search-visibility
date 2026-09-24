"""CLI commands for search, scraping, features, and view building"""
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from sqlalchemy import func

from .database import get_session
from .models import Prompt, Response, Citation, SearchResult, Page, PageFeature
from .search import search_prompt, save_search_results
from .scrape import scrape_urls, save_page, ScrapedPage
from .features import extract_features_batch
from .config import get_settings

app = typer.Typer(help="Search, scraping, and feature extraction")
console = Console()
settings = get_settings()


# ============ SEARCH COMMANDS ============

def run_search(limit: int = None) -> dict:
    """Run Brave Search for prompts"""
    stats = {"prompts_processed": 0, "results_saved": 0, "errors": 0}
    
    with get_session() as session:
        query = session.query(Prompt).order_by(Prompt.id)
        if limit:
            query = query.limit(limit)
        prompts = [(p.id, p.text) for p in query.all()]
    
    if not prompts:
        console.print("[yellow]No prompts found[/yellow]")
        return stats
    
    console.print(f"[bold]Searching for {len(prompts)} prompts...[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Searching...", total=len(prompts))
        
        for prompt_id, prompt_text in prompts:
            progress.update(task, description=f"Prompt {prompt_id}: {prompt_text[:50]}...")
            
            try:
                results = search_prompt(prompt_id, prompt_text)
                saved = save_search_results(prompt_id, results)
                
                stats["prompts_processed"] += 1
                stats["results_saved"] += saved
                
            except Exception as e:
                console.print(f"[red]Error on prompt {prompt_id}: {e}[/red]")
                stats["errors"] += 1
            
            progress.advance(task)
    
    return stats


@app.command()
def search(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of prompts to process"),
) -> None:
    """Run Brave Search for prompts and save results."""
    if not settings.brave_api_key or settings.brave_api_key == "your_brave_api_key_here":
        console.print("[red]Error: BRAVE_API_KEY not set in .env[/red]")
        raise typer.Exit(1)
    
    console.print("[bold]Starting Brave Search...[/bold]")
    if limit:
        console.print(f"[bold]Limit: {limit} prompts[/bold]")
    
    stats = run_search(limit=limit)
    
    table = Table(title="Search Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="green")
    
    table.add_row("Prompts Processed", str(stats["prompts_processed"]))
    table.add_row("Results Saved", str(stats["results_saved"]))
    table.add_row("Errors", str(stats["errors"]))
    
    console.print(table)


@app.command()
def search_stats() -> None:
    """Show search results statistics."""
    from sqlalchemy import func
    
    with get_session() as session:
        total = session.query(SearchResult).count()
        by_prompt = session.query(SearchResult.prompt_id, func.count(SearchResult.id)).group_by(SearchResult.prompt_id).all()
        unique_domains = session.query(func.count(func.distinct(SearchResult.domain))).scalar()
    
    console.print(f"[bold]Total Search Results:[/bold] {total}")
    console.print(f"[bold]Unique Domains:[/bold] {unique_domains}")
    console.print(f"[bold]Prompts with Results:[/bold] {len(by_prompt)}")
    
    # Show distribution
    counts = [c for _, c in by_prompt]
    if counts:
        console.print(f"  Avg per prompt: {sum(counts)/len(counts):.1f}")
        console.print(f"  Min: {min(counts)}, Max: {max(counts)}")


# ============ SCRAPE COMMANDS ============

def get_urls_to_scrape(limit: int = None) -> list[str]:
    """Get unique URLs from citations and search_results"""
    with get_session() as session:
        # Get URLs from citations
        citation_urls = [r[0] for r in session.query(Citation.url).distinct().all()]
        
        # Get URLs from search_results
        search_urls = [r[0] for r in session.query(SearchResult.url).distinct().all()]
        
        # Combine and deduplicate
        all_urls = list(set(citation_urls + search_urls))
        
        # Filter out already scraped
        scraped_urls = [r[0] for r in session.query(Page.url).all()]
        urls_to_scrape = [u for u in all_urls if u not in scraped_urls]
        
        if limit:
            urls_to_scrape = urls_to_scrape[:limit]
        
        return urls_to_scrape


def run_scrape(limit: int = None, max_workers: int = 5) -> dict:
    """Scrape URLs"""
    urls = get_urls_to_scrape(limit)
    
    if not urls:
        console.print("[yellow]No URLs to scrape[/yellow]")
        return {"scraped": 0, "failed": 0, "skipped": 0}
    
    console.print(f"[bold]Scraping {len(urls)} URLs...[/bold]")
    
    stats = scrape_urls(urls, max_workers=max_workers)
    return stats


@app.command()
def scrape(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of URLs to scrape"),
    workers: int = typer.Option(5, "--workers", "-w", help="Max parallel workers"),
) -> None:
    """Scrape URLs from citations and search results."""
    console.print("[bold]Starting scraping...[/bold]")
    if limit:
        console.print(f"[bold]Limit: {limit} URLs[/bold]")
    console.print(f"[bold]Workers: {workers}[/bold]")
    
    stats = run_scrape(limit=limit, max_workers=workers)
    
    table = Table(title="Scraping Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="green")
    
    table.add_row("Scraped", str(stats["scraped"]))
    table.add_row("Failed", str(stats["failed"]))
    table.add_row("Skipped", str(stats["skipped"]))
    
    console.print(table)


@app.command()
def scrape_stats() -> None:
    """Show scraping statistics."""
    with get_session() as session:
        total = session.query(Page).count()
        with_text = session.query(Page).filter(Page.text.isnot(None)).count()
        by_status = session.query(Page.status, func.count(Page.url)).group_by(Page.status).all()
    
    console.print(f"[bold]Total Pages:[/bold] {total}")
    console.print(f"[bold]Pages with Text:[/bold] {with_text}")
    
    table = Table(title="By Status")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right")
    for status, count in by_status:
        table.add_row(str(status), str(count))
    console.print(table)


# ============ FEATURES COMMANDS ============

def get_urls_for_features(limit: int = None) -> list[str]:
    """Get URLs that have page content but no features yet"""
    with get_session() as session:
        # Get pages with text
        pages_with_text = session.query(Page.url).filter(Page.text.isnot(None)).all()
        page_urls = [r[0] for r in pages_with_text]
        
        # Get URLs that already have features
        featured_urls = [r[0] for r in session.query(PageFeature.url).all()]
        
        urls = [u for u in page_urls if u not in featured_urls]
        
        if limit:
            urls = urls[:limit]
        
        return urls


def get_prompt_texts_and_categories(urls: list[str]) -> tuple[dict, dict]:
    """Get prompt text and category for each URL via search_results and citations"""
    prompt_texts = {}
    categories = {}
    
    with get_session() as session:
        # Via search_results
        search_results = session.query(SearchResult.url, SearchResult.prompt_id).filter(
            SearchResult.url.in_(urls)
        ).all()
        
        prompt_ids = set(sr.prompt_id for sr in search_results)
        prompts = session.query(Prompt.id, Prompt.text, Prompt.category).filter(
            Prompt.id.in_(prompt_ids)
        ).all()
        prompt_map = {p.id: (p.text, p.category) for p in prompts}
        
        for sr in search_results:
            if sr.prompt_id in prompt_map:
                prompt_texts[sr.url] = prompt_map[sr.prompt_id][0]
                categories[sr.url] = prompt_map[sr.prompt_id][1]
        
        # Via citations -> responses -> prompts
        citations = session.query(Citation.url, Response.prompt_id).join(
            Response, Citation.response_id == Response.id
        ).filter(Citation.url.in_(urls)).all()
        
        for c in citations:
            if c.prompt_id in prompt_map:
                prompt_texts[c.url] = prompt_map[c.prompt_id][0]
                categories[c.url] = prompt_map[c.prompt_id][1]
    
    return prompt_texts, categories


def run_features(limit: int = None, max_workers: int = 4) -> dict:
    """Extract features for pages"""
    urls = get_urls_for_features(limit)
    
    if not urls:
        console.print("[yellow]No URLs need feature extraction[/yellow]")
        return {"processed": 0, "failed": 0}
    
    console.print(f"[bold]Extracting features for {len(urls)} URLs...[/bold]")
    
    prompt_texts, categories = get_prompt_texts_and_categories(urls)
    
    stats = extract_features_batch(
        urls=urls,
        prompt_texts=prompt_texts,
        categories=categories,
        max_workers=max_workers
    )
    
    return stats


@app.command()
def features(
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of URLs to process"),
    workers: int = typer.Option(4, "--workers", "-w", help="Max parallel workers"),
) -> None:
    """Extract features for scraped pages."""
    console.print("[bold]Starting feature extraction...[/bold]")
    if limit:
        console.print(f"[bold]Limit: {limit} URLs[/bold]")
    console.print(f"[bold]Workers: {workers}[/bold]")
    
    stats = run_features(limit=limit, max_workers=workers)
    
    table = Table(title="Feature Extraction Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="green")
    
    table.add_row("Processed", str(stats["processed"]))
    table.add_row("Failed", str(stats["failed"]))
    
    console.print(table)


@app.command()
def features_stats() -> None:
    """Show feature extraction statistics."""
    from sqlalchemy import func
    
    with get_session() as session:
        total = session.query(PageFeature).count()
        by_domain_type = session.query(PageFeature.domain_type, func.count(PageFeature.url)).group_by(PageFeature.domain_type).all()
    
    console.print(f"[bold]Total Pages with Features:[/bold] {total}")
    
    table = Table(title="By Domain Type")
    table.add_column("Domain Type", style="cyan")
    table.add_column("Count", justify="right")
    for dtype, count in by_domain_type:
        table.add_row(dtype, str(count))
    console.print(table)


# ============ VIEW BUILDING ============

@app.command()
def build_view() -> None:
    """Create the labeled_dataset SQL view."""
    console.print("[bold]Building labeled_dataset view...[/bold]")
    
    view_sql = """
    CREATE VIEW IF NOT EXISTS labeled_dataset AS
    WITH citation_urls AS (
        SELECT DISTINCT c.url, r.prompt_id
        FROM citations c
        JOIN responses r ON c.response_id = r.id
    ),
    search_urls AS (
        SELECT DISTINCT sr.url, sr.prompt_id, sr.search_rank
        FROM search_results sr
    ),
    all_urls AS (
        SELECT url, prompt_id, search_rank FROM search_urls
        UNION
        SELECT url, prompt_id, NULL as search_rank FROM citation_urls
    ),
    cited_flags AS (
        SELECT DISTINCT c.url, r.prompt_id, 1 as cited
        FROM citations c
        JOIN responses r ON c.response_id = r.id
    )
    SELECT 
        pf.*,
        au.prompt_id,
        au.search_rank,
        COALESCE(cf.cited, 0) as cited
    FROM page_features pf
    JOIN all_urls au ON pf.url = au.url
    LEFT JOIN cited_flags cf ON cf.url = au.url AND cf.prompt_id = au.prompt_id
    """
    
    with get_session() as session:
        from sqlalchemy import text
        # Drop view first (SQLite doesn't support CREATE OR REPLACE VIEW)
        session.execute(text("DROP VIEW IF EXISTS labeled_dataset"))
        session.execute(text(view_sql))
        session.commit()
    
    console.print("[bold green]View created successfully![/bold green]")
    
    # Show sample
    with get_session() as session:
        from sqlalchemy import text
        result = session.execute(text("SELECT * FROM labeled_dataset LIMIT 5")).fetchall()
    
    if result:
        table = Table(title="labeled_dataset Sample (first 5 rows)")
        cols = result[0].keys() if hasattr(result[0], 'keys') else [f"col_{i}" for i in range(len(result[0]))]
        for col in cols:
            table.add_column(col)
        for row in result:
            table.add_row(*[str(v) for v in row])
        console.print(table)


@app.command()
def view_stats() -> None:
    """Show labeled_dataset view statistics."""
    with get_session() as session:
        from sqlalchemy import text, func
        
        total = session.execute(text("SELECT COUNT(*) FROM labeled_dataset")).scalar()
        cited_count = session.execute(text("SELECT COUNT(*) FROM labeled_dataset WHERE cited = 1")).scalar()
        by_category = session.execute(text("""
            SELECT p.category, COUNT(*) 
            FROM labeled_dataset ld
            JOIN prompts p ON ld.prompt_id = p.id
            GROUP BY p.category
        """)).fetchall()
        by_domain_type = session.execute(text("""
            SELECT domain_type, COUNT(*) 
            FROM labeled_dataset 
            GROUP BY domain_type
        """)).fetchall()
    
    console.print(f"[bold]Total Rows:[/bold] {total}")
    console.print(f"[bold]Cited:[/bold] {cited_count} ({cited_count/total*100:.1f}%)" if total else "")
    
    table = Table(title="By Category")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    for cat, count in by_category:
        table.add_row(cat, str(count))
    console.print(table)
    
    table2 = Table(title="By Domain Type")
    table2.add_column("Domain Type", style="cyan")
    table2.add_column("Count", justify="right")
    for dtype, count in by_domain_type:
        table2.add_row(dtype, str(count))
    console.print(table2)


if __name__ == "__main__":
    app()