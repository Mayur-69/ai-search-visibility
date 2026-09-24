"""Brave Search API integration"""
import json
import hashlib
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from brave import Brave
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import get_settings
from .cache import cached, _cache_key, _cache_path

settings = get_settings()


@dataclass
class SearchResultItem:
    """A single search result from Brave"""
    url: str
    domain: str
    title: str
    snippet: str
    search_rank: int


def _get_brave_client() -> Brave:
    """Create Brave Search client"""
    return Brave(api_key=settings.brave_api_key)


def _generate_cache_key(prompt_id: int, query: str) -> str:
    """Generate cache key for search query"""
    key_data = f"brave_search:{prompt_id}:{query}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception),
)
@cached
def _call_brave_search(query: str, count: int = 10) -> dict:
    """Call Brave Search API with caching and retries"""
    client = _get_brave_client()
    return client.search(q=query, count=count)


def _parse_brave_results(raw_response: dict) -> list[SearchResultItem]:
    """Parse Brave Search response into structured results"""
    results = []
    
    web_results = raw_response.get("web", {}).get("results", [])
    for i, result in enumerate(web_results):
        url = result.get("url", "")
        if not url:
            continue
        
        domain = urlparse(url).netloc.replace("www.", "")
        results.append(SearchResultItem(
            url=url,
            domain=domain,
            title=result.get("title", ""),
            snippet=result.get("description", ""),
            search_rank=i + 1,
        ))
    
    return results


def search_prompt(prompt_id: int, query: str, count: int = 10) -> list[SearchResultItem]:
    """Search for a prompt and return parsed results"""
    raw = _call_brave_search(query, count)
    return _parse_brave_results(raw)


def save_search_results(prompt_id: int, results: list[SearchResultItem]) -> int:
    """Save search results to database. Returns count saved."""
    from .database import get_session
    from .models import SearchResult
    
    saved = 0
    with get_session() as session:
        for item in results:
            # Check for existing
            existing = session.query(SearchResult).filter(
                SearchResult.prompt_id == prompt_id,
                SearchResult.url == item.url
            ).first()
            if existing:
                continue
            
            sr = SearchResult(
                prompt_id=prompt_id,
                url=item.url,
                domain=item.domain,
                search_rank=item.search_rank,
            )
            session.add(sr)
            saved += 1
        
        session.commit()
    
    return saved