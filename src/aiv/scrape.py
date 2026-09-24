"""Web scraping with httpx + trafilatura + Playwright fallback"""
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import httpx
import trafilatura
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import get_settings
from .cache import cached, settings

logger = logging.getLogger(__name__)

# Known review sites for domain classification
REVIEW_SITES = {
    "g2.com", "capterra.com", "trustradius.com", "softwareadvice.com",
    "getapp.com", "financesonline.com", "techradar.com", "pcmag.com",
    "zdnet.com", "cnet.com", "techcrunch.com", "venturebeat.com",
}

# Vendor domains (common SaaS vendors)
VENDOR_SITES = {
    "salesforce.com", "hubspot.com", "microsoft.com", "google.com",
    "aws.amazon.com", "atlassian.com", "zoom.us", "slack.com",
    "notion.so", "github.com", "gitlab.com", "jira.com",
}


@dataclass
class ScrapedPage:
    """Result of scraping a URL"""
    url: str
    status: int
    title: Optional[str]
    text: Optional[str]
    html: Optional[str]


def _generate_cache_key(url: str) -> str:
    """Generate cache key for a URL"""
    return hashlib.sha256(url.encode()).hexdigest()[:32]


def _get_cache_path(url: str, suffix: str = ".html") -> Path:
    """Get cache file path for a URL"""
    key = _generate_cache_key(url)
    return settings.cache_dir / f"{key}{suffix}"


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
)
def _fetch_with_httpx(url: str, timeout: float = 30.0) -> Optional[httpx.Response]:
    """Fetch URL with httpx"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        return response


def _fetch_with_playwright(url: str, timeout: float = 30.0) -> Optional[str]:
    """Fetch URL with Playwright (for JS-rendered pages)"""
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        logger.warning(f"Playwright failed for {url}: {e}")
        return None


def _extract_with_trafilatura(html: str, url: str) -> tuple[Optional[str], Optional[str]]:
    """Extract title and text using trafilatura"""
    try:
        # Extract main content
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            url=url,
        )
        
        # Extract metadata for title
        metadata = trafilatura.extract_metadata(html)
        title = metadata.title if metadata and metadata.title else None
        
        return title, text
    except Exception as e:
        logger.warning(f"Trafilatura extraction failed for {url}: {e}")
        return None, None


def _extract_title_from_html(html: str) -> Optional[str]:
    """Fallback title extraction from raw HTML"""
    try:
        # Try <title> tag
        match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try og:title
        match = re.search(r'property="og:title"\s+content="([^"]+)"', html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    except Exception:
        pass
    return None


def scrape_url(url: str, use_playwright_fallback: bool = True) -> ScrapedPage:
    """Scrape a single URL with httpx + trafilatura, Playwright fallback"""
    cache_path = _get_cache_path(url)
    
    # Try cache first
    if cache_path.exists():
        try:
            html = cache_path.read_text(encoding="utf-8")
            title, text = _extract_with_trafilatura(html, url)
            if text:
                return ScrapedPage(url=url, status=200, title=title, text=text, html=html)
        except Exception:
            pass
    
    # Try httpx first
    try:
        response = _fetch_with_httpx(url)
        html = response.text
        status = response.status_code
        
        # Cache HTML
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(html, encoding="utf-8")
        
        title, text = _extract_with_trafilatura(html, url)
        if text:
            return ScrapedPage(url=url, status=status, title=title, text=text, html=html)
    except Exception as e:
        logger.warning(f"httpx failed for {url}: {e}")
    
    # Playwright fallback for JS-heavy pages
    if use_playwright_fallback:
        try:
            html = _fetch_with_playwright(url)
            if html:
                # Cache HTML
                cache_path.write_text(html, encoding="utf-8")
                title, text = _extract_with_trafilatura(html, url)
                if text:
                    return ScrapedPage(url=url, status=200, title=title, text=text, html=html)
        except Exception as e:
            logger.warning(f"Playwright fallback failed for {url}: {e}")
    
    # If we got HTML but no text, try to at least get title
    if 'html' in locals() and html:
        title = _extract_title_from_html(html)
        return ScrapedPage(url=url, status=200, title=title, text=None, html=html)
    
    return ScrapedPage(url=url, status=0, title=None, text=None, html=None)


def save_page(page: ScrapedPage) -> bool:
    """Save scraped page to database. Returns True if saved."""
    from .database import get_session
    from .models import Page
    
    if not page.text and not page.title:
        return False
    
    with get_session() as session:
        existing = session.query(Page).filter(Page.url == page.url).first()
        if existing:
            # Update if we have better content
            if page.text and (not existing.text or len(page.text) > len(existing.text)):
                existing.status = page.status
                existing.title = page.title
                existing.text = page.text
                existing.html = page.html
                session.commit()
            return False
        
        p = Page(
            url=page.url,
            status=page.status,
            title=page.title,
            text=page.text,
            html=page.html,
        )
        session.add(p)
        session.commit()
        return True


def scrape_urls(urls: list[str], max_workers: int = 5) -> dict:
    """Scrape multiple URLs in parallel. Returns stats."""
    stats = {"scraped": 0, "failed": 0, "skipped": 0}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(scrape_url, url): url for url in urls}
        
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                page = future.result()
                if save_page(page):
                    stats["scraped"] += 1
                else:
                    stats["failed"] += 1
                    logger.warning(f"Failed to save page: {url}")
            except Exception as e:
                stats["failed"] += 1
                logger.error(f"Error scraping {url}: {e}")
    
    return stats