"""Page feature extraction"""
import hashlib
import json
import logging
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import get_settings
from .cache import cached, settings
from .categories import get_all_brands

logger = logging.getLogger(__name__)

# Review sites for domain classification
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

# Reddit
REDDIT_DOMAINS = {"reddit.com", "old.reddit.com", "www.reddit.com"}

# News domains
NEWS_DOMAINS = {
    "techcrunch.com", "venturebeat.com", "theverge.com", "wired.com",
    "arstechnica.com", "engadget.com", "reuters.com", "bloomberg.com",
    "wsj.com", "nytimes.com", "ft.com", "economist.com",
}


@dataclass
class PageFeatures:
    """Extracted features for a page"""
    url: str
    word_count: int = 0
    h2_count: int = 0
    h3_count: int = 0
    list_count: int = 0
    has_table: int = 0
    has_faq: int = 0
    has_json_ld: int = 0
    json_ld_types: Optional[str] = None
    days_since_published: Optional[int] = None
    days_since_updated: Optional[int] = None
    domain_type: str = "other"
    category_brand_mentions: int = 0
    prompt_similarity: Optional[int] = None  # Scaled * 10000


# Global embedding model (lazy loaded)
_embedding_model: Optional[SentenceTransformer] = None


def _get_embedding_model() -> SentenceTransformer:
    """Get or create the embedding model"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


def _generate_cache_key(url: str) -> str:
    """Generate cache key for URL features"""
    return hashlib.sha256(f"features:{url}".encode()).hexdigest()[:32]


def _get_features_cache_path(url: str) -> Path:
    """Get cache file path for features"""
    key = _generate_cache_key(url)
    return settings.cache_dir / f"{key}.features.json"


def _classify_domain(url: str) -> str:
    """Classify domain type based on URL"""
    domain = urlparse(url).netloc.lower().replace("www.", "")
    
    if domain in REDDIT_DOMAINS:
        return "reddit"
    
    if domain in NEWS_DOMAINS:
        return "news"
    
    if domain in REVIEW_SITES:
        return "review_site"
    
    if domain in VENDOR_SITES:
        return "vendor"
    
    # Heuristic for blogs
    blog_indicators = ["blog", "medium.com", "substack.com", "wordpress.com", "ghost.io"]
    if any(indicator in domain for indicator in blog_indicators):
        return "blog"
    
    return "other"


def _extract_json_ld(html: str) -> tuple[int, Optional[str], Optional[datetime], Optional[datetime]]:
    """Extract JSON-LD structured data from HTML"""
    has_json_ld = 0
    json_ld_types = []
    published_date = None
    updated_date = None
    
    try:
        soup = BeautifulSoup(html, "html.parser")
        scripts = soup.find_all("script", type="application/ld+json")
        
        for script in scripts:
            try:
                data = json.loads(script.string)
                has_json_ld = 1
                
                # Handle both single object and array
                items = data if isinstance(data, list) else [data]
                
                for item in items:
                    if isinstance(item, dict):
                        # Extract @type
                        item_type = item.get("@type")
                        if item_type:
                            if isinstance(item_type, list):
                                json_ld_types.extend(item_type)
                            else:
                                json_ld_types.append(item_type)
                        
                        # Extract dates
                        if not published_date:
                            date_pub = item.get("datePublished")
                            if date_pub:
                                try:
                                    published_date = datetime.fromisoformat(date_pub.replace("Z", "+00:00"))
                                except Exception:
                                    pass
                        
                        if not updated_date:
                            date_mod = item.get("dateModified")
                            if date_mod:
                                try:
                                    updated_date = datetime.fromisoformat(date_mod.replace("Z", "+00:00"))
                                except Exception:
                                    pass
            except Exception:
                continue
    except Exception:
        pass
    
    # Deduplicate types
    json_ld_types = list(set(json_ld_types))
    types_str = ",".join(json_ld_types) if json_ld_types else None
    
    return has_json_ld, types_str, published_date, updated_date


def _extract_dates_from_meta(html: str) -> tuple[Optional[datetime], Optional[datetime]]:
    """Extract dates from meta tags as fallback"""
    published = None
    updated = None
    
    try:
        soup = BeautifulSoup(html, "html.parser")
        
        # Common meta tags for published date
        pub_selectors = [
            ('meta', {'property': 'article:published_time'}),
            ('meta', {'name': 'publish_date'}),
            ('meta', {'itemprop': 'datePublished'}),
            ('time', {'datetime': True}),
        ]
        
        for tag, attrs in pub_selectors:
            elem = soup.find(tag, attrs)
            if elem:
                date_str = elem.get('content') or elem.get('datetime') or elem.get_text()
                if date_str:
                    try:
                        published = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        break
                    except Exception:
                        continue
        
        # Common meta tags for updated date
        mod_selectors = [
            ('meta', {'property': 'article:modified_time'}),
            ('meta', {'name': 'lastmod'}),
            ('meta', {'itemprop': 'dateModified'}),
        ]
        
        for tag, attrs in mod_selectors:
            elem = soup.find(tag, attrs)
            if elem:
                date_str = elem.get('content') or elem.get('datetime')
                if date_str:
                    try:
                        updated = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        break
                    except Exception:
                        continue
    except Exception:
        pass
    
    return published, updated


def _parse_date_fallback(date_str: str) -> Optional[datetime]:
    """Try to parse date from various formats"""
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%B %d, %Y",
        "%d %B %Y",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except Exception:
            continue
    return None


def extract_features(
    url: str,
    html: str,
    text: str,
    prompt_text: Optional[str] = None,
    category: Optional[str] = None
) -> PageFeatures:
    """Extract all features for a page"""
    # Check cache
    cache_path = _get_features_cache_path(url)
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cached_data = json.load(f)
                # If prompt_similarity not cached, we need to recompute
                if prompt_text and cached_data.get("prompt_similarity") is None:
                    pass  # Will recompute below
                else:
                    return PageFeatures(**cached_data)
        except Exception:
            pass
    
    features = PageFeatures(url=url)
    
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    
    # Word count
    features.word_count = len(text.split()) if text else 0
    
    # Heading counts
    features.h2_count = len(soup.find_all("h2"))
    features.h3_count = len(soup.find_all("h3"))
    
    # List counts
    features.list_count = len(soup.find_all("ul")) + len(soup.find_all("ol"))
    
    # Has table
    features.has_table = 1 if soup.find("table") else 0
    
    # Has FAQ - check for FAQ schema or FAQ heading
    faq_found = False
    # Check JSON-LD for FAQ schema
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict) and item.get("@type") == "FAQPage":
                    faq_found = True
                    break
        except Exception:
            continue
    
    # Check for FAQ heading
    if not faq_found:
        for heading in soup.find_all(["h1", "h2", "h3", "h4"]):
            if "faq" in heading.get_text().lower() or "frequently asked" in heading.get_text().lower():
                faq_found = True
                break
    
    features.has_faq = 1 if faq_found else 0
    
    # JSON-LD extraction
    has_json_ld, json_ld_types, pub_date, mod_date = _extract_json_ld(html)
    features.has_json_ld = has_json_ld
    features.json_ld_types = json_ld_types
    
    # Dates - try JSON-LD first, then meta tags
    if not pub_date or not mod_date:
        meta_pub, meta_mod = _extract_dates_from_meta(html)
        if not pub_date:
            pub_date = meta_pub
        if not mod_date:
            mod_date = meta_mod
    
    now = datetime.now()
    if pub_date:
        features.days_since_published = (now - pub_date).days
    if mod_date:
        features.days_since_updated = (now - mod_date).days
    
    # Domain type
    features.domain_type = _classify_domain(url)
    
    # Category brand mentions
    if category:
        brands = get_all_brands()
        category_brands = [b for b in brands if any(cat in b.name.lower() for cat in [category.lower()])]
        # Actually, we need to filter by category properly
        from .categories import load_categories
        cats = load_categories()
        if category in cats:
            cat_brands = cats[category].brands
            all_aliases = []
            for b in cat_brands:
                all_aliases.extend(b.aliases)
            
            text_lower = text.lower()
            for alias in all_aliases:
                if alias.lower() in text_lower:
                    features.category_brand_mentions += 1
    
    # Prompt similarity (cosine similarity * 10000)
    if prompt_text and text:
        model = _get_embedding_model()
        embeddings = model.encode([prompt_text, text[:8000]])  # Limit text length
        similarity = float(np.dot(embeddings[0], embeddings[1]) / 
                          (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
        features.prompt_similarity = int(similarity * 10000)
    
    # Cache features
    try:
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(features.__dict__, f)
    except Exception:
        pass
    
    return features


def save_features(features: PageFeatures) -> bool:
    """Save features to database"""
    from .database import get_session
    from .models import PageFeature
    
    with get_session() as session:
        existing = session.query(PageFeature).filter(PageFeature.url == features.url).first()
        if existing:
            # Update
            for key, value in features.__dict__.items():
                setattr(existing, key, value)
        else:
            pf = PageFeature(**features.__dict__)
            session.add(pf)
        
        session.commit()
        return True


def extract_features_batch(
    urls: list[str],
    prompt_texts: dict[str, str],
    categories: dict[str, str],
    max_workers: int = 4
) -> dict:
    """Extract features for multiple URLs in parallel"""
    from .database import get_session
    from .models import Page
    
    stats = {"processed": 0, "failed": 0}
    
    # Load pages from DB
    with get_session() as session:
        pages = session.query(Page).filter(Page.url.in_(urls)).all()
        page_map = {p.url: (p.html, p.text) for p in pages}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {}
        
        for url in urls:
            if url not in page_map:
                stats["failed"] += 1
                continue
            
            html, text = page_map[url]
            if not text:
                stats["failed"] += 1
                continue
            
            prompt_text = prompt_texts.get(url)
            category = categories.get(url)
            
            future = executor.submit(extract_features, url, html, text, prompt_text, category)
            future_to_url[future] = url
        
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                features = future.result()
                save_features(features)
                stats["processed"] += 1
            except Exception as e:
                stats["failed"] += 1
                logger.error(f"Feature extraction failed for {url}: {e}")
    
    return stats