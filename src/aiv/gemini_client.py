"""Gemini client with Google Search grounding"""
import json
import hashlib
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import httpx
from google import genai
from google.genai import types

from .config import get_settings
from .cache import cached, RateLimiter, retry_with_backoff

settings = get_settings()


@dataclass
class GroundingSource:
    """A grounding source from Gemini response"""
    url: str
    domain: str
    title: Optional[str] = None


@dataclass
class CollectResult:
    """Result of collecting a response for a prompt"""
    prompt_id: int
    model: str
    text: str
    raw_json: str
    grounding_sources: list[GroundingSource]


def _get_gemini_client() -> genai.Client:
    """Create Gemini client with API key"""
    return genai.Client(api_key=settings.gemini_api_key)


def _generate_cache_key(prompt_id: int, prompt_text: str, model: str) -> str:
    """Generate cache key for a prompt+model combination"""
    key_data = f"{prompt_id}:{model}:{prompt_text}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


_rate_limiter = RateLimiter(settings.gemini_rpm)


@retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=60.0, retry_on=(Exception,))
@cached
def _call_gemini_with_grounding(prompt_text: str, model: str = "gemini-1.5-pro") -> dict:
    """Call Gemini with Google Search grounding enabled. Returns raw API response as dict."""
    _rate_limiter.wait()
    
    client = _get_gemini_client()
    
    # Configure grounding with Google Search
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )
    
    config = types.GenerateContentConfig(
        tools=[grounding_tool],
        temperature=0.1,
        max_output_tokens=8192,
    )
    
    response = client.models.generate_content(
        model=model,
        contents=prompt_text,
        config=config,
    )
    
    # Convert to dict for caching
    return json.loads(response.model_dump_json())


def _extract_grounding_sources(response_dict: dict) -> list[GroundingSource]:
    """Extract grounding source URLs from Gemini response"""
    sources = []
    
    # Navigate the response structure to find grounding metadata
    candidates = response_dict.get("candidates", [])
    for candidate in candidates:
        grounding_metadata = candidate.get("grounding_metadata", {})
        grounding_chunks = grounding_metadata.get("grounding_chunks", [])
        
        for chunk in grounding_chunks:
            web = chunk.get("web", {})
            url = web.get("uri", "")
            title = web.get("title", "")
            
            if url:
                # Domain will be resolved after following redirects
                domain = urlparse(url).netloc.replace("www.", "")
                sources.append(GroundingSource(url=url, domain=domain, title=title))
    
    return sources


def _extract_answer_text(response_dict: dict) -> str:
    """Extract the answer text from Gemini response"""
    candidates = response_dict.get("candidates", [])
    if not candidates:
        return ""
    
    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    
    text_parts = []
    for part in parts:
        if "text" in part:
            text_parts.append(part["text"])
    
    return "\n".join(text_parts)


@retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0, retry_on=(httpx.HTTPError,))
def _resolve_url(url: str, timeout: float = 10.0) -> str:
    """Follow redirects to get final URL. Returns final URL or original on failure."""
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            response = client.head(url, headers={"User-Agent": "Mozilla/5.0 (compatible; AIV/0.1)"})
            return str(response.url)
    except Exception:
        return url


def collect_response(prompt_id: int, prompt_text: str, model: str = "gemini-1.5-pro") -> CollectResult:
    """Collect a single response for a prompt using Gemini with grounding."""
    # Call Gemini (cached)
    raw_response = _call_gemini_with_grounding(prompt_text, model)
    
    # Extract answer text
    answer_text = _extract_answer_text(raw_response)
    
    # Extract grounding sources
    raw_sources = _extract_grounding_sources(raw_response)
    
    # Resolve URLs (follow redirects)
    resolved_sources = []
    for source in raw_sources:
        final_url = _resolve_url(source.url)
        final_domain = urlparse(final_url).netloc.replace("www.", "")
        resolved_sources.append(GroundingSource(
            url=final_url,
            domain=final_domain,
            title=source.title
        ))
    
    return CollectResult(
        prompt_id=prompt_id,
        model=model,
        text=answer_text,
        raw_json=json.dumps(raw_response),
        grounding_sources=resolved_sources,
    )