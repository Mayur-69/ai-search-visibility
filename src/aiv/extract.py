"""Brand extraction using Gemini structured output"""
import json
import hashlib
from dataclasses import dataclass
from typing import Optional, List, Literal
from enum import Enum

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from .config import get_settings
from .cache import cached, RateLimiter, retry_with_backoff
from .categories import normalize_brand

settings = get_settings()


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class BrandMention(BaseModel):
    """A brand mention extracted from LLM response"""
    brand: str = Field(description="Brand name as mentioned in the response")
    rank: int = Field(description="Order of appearance (1 = first mentioned)", ge=1)
    sentiment: Sentiment = Field(description="Sentiment toward the brand")


class ExtractionResult(BaseModel):
    """Structured output for brand extraction"""
    mentions: List[BrandMention] = Field(default_factory=list)


@dataclass
class ExtractedMention:
    """Normalized brand mention ready for storage"""
    brand: str          # Canonical brand name
    original_brand: str # Original text from response
    rank: int
    sentiment: str


_rate_limiter = RateLimiter(settings.gemini_rpm)


def _get_extraction_schema() -> dict:
    """Get JSON schema for structured output"""
    return ExtractionResult.model_json_schema()


@retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=60.0, retry_on=(Exception,))
@cached
def _call_gemini_structured(prompt_text: str, response_text: str, model: str = "gemini-2.5-flash") -> dict:
    """Call Gemini with structured output for brand extraction."""
    _rate_limiter.wait()
    
    client = genai.Client(api_key=settings.gemini_api_key)
    
    system_prompt = """You are an expert at identifying B2B SaaS brand mentions in AI assistant responses.
Extract all brand names mentioned, their order of appearance (rank), and sentiment.

Rules:
1. Only extract brands that are explicitly mentioned in the response text
2. Rank starts at 1 for the first brand mentioned, increment for each new brand
3. If a brand is mentioned multiple times, use its first appearance rank
4. Sentiment: positive (praise/recommendation), neutral (factual mention), negative (criticism/complaint)
5. Include variations like "Otter", "Otter.ai", "Otter AI" as the same brand
6. Do NOT infer brands not explicitly in the text"""
    
    user_prompt = f"""Response text to analyze:
---
{response_text}
---

Extract all brand mentions with rank and sentiment."""
    
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=_get_extraction_schema(),
        temperature=0.0,
        max_output_tokens=2048,
    )
    
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=config,
    )
    
    return json.loads(response.text)


def extract_brands(response_id: int, response_text: str, model: str = "gemini-2.5-flash") -> List[ExtractedMention]:
    """Extract and normalize brand mentions from a response."""
    # Call Gemini with structured output
    raw_result = _call_gemini_structured("", response_text, model)
    
    # Parse structured output
    extraction = ExtractionResult(**raw_result)
    
    # Normalize brand names
    normalized = []
    for mention in extraction.mentions:
        canonical = normalize_brand(mention.brand)
        if canonical:
            normalized.append(ExtractedMention(
                brand=canonical,
                original_brand=mention.brand,
                rank=mention.rank,
                sentiment=mention.sentiment.value,
            ))
        else:
            # Brand not in our known list - skip or keep as-is?
            # For now, skip unknown brands
            pass
    
    return normalized


def save_mentions(response_id: int, mentions: List[ExtractedMention]) -> int:
    """Save extracted mentions to database. Returns count saved."""
    from .database import get_session
    from .models import Mention
    
    saved = 0
    with get_session() as session:
        # Check if already has mentions
        existing = session.query(Mention).filter(Mention.response_id == response_id).count()
        if existing > 0:
            return 0
        
        for mention in mentions:
            m = Mention(
                response_id=response_id,
                brand=mention.brand,
                rank=mention.rank,
                sentiment=mention.sentiment,
            )
            session.add(m)
            saved += 1
        
        session.commit()
    
    return saved