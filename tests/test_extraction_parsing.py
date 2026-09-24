"""Tests for extraction parsing"""
import pytest
import json
from aiv.extract import ExtractionResult, BrandMention, Sentiment


def test_brand_mention_validation():
    """Test BrandMention model validation"""
    mention = BrandMention(brand="Otter.ai", rank=1, sentiment=Sentiment.POSITIVE)
    assert mention.brand == "Otter.ai"
    assert mention.rank == 1
    assert mention.sentiment == Sentiment.POSITIVE


def test_brand_mention_rank_validation():
    """Test rank must be >= 1"""
    with pytest.raises(ValueError):
        BrandMention(brand="Test", rank=0, sentiment=Sentiment.NEUTRAL)


def test_extraction_result_parsing():
    """Test parsing extraction result from JSON"""
    data = {
        "mentions": [
            {"brand": "Otter.ai", "rank": 1, "sentiment": "positive"},
            {"brand": "Fireflies.ai", "rank": 2, "sentiment": "neutral"},
        ]
    }
    result = ExtractionResult(**data)
    assert len(result.mentions) == 2
    assert result.mentions[0].brand == "Otter.ai"
    assert result.mentions[1].sentiment == Sentiment.NEUTRAL


def test_sentiment_enum():
    """Test Sentiment enum values"""
    assert Sentiment.POSITIVE.value == "positive"
    assert Sentiment.NEUTRAL.value == "neutral"
    assert Sentiment.NEGATIVE.value == "negative"