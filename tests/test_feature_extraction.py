"""Tests for feature extraction"""
import pytest
from aiv.features import PageFeatures, _classify_domain


def test_page_features_defaults():
    """Test PageFeatures default values"""
    features = PageFeatures(url="https://example.com")
    assert features.url == "https://example.com"
    assert features.word_count == 0
    assert features.h2_count == 0
    assert features.has_faq == 0
    assert features.domain_type == "other"


def test_classify_domain_review_site():
    """Test domain classification for review sites"""
    assert _classify_domain("https://g2.com/product") == "review_site"
    assert _classify_domain("https://capterra.com/product") == "review_site"
    assert _classify_domain("https://trustradius.com/product") == "review_site"


def test_classify_domain_reddit():
    """Test domain classification for Reddit"""
    assert _classify_domain("https://reddit.com/r/test") == "reddit"
    assert _classify_domain("https://www.reddit.com/r/test") == "reddit"
    assert _classify_domain("https://old.reddit.com/r/test") == "reddit"


def test_classify_domain_vendor():
    """Test domain classification for vendor sites"""
    assert _classify_domain("https://salesforce.com") == "vendor"
    assert _classify_domain("https://hubspot.com") == "vendor"


def test_classify_domain_news():
    """Test domain classification for news sites"""
    assert _classify_domain("https://techcrunch.com/article") == "news"
    assert _classify_domain("https://theverge.com/article") == "news"


def test_classify_domain_blog():
    """Test domain classification for blogs"""
    assert _classify_domain("https://example.blogspot.com") == "blog"
    assert _classify_domain("https://medium.com/@user") == "blog"
    assert _classify_domain("https://substack.com/@user") == "blog"


def test_classify_domain_other():
    """Test domain classification for other"""
    assert _classify_domain("https://example.com") == "other"
    assert _classify_domain("https://unknown.site") == "other"