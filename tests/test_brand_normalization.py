"""Tests for brand normalization"""
import pytest
from aiv.categories import normalize_brand, get_brand_aliases


def test_normalize_brand_known():
    """Test normalization of known brand aliases"""
    assert normalize_brand("Otter") == "Otter.ai"
    assert normalize_brand("Otter AI") == "Otter.ai"
    assert normalize_brand("Fireflies") == "Fireflies.ai"
    assert normalize_brand("Fireflies AI") == "Fireflies.ai"
    assert normalize_brand("Fathom.video") == "Fathom"
    assert normalize_brand("tldv") == "tl;dv"
    assert normalize_brand("Gong") == "Gong.io"
    assert normalize_brand("Chorus") == "Chorus.ai"


def test_normalize_brand_unknown():
    """Test unknown brand returns None"""
    assert normalize_brand("UnknownBrand") is None
    assert normalize_brand("Random") is None


def test_get_brand_aliases():
    """Test brand aliases mapping"""
    aliases = get_brand_aliases()
    assert "otter" in aliases
    assert aliases["otter"] == "Otter.ai"
    assert "fireflies.ai" in aliases
    assert aliases["fireflies.ai"] == "Fireflies.ai"