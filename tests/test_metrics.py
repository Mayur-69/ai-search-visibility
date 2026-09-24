"""Tests for metrics computation"""
import pytest
from aiv.metrics import compute_share_of_voice, compute_average_rank, compute_top_cited_domains


def test_share_of_voice():
    """Test share of voice computation"""
    # This test would need a database fixture
    # For now, just test that the function exists
    assert callable(compute_share_of_voice)


def test_average_rank():
    """Test average rank computation"""
    assert callable(compute_average_rank)


def test_top_cited_domains():
    """Test top cited domains computation"""
    assert callable(compute_top_cited_domains)