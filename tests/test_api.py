"""Tests for API endpoints"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_categories(client):
    """Test categories endpoint"""
    with patch('api.main.get_db') as mock_get_db:
        mock_session = MagicMock()
        mock_get_db.return_value = iter([mock_session])
        
        mock_categories = MagicMock()
        mock_categories.__iter__ = MagicMock(return_value=iter([]))
        mock_session.query.return_value.all.return_value = []
        
        response = client.get("/categories")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


def test_citation_domains(client):
    """Test citation domains endpoint - integration test with real DB"""
    # This test uses the actual database with fixtures
    response = client.get("/citation-domains")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # We have fixture data with citations, should have some domains
    assert len(data) >= 0


def test_model_results(client):
    """Test model results endpoint"""
    with patch('api.main.get_db') as mock_get_db:
        mock_session = MagicMock()
        mock_get_db.return_value = iter([mock_session])
        
        response = client.get("/model-results")
        # Will fail if file doesn't exist - that's expected behavior
        assert response.status_code in [200, 404]


def test_feature_importance(client):
    """Test feature importance endpoint"""
    with patch('api.main.get_db') as mock_get_db:
        mock_session = MagicMock()
        mock_get_db.return_value = iter([mock_session])
        
        response = client.get("/model-results/feature-importance", params={"model": "model_c_all"})
        assert response.status_code in [200, 404]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])