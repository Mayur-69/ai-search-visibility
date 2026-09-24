"""Business metrics: share of voice, average rank, top cited domains"""
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from .database import get_session
from .models import Mention, Citation, Response, Prompt
from .categories import load_categories


def compute_share_of_voice() -> Dict:
    """Compute share of voice per brand per category"""
    with get_session() as session:
        # Get all mentions with response -> prompt -> category
        mentions = session.query(
            Mention.brand,
            Prompt.category,
            Response.id
        ).join(Response, Mention.response_id == Response.id
        ).join(Prompt, Response.prompt_id == Prompt.id
        ).all()
    
    # Count responses per category
    category_response_counts = {}
    for brand, category, resp_id in mentions:
        if category not in category_response_counts:
            category_response_counts[category] = set()
        category_response_counts[category].add(resp_id)
    
    # Count mentions per brand per category
    brand_category_counts = {}
    for brand, category, resp_id in mentions:
        key = (category, brand)
        if key not in brand_category_counts:
            brand_category_counts[key] = set()
        brand_category_counts[key].add(resp_id)
    
    # Compute share of voice
    sov = {}
    for category, resp_set in category_response_counts.items():
        total_responses = len(resp_set)
        sov[category] = {}
        for (cat, brand), mention_resps in brand_category_counts.items():
            if cat == category:
                sov[category][brand] = len(mention_resps) / total_responses * 100
    
    return sov


def compute_average_rank() -> Dict:
    """Compute average rank per brand per category"""
    with get_session() as session:
        mentions = session.query(
            Mention.brand,
            Mention.rank,
            Prompt.category
        ).join(Response, Mention.response_id == Response.id
        ).join(Prompt, Response.prompt_id == Prompt.id
        ).all()
    
    # Group by category and brand
    brand_ranks = {}
    for brand, rank, category in mentions:
        key = (category, brand)
        if key not in brand_ranks:
            brand_ranks[key] = []
        brand_ranks[key].append(rank)
    
    # Compute averages
    avg_rank = {}
    for (category, brand), ranks in brand_ranks.items():
        if category not in avg_rank:
            avg_rank[category] = {}
        avg_rank[category][brand] = {
            "avg_rank": sum(ranks) / len(ranks),
            "mentions": len(ranks),
            "min_rank": min(ranks),
            "max_rank": max(ranks),
        }
    
    return avg_rank


def compute_top_cited_domains(top_n: int = 20) -> List[Dict]:
    """Compute top N cited domains"""
    with get_session() as session:
        citations = session.query(Citation.domain).all()
    
    domain_counts = Counter([c.domain for c in citations])
    top_domains = domain_counts.most_common(top_n)
    
    return [{"domain": domain, "count": count} for domain, count in top_domains]


def compute_all_metrics() -> Dict:
    """Compute all metrics"""
    metrics = {
        "share_of_voice": compute_share_of_voice(),
        "average_rank": compute_average_rank(),
        "top_cited_domains": compute_top_cited_domains(20),
    }
    return metrics


def save_metrics(metrics: Dict, output_path: str = "data/metrics.json") -> None:
    """Save metrics to JSON"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def run_metrics() -> Dict:
    """Run all metrics computation"""
    metrics = compute_all_metrics()
    save_metrics(metrics)
    return metrics


if __name__ == "__main__":
    run_metrics()