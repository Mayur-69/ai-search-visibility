"""Create mock search results for testing"""
from aiv.database import get_session
from aiv.models import SearchResult, Prompt


def create_mock_search_results():
    """Add mock search results for testing"""
    with get_session() as session:
        prompts = session.query(Prompt).order_by(Prompt.id).limit(5).all()
        
        mock_domains = [
            "example.com", "test.com", "demo.com", "sample.com", "mock.com",
            "fake.com", "placeholder.com", "dummy.com", "trial.com", "beta.com"
        ]
        
        for prompt in prompts:
            # Check if already has search results
            existing = session.query(SearchResult).filter(SearchResult.prompt_id == prompt.id).count()
            if existing > 0:
                continue
            
            for rank in range(1, 11):
                domain = mock_domains[rank - 1]
                sr = SearchResult(
                    prompt_id=prompt.id,
                    url=f"https://{domain}/page-{rank}",
                    domain=domain,
                    search_rank=rank,
                )
                session.add(sr)
        
        session.commit()
        print(f"Created mock search results for {len(prompts)} prompts")


if __name__ == "__main__":
    create_mock_search_results()