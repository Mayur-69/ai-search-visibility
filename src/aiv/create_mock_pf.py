"""Create mock page_features for search URLs"""
import random
from aiv.database import get_session
from aiv.models import PageFeature, Page, SearchResult


def create_mock_page_features():
    """Add mock page_features for search URLs"""
    with get_session() as session:
        # Get search URLs that don't have page_features
        search_urls = session.query(SearchResult.url).all()
        search_urls = set([u[0] for u in search_urls])
        
        existing_pf = session.query(PageFeature.url).all()
        existing_pf = set([u[0] for u in existing_pf])
        
        missing_urls = search_urls - existing_pf
        
        # Also need to add to pages table
        existing_pages = session.query(Page.url).all()
        existing_pages = set([u[0] for u in existing_pages])
        
        for url in missing_urls:
            if url not in existing_pages:
                # Add mock page
                page = Page(
                    url=url,
                    status=200,
                    title=f"Mock page for {url}",
                    text=f"This is mock content for {url}",
                    html=f"<html><body>Mock content for {url}</body></html>",
                )
                session.add(page)
        
        session.flush()
        
        # Now add page_features
        for url in missing_urls:
            if url not in existing_pf:
                pf = PageFeature(
                    url=url,
                    word_count=random.randint(100, 1000),
                    h2_count=random.randint(0, 10),
                    h3_count=random.randint(0, 20),
                    list_count=random.randint(0, 15),
                    has_table=random.randint(0, 1),
                    has_faq=random.randint(0, 1),
                    has_json_ld=random.randint(0, 1),
                    json_ld_types="WebPage" if random.random() > 0.5 else None,
                    days_since_published=random.randint(1, 365) if random.random() > 0.3 else None,
                    days_since_updated=random.randint(1, 365) if random.random() > 0.3 else None,
                    domain_type=random.choice(["blog", "other", "review_site", "vendor", "news"]),
                    category_brand_mentions=random.randint(0, 5),
                    prompt_similarity=random.randint(0, 10000),
                )
                session.add(pf)
        
        session.commit()
        print(f"Created mock page_features for {len(missing_urls)} URLs")


if __name__ == "__main__":
    create_mock_page_features()