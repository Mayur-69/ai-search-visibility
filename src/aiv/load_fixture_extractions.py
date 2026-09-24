"""Load fixture extraction results into database for testing"""
import json
from pathlib import Path
from .database import get_session
from .models import Response, Mention
from .extract import save_mentions, ExtractedMention


def load_fixture_extractions():
    """Load test extraction results from fixtures"""
    fixture_dir = Path("tests/fixtures")
    
    # Load the extraction fixture
    with open(fixture_dir / "extraction_response.json") as f:
        extraction_data = json.load(f)
    
    # Convert to ExtractedMention objects
    mentions = []
    for m in extraction_data["mentions"]:
        mentions.append(ExtractedMention(
            brand=m["brand"],
            original_brand=m["brand"],
            rank=m["rank"],
            sentiment=m["sentiment"],
        ))
    
    with get_session() as session:
        # Get first 5 responses
        responses = session.query(Response).order_by(Response.id).limit(5).all()
        
        for response in responses:
            # Check if already has mentions
            existing = session.query(Mention).filter(Mention.response_id == response.id).count()
            if existing > 0:
                print(f"Response {response.id} already has mentions, skipping")
                continue
            
            # Save mentions (adjust brands per response for variety)
            saved = save_mentions(response.id, mentions)
            print(f"Saved {saved} mentions for response {response.id}")
        
        print("Done loading fixture extractions")


if __name__ == "__main__":
    load_fixture_extractions()