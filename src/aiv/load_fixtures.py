"""Load fixture responses into database for testing"""
import json
from pathlib import Path
from .database import get_session
from .models import Response, Citation, Prompt
from .gemini_client import GroundingSource


def load_fixture_responses():
    """Load test responses from fixtures"""
    fixture_dir = Path("tests/fixtures")
    
    # URL resolution mapping
    with open(fixture_dir / "resolved_urls.json") as f:
        url_map = json.load(f)
    
    with get_session() as session:
        # Get first 5 prompts
        prompts = session.query(Prompt).order_by(Prompt.id).limit(5).all()
        
        for i, prompt in enumerate(prompts):
            fixture_file = fixture_dir / f"gemini_response_{i+1}.json"
            if not fixture_file.exists():
                fixture_file = fixture_dir / "gemini_response.json"
            
            with open(fixture_file) as f:
                raw_response = json.load(f)
            
            # Extract answer text
            candidates = raw_response.get("candidates", [])
            answer_text = ""
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    if "text" in part:
                        answer_text += part["text"]
            
            # Create response
            response = Response(
                prompt_id=prompt.id,
                model="gemini-test",
                text=answer_text,
                raw_json=json.dumps(raw_response),
            )
            session.add(response)
            session.flush()
            
            # Extract and resolve grounding sources
            grounding_chunks = candidates[0].get("groundingMetadata", {}).get("groundingChunks", []) if candidates else []
            
            for pos, chunk in enumerate(grounding_chunks):
                web = chunk.get("web", {})
                original_url = web.get("uri", "")
                final_url = url_map.get(original_url, original_url)
                from urllib.parse import urlparse
                domain = urlparse(final_url).netloc.replace("www.", "")
                
                citation = Citation(
                    response_id=response.id,
                    url=final_url,
                    domain=domain,
                    position=pos + 1,
                )
                session.add(citation)
        
        session.commit()
        print(f"Loaded {len(prompts)} fixture responses")


if __name__ == "__main__":
    load_fixture_responses()