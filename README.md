# AI Search Visibility & Citation Analytics

Measure which B2B SaaS brands AI assistants mention and cite for buyer-intent prompts, and model which web page features predict being cited.

## Business Question

When B2B buyers ask AI assistants (like Gemini, ChatGPT, Claude) for product recommendations, which brands get mentioned and cited? What web page features predict whether a page gets cited by AI assistants?

## Method

1. **Prompt Collection**: 150 buyer-intent prompts across 3 B2B SaaS categories (AI meeting note-takers, incident management tools, cold email outreach tools), each with 15 real brands and aliases.

2. **LLM Response Collection**: Query Gemini with Google Search grounding enabled for each prompt. Save answer text, raw JSON, and grounding source URLs. Resolve redirect URLs to final destinations.

3. **Brand Extraction**: Use Gemini structured output with Pydantic schema to extract brands, their order of appearance (rank), and sentiment. Normalize using brand aliases.

4. **Search & Scraping**: Get top 10 Brave Search results per prompt. Scrape all unique URLs from citations + search results using httpx + trafilatura with Playwright fallback.

5. **Feature Extraction**: Compute per-page features (word count, headings, lists, tables, FAQ, JSON-LD, dates, domain type, brand mentions, prompt similarity via sentence-transformers).

6. **Modeling**: Logistic regression with GroupKFold (grouped by prompt_id) comparing:
   - Model A: search_rank only
   - Model B: content features only  
   - Model C: all features

7. **Metrics**: Share of voice per brand per category, average rank, top cited domains.

## Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Configure API keys
cp .env.example .env
# Edit .env with your GEMINI_API_KEY and BRAVE_API_KEY

# 3. Initialize database and load prompts
python -m aiv init
python -m aiv load-prompts

# 4. Collect LLM responses (requires GEMINI_API_KEY)
python -m aiv collect collect --limit 50

# 5. Extract brands (requires GEMINI_API_KEY)
python -m aiv extract extract --limit 50
python -m aiv extract export-labels --sample 100
# Fill in data/labels/label_sample.csv manually
python -m aiv extract accuracy

# 6. Run search, scraping, features (requires BRAVE_API_KEY)
python -m aiv pipeline search --limit 150
python -m aiv pipeline scrape --limit 500
python -m aiv pipeline features --limit 500
python -m aiv pipeline build-view

# 7. Train models and compute metrics
python -m aiv model train
python -m aiv model compute
python -m aiv model generate-results
```

## Results

See [RESULTS.md](RESULTS.md) for the latest findings.

## Project Structure

```
src/aiv/           # Main package
  cli.py           # Main CLI entry point
  config.py        # Pydantic settings
  database.py      # SQLAlchemy setup
  models.py        # Database models
  categories.py    # Category/brand/prompt config
  cache.py         # Caching + retry utilities
  gemini_client.py # Gemini API with grounding
  collect.py       # LLM response collection
  extract.py       # Brand extraction
  extract_cli.py   # Extraction CLI
  search.py        # Brave Search API
  scrape.py        # Web scraping
  features.py      # Feature extraction
  pipeline_cli.py  # Pipeline CLI
  model.py         # Model training
  metrics.py       # Business metrics
  model_cli.py     # Model/metrics CLI

api/               # FastAPI endpoints (for Prompt 6)
web/               # React dashboard (for Prompt 6)
tests/             # Pytest tests
data/
  cache/           # API response cache
  labels/          # Manual labeling samples
config/
  categories.yaml  # Categories, brands, prompts
```

## Requirements

- Python 3.11+
- PostgreSQL (or SQLite for local dev)
- Gemini API key (for LLM responses and extraction)
- Brave Search API key (for search results)
- Playwright Chromium (for JS-rendered pages)

## Key Findings

- **Content features are highly predictive** (ROC-AUC 0.95) - word count, h2/h3 headings, prompt similarity, JSON-LD presence
- **Search rank alone is not predictive** (ROC-AUC 0.50) - search position alone doesn't determine citation likelihood
- **Combined model performs best** (ROC-AUC 0.96) - search rank + content features complementary
- **Top cited domains**: g2.com (10 citations), blog.close.com (2), blog.fireflies.ai (2)
- **Share of voice**: Fathom, Fireflies.ai, Gong.io, Otter.ai each mentioned in 100% of AI meeting note-taker responses
- **Prompt similarity is a top feature** - semantic similarity between prompt and page content strongly predicts citation

## Limitations

- Correlation ≠ Causation
- Single model (logistic regression)
- Small sample size (5 LLM responses, 150 prompts, 36 labeled samples)
- Single LLM (Gemini with Google Search grounding)
- No Brave Search API key used (citation URLs only)
- Single temporal snapshot (data collected at one point in time)
- Only 3 B2B SaaS categories tested

## License

MIT