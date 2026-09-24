# Project: AI Search Visibility & Citation Analytics
Goal: measure which B2B SaaS brands AI assistants mention and cite for
buyer-intent prompts, and model which web page features predict being cited.

Stack: Python 3.11, PostgreSQL (docker), SQLAlchemy, Pydantic settings,
google-genai (Gemini with Google Search grounding), Brave Search API, httpx,
trafilatura, Playwright fallback, sentence-transformers, scikit-learn,
FastAPI, React (Vite) + Recharts, pytest.

Rules:
- Cache every external API response to data/cache/ keyed by a hash of inputs;
  never re-call an API if the result is cached.
- Retry with exponential backoff on 429/5xx errors; rate limits come from config.
- All secrets come from .env; commit only .env.example.
- Log row counts at every pipeline stage.
- Tests must never call live APIs; use saved fixtures in tests/fixtures/.
- Small, typed functions with short comments on non-obvious logic.
- Before coding each phase, show me the plan and wait for my approval. Prompt 1 (setup):

Read AGENTS.md. Scaffold the project: src/aiv/ package, api/, web/, tests/,
data/cache/, data/labels/, docker-compose.yml with Postgres, .env.example
(GEMINI_API_KEY, BRAVE_API_KEY, DATABASE_URL), requirements.txt, and a CLI
(python -m aiv.cli <command>).
Create SQLAlchemy models for tables: prompts(id, category, text, intent),
responses(id, prompt_id, model, text, raw_json, created_at),
mentions(response_id, brand, rank, sentiment),
citations(response_id, url, domain, position),
search_results(prompt_id, url, domain, search_rank),
pages(url, status, title, text), page_features(url + one column per feature).
Create config/categories.yaml with 3 categories: AI meeting note-takers,
incident management tools, cold email outreach tools. For each: 10-15 real
brands with name aliases, and 50 varied buyer-intent prompts (best-for-X,
alternatives to Y, comparisons, pricing, use-case questions).
Add a CLI command to load prompts into the database.

Prompt 2 (collect LLM answers):

Build src/aiv/collect.py: for each prompt, call Gemini with Google Search
grounding enabled. Save answer text, raw JSON, and the grounding source URLs.
Grounding URLs are redirect links, so resolve each to its final URL and
extract the domain. Store in responses and citations tables. Use caching and
retries per AGENTS.md. Add a --limit flag. Run it with --limit 5 and show me
the saved results before running all prompts.

Prompt 3 (extract brands + accuracy check):

Build src/aiv/extract.py: for each response, use Gemini structured output with
a Pydantic schema to extract brands mentioned, their order of appearance
(rank), and sentiment (positive/neutral/negative). Normalize brand names using
the aliases in categories.yaml. Store in the mentions table.
Then export a random sample of 100 responses to data/labels/label_sample.csv
with an empty column for me to type the correct brands, and write a script
that computes brand-extraction precision and recall against my labels.

At this point, open the CSV and fill in the correct brands for each response yourself. Then run the accuracy script.

Prompt 4 (search results, scraping, features):

Build src/aiv/search.py: for each prompt, get the top 10 results from the
Brave Search API and store them in search_results.
Build src/aiv/scrape.py: fetch every unique URL from citations and
search_results using httpx + trafilatura, with a Playwright fallback for
JavaScript pages; timeouts, retries, caching, skip failures and log them.
Build src/aiv/features.py computing per page: word count, H2 count, H3 count,
list count, has table, has FAQ section, has schema.org JSON-LD and its types,
days since published/updated, domain type (reddit, review site, vendor,
blog, news, other), number of category brands mentioned, and cosine
similarity between prompt and page text using sentence-transformers
all-MiniLM-L6-v2.
Create a SQL view labeled_dataset with one row per (prompt_id, url):
all features, search_rank, and cited = 1 if Gemini cited that URL for that
prompt else 0.

Prompt 5 (model + metrics):

Build src/aiv/model.py: logistic regression with StandardScaler, evaluated
with GroupKFold grouped by prompt_id so no prompt appears in both train and
test. Compare 3 models: (a) search_rank only, (b) content features only,
(c) all features. Report mean ROC-AUC for each and save coefficients to
data/model_results.json.
Build src/aiv/metrics.py: share of voice per brand per category (% of
responses mentioning it), average rank per brand, and top 20 cited domains.
Create RESULTS.md with: prompts count, responses count, pages scraped,
extraction precision/recall, the 3 ROC-AUC scores, and top 5 findings.

Prompt 6 (API, dashboard, tests, README):

Build api/main.py with FastAPI endpoints: /categories, /share-of-voice
?category=, /citation-domains, /model-results.
Build a React (Vite) + Recharts dashboard in web/ with 4 charts: share of
voice by brand, average rank, top cited domains, feature importance.
Add Dockerfiles and update docker-compose.yml to run db, api and web together.
Write pytest tests for brand normalization, extraction parsing, feature
extraction and metrics using fixtures.
Write README.md: business question, method, results, limitations (correlation
not causation, one model, sample size), and how to run it.

After Prompt 6:

Push the project to GitHub, but not your .env file.