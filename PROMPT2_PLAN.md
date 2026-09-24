"""Plan for Prompt 2: Collect LLM Answers with Gemini + Google Search Grounding

Goal: For each prompt, call Gemini with Google Search grounding enabled. Save answer text, raw JSON, and grounding source URLs. Resolve redirect URLs to final URLs and extract domains. Store in responses and citations tables.

Components:
1. src/aiv/collect.py - Main collection module
2. Gemini client with search grounding
3. URL resolution (follow redirects)
4. Caching layer (reuse cache.py)
5. Retry logic with exponential backoff
6. CLI command: collect --limit N

Flow:
- Load prompts from database
- For each prompt (up to --limit):
  - Check cache for existing response
  - Call Gemini with search grounding
  - Extract answer text, raw JSON, grounding URLs
  - Resolve each grounding URL (follow redirects, get final URL + domain)
  - Save to responses table
  - Save citations to citations table
- Log row counts at each stage

Key Implementation Details:
- Use google-genai library with grounding enabled
- Grounding URLs are redirect links - need to resolve with httpx
- Cache keyed by (prompt_id, model) hash
- Retry on 429/5xx with exponential backoff
- Rate limiting per config (GEMINI_RPM)
- Log: prompts processed, responses saved, citations saved, cache hits/misses