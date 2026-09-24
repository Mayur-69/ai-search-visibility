"""Plan for Prompt 3: Extract Brands + Accuracy Check

Goal: For each response, use Gemini structured output with Pydantic schema to extract brands mentioned, their order of appearance (rank), and sentiment (positive/neutral/negative). Normalize brand names using aliases in categories.yaml. Store in mentions table. Export random sample of 100 responses to data/labels/label_sample.csv for manual labeling. Write script to compute precision/recall against labels.

Components:
1. src/aiv/extract.py - Main extraction module
2. Pydantic schema for structured output (BrandMention list)
3. Brand normalization using categories.yaml aliases
4. Store in mentions table
5. Export label_sample.csv (100 random responses)
6. Accuracy script: compute precision/recall vs manual labels

Flow:
- Load responses from database (that don't have mentions yet)
- For each response:
  - Call Gemini with structured output schema
  - Parse extracted brands with rank and sentiment
  - Normalize brand names using aliases
  - Save to mentions table
- Export 100 random responses to CSV with empty "correct_brands" column
- Accuracy script reads CSV, computes precision/recall

Key Implementation Details:
- Use google-genai with response_schema for structured output
- Schema: List[BrandMention] with brand, rank, sentiment fields
- Sentiment enum: positive, neutral, negative
- Normalize using get_brand_aliases() from categories.py
- Cache API calls
- Retry with backoff
- Log row counts