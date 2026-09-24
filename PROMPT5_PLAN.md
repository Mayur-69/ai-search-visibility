"""Plan for Prompt 5: Model + Metrics

Goal: 
1. Train logistic regression models with GroupKFold (grouped by prompt_id)
2. Compare 3 models: (a) search_rank only, (b) content features only, (c) all features
3. Report mean ROC-AUC for each, save coefficients to data/model_results.json
4. Build metrics: share of voice per brand per category, average rank per brand, top 20 cited domains
5. Create RESULTS.md with summary

---

### 1. src/aiv/model.py - Model Training

**Data**: labeled_dataset view (prompt_id, url, features..., search_rank, cited)

**Features to use:**
- Model A (search_rank only): search_rank
- Model B (content features only): word_count, h2_count, h3_count, list_count, has_table, has_faq, has_json_ld, days_since_published, days_since_updated, domain_type (one-hot), category_brand_mentions, prompt_similarity
- Model C (all features): search_rank + all content features

**Preprocessing:**
- StandardScaler on numeric features
- One-hot encode domain_type
- Handle missing values (fill with 0 or median)
- GroupKFold with n_splits=5, grouped by prompt_id

**Models:**
- LogisticRegression with default params (C=1.0, max_iter=1000)
- Cross-validation with GroupKFold
- Compute ROC-AUC for each fold, report mean ± std

**Output:**
- data/model_results.json with:
  - Model name, mean ROC-AUC, std ROC-AUC
  - Coefficients for each feature (for Model C)
  - Feature importance ranking

---

### 2. src/aiv/metrics.py - Business Metrics

**Share of Voice (per category):**
- % of responses mentioning each brand
- Formula: count(responses mentioning brand) / total responses in category

**Average Rank (per category):**
- Mean rank of each brand across responses where mentioned
- Only for responses where brand appears

**Top 20 Cited Domains:**
- Count citations per domain across all responses
- Sort descending, take top 20

---

### 3. RESULTS.md

Content:
- Prompts count (150)
- Responses count (need to collect more for full run)
- Pages scraped
- Extraction precision/recall (from Prompt 3)
- 3 ROC-AUC scores
- Top 5 findings

---

### CLI Commands

- `model train` - Train all 3 models
- `model eval` - Evaluate and save results
- `metrics compute` - Compute all metrics
- `results generate` - Create RESULTS.md

---

### Dependencies
- scikit-learn (already installed)
- numpy, pandas (already installed)