"""FastAPI endpoints for AI Search Visibility & Citation Analytics"""
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import func, text

from aiv.database import get_db
from aiv.models import Prompt, Response, Mention, Citation, PageFeature
from aiv.categories import load_categories

app = FastAPI(title="AI Search Visibility API", version="0.1.0")


class CategoryResponse(BaseModel):
    name: str
    brand_count: int
    prompt_count: int
    intents: List[str]


class ShareOfVoiceItem(BaseModel):
    brand: str
    share_pct: float
    mention_count: int


class CitedDomainItem(BaseModel):
    domain: str
    count: int


class ModelResultItem(BaseModel):
    model: str
    features: str
    mean_roc_auc: float
    std_roc_auc: float
    n_samples: int
    n_features: int


class FeatureImportanceItem(BaseModel):
    feature: str
    coefficient: float


@app.get("/categories", response_model=List[CategoryResponse])
def get_categories():
    """List all categories with brand and prompt counts"""
    categories = load_categories()
    return [
        CategoryResponse(
            name=cat.name,
            brand_count=len(cat.brands),
            prompt_count=len(cat.prompts),
            intents=sorted(set(p.intent for p in cat.prompts))
        )
        for cat in categories.values()
    ]


@app.get("/share-of-voice", response_model=List[ShareOfVoiceItem])
def get_share_of_voice(category: str = Query(..., description="Category name")):
    """Share of voice per brand for a category (% of responses mentioning it)"""
    with next(get_db()) as session:
        # Total responses in category
        total_responses = session.query(func.count(Response.id)).join(
            Prompt, Response.prompt_id == Prompt.id
        ).filter(Prompt.category == category).scalar()
        
        if total_responses == 0:
            return []
        
        # Mentions per brand in category
        results = session.query(
            Mention.brand,
            func.count(func.distinct(Response.id)).label("mention_count")
        ).join(
            Response, Mention.response_id == Response.id
        ).join(
            Prompt, Response.prompt_id == Prompt.id
        ).filter(
            Prompt.category == category
        ).group_by(Mention.brand).all()
        
        return [
            ShareOfVoiceItem(
                brand=brand,
                share_pct=round(count / total_responses * 100, 1),
                mention_count=count
            )
            for brand, count in sorted(results, key=lambda x: x[1], reverse=True)
        ]


@app.get("/citation-domains", response_model=List[CitedDomainItem])
def get_citation_domains(limit: int = Query(20, ge=1, le=100)):
    """Top cited domains across all responses"""
    with next(get_db()) as session:
        results = session.query(
            Citation.domain,
            func.count(Citation.id).label("count")
        ).group_by(Citation.domain).order_by(
            func.count(Citation.id).desc()
        ).limit(limit).all()
        
        return [
            CitedDomainItem(domain=domain, count=count)
            for domain, count in results
        ]


@app.get("/model-results", response_model=List[ModelResultItem])
def get_model_results():
    """Model performance results"""
    with next(get_db()) as session:
        from pathlib import Path
        import json
        
        results_path = Path("data/model_results.json")
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Model results not found. Run training first.")
        
        with open(results_path) as f:
            data = json.load(f)
        
        return [
            ModelResultItem(
                model=model_name,
                features={
                    "model_a_search_rank": "search_rank only",
                    "model_b_content": "content features only",
                    "model_c_all": "all features"
                }.get(model_name, model_name),
                mean_roc_auc=result["mean_roc_auc"],
                std_roc_auc=result["std_roc_auc"],
                n_samples=result["n_samples"],
                n_features=result["n_features"]
            )
            for model_name, result in data.items()
        ]


@app.get("/model-results/feature-importance", response_model=List[FeatureImportanceItem])
def get_feature_importance(model: str = Query("model_c_all", description="Model name")):
    """Feature importance coefficients for a model"""
    with next(get_db()) as session:
        from pathlib import Path
        import json
        
        results_path = Path("data/model_results.json")
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Model results not found")
        
        with open(results_path) as f:
            data = json.load(f)
        
        if model not in data:
            raise HTTPException(status_code=404, detail=f"Model {model} not found")
        
        fi = data[model].get("feature_importance", {})
        sorted_fi = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return [
            FeatureImportanceItem(feature=feat, coefficient=round(coef, 4))
            for feat, coef in sorted_fi
        ]


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}