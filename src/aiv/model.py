"""Model training with logistic regression and GroupKFold"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .database import get_session
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# Feature columns for each model type
SEARCH_RANK_FEATURES = ["search_rank"]

CONTENT_FEATURES = [
    "word_count",
    "h2_count",
    "h3_count",
    "list_count",
    "has_table",
    "has_faq",
    "has_json_ld",
    "days_since_published",
    "days_since_updated",
    "category_brand_mentions",
    "prompt_similarity",
    "domain_type",  # Will be one-hot encoded
]

ALL_FEATURES = SEARCH_RANK_FEATURES + CONTENT_FEATURES


def load_labeled_data() -> pd.DataFrame:
    """Load labeled dataset from the SQL view"""
    with get_session() as session:
        from sqlalchemy import text
        result = session.execute(text("SELECT * FROM labeled_dataset"))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df


def preprocess_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess features for model training"""
    # Select features and target
    X = df[feature_cols].copy()
    y = df["cited"].values
    groups = df["prompt_id"].values
    
    # Columns that should be numeric (even if stored as object in DB)
    numeric_col_names = {
        "word_count", "h2_count", "h3_count", "list_count",
        "has_table", "has_faq", "has_json_ld",
        "days_since_published", "days_since_updated",
        "category_brand_mentions", "prompt_similarity",
        "search_rank",
    }
    
    # Handle missing values
    for col in X.columns:
        if col in numeric_col_names:
            # Convert to numeric, coercing errors to NaN
            X[col] = pd.to_numeric(X[col], errors="coerce")
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0
            X[col] = X[col].fillna(median_val)
        else:
            X[col] = X[col].fillna("unknown")
    
    return X, y, groups


def create_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    """Create preprocessing pipeline"""
    numeric_features = [c for c in feature_cols if c != "domain_type"]
    categorical_features = ["domain_type"] if "domain_type" in feature_cols else []
    
    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features))
    
    return ColumnTransformer(transformers, remainder="drop")


def train_model(X: np.ndarray, y: np.ndarray, groups: np.ndarray, feature_names: List[str]) -> Dict:
    """Train logistic regression with GroupKFold cross-validation"""
    cv = GroupKFold(n_splits=5)
    
    # Create pipeline
    preprocessor = create_preprocessor(feature_names)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ])
    
    fold_scores = []
    fold_coefs = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Compute ROC-AUC
        auc = roc_auc_score(y_val, y_pred_proba)
        fold_scores.append(auc)
        logger.info(f"Fold {fold_idx + 1}: ROC-AUC = {auc:.4f}")
        
        # Get coefficients (after preprocessing)
        # We need to get the feature names after one-hot encoding
        if fold_idx == 0:
            # Get transformed feature names
            preprocessor.fit(X_train)
            transformed_names = preprocessor.get_feature_names_out()
        
        # Get coefficients from the classifier
        coef = model.named_steps["classifier"].coef_[0]
        fold_coefs.append(coef)
    
    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    
    # Average coefficients across folds
    avg_coef = np.mean(fold_coefs, axis=0)
    
    # Create feature importance dict
    if len(transformed_names) == len(avg_coef):
        feature_importance = dict(zip(transformed_names, avg_coef))
    else:
        feature_importance = {f"feature_{i}": float(c) for i, c in enumerate(avg_coef)}
    
    return {
        "mean_roc_auc": float(mean_auc),
        "std_roc_auc": float(std_auc),
        "fold_scores": [float(s) for s in fold_scores],
        "feature_importance": feature_importance,
        "n_samples": len(X),
        "n_features": len(transformed_names) if 'transformed_names' in locals() else len(feature_cols),
    }


def train_all_models() -> Dict:
    """Train all three model variants"""
    df = load_labeled_data()
    
    if len(df) == 0:
        raise ValueError("No data in labeled_dataset view")
    
    logger.info(f"Loaded {len(df)} samples from labeled_dataset")
    
    results = {}
    
    # Model A: search_rank only - skip if all NaN
    logger.info("Training Model A (search_rank only)...")
    if df["search_rank"].notna().any():
        X_a, y_a, groups_a = preprocess_features(df, SEARCH_RANK_FEATURES)
        results["model_a_search_rank"] = train_model(X_a, y_a, groups_a, SEARCH_RANK_FEATURES)
    else:
        logger.warning("search_rank all NaN, skipping Model A")
        results["model_a_search_rank"] = {"mean_roc_auc": 0.5, "std_roc_auc": 0.0, "fold_scores": [], "feature_importance": {}, "n_samples": len(df), "n_features": 1}
    
    # Model B: content features only
    logger.info("Training Model B (content features only)...")
    X_b, y_b, groups_b = preprocess_features(df, CONTENT_FEATURES)
    results["model_b_content"] = train_model(X_b, y_b, groups_b, CONTENT_FEATURES)
    
    # Model C: all features
    logger.info("Training Model C (all features)...")
    X_c, y_c, groups_c = preprocess_features(df, ALL_FEATURES)
    results["model_c_all"] = train_model(X_c, y_c, groups_c, ALL_FEATURES)
    
    return results


def save_model_results(results: Dict, output_path: str = "data/model_results.json") -> None:
    """Save model results to JSON"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Model results saved to {output_path}")


def run_model_training() -> Dict:
    """Run complete model training pipeline"""
    results = train_all_models()
    save_model_results(results)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_model_training()