# backend/pipeline/modeling.py
# Phase 4: Adaptive model selection, training, SHAP explainability, and draft scoring.
#
# Model selection strategy (based on dataset size):
#   < 50 samples  → Ridge Regression (L2 regularized, prevents overfitting)
#   50–200 samples → Random Forest Regressor (handles interactions, robust)
#   200+ samples   → XGBoost (best performance, full SHAP support)

import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb


def select_model(n_samples: int):
    """Select the best model based on dataset size.

    Args:
        n_samples: Number of training samples.

    Returns:
        Tuple of (model instance, model name string, needs_scaling bool).
    """
    if n_samples < 50:
        model = Ridge(alpha=1.0)
        model_name = "Ridge Regression"
        needs_scaling = True
    elif n_samples < 200:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
        )
        model_name = "Random Forest"
        needs_scaling = False
    else:
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
        model_name = "XGBoost"
        needs_scaling = False

    return model, model_name, needs_scaling


def train_and_explain(X: pd.DataFrame, y: pd.Series):
    """Train the adaptive model and compute SHAP feature importance.

    Args:
        X: Feature matrix (all numeric, from extract_features).
        y: Target variable (conversion_rate).

    Returns:
        model: Trained model instance.
        scaler: StandardScaler (or None if not needed).
        shap_values: Array of SHAP values per sample per feature.
        feature_importance: Dict of {feature_name: mean_abs_shap_value}, sorted desc.
        cv_metrics: Dict with model_name, n_samples, cv_r2, cv_mae stats.
    """
    n = len(X)
    model, model_name, needs_scaling = select_model(n)

    # Handle scaling for linear models
    if needs_scaling:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)
    else:
        scaler = None
        X_fit = X.values

    # Cross-validation for honest performance estimate
    n_splits = min(5, max(3, n // 10))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(model, X_fit, y, cv=kf, scoring="r2")
    cv_mae = cross_val_score(
        model, X_fit, y, cv=kf, scoring="neg_mean_absolute_error"
    )

    # Train final model on full data
    model.fit(X_fit, y)

    # SHAP explanation
    if needs_scaling:
        explainer = shap.LinearExplainer(model, X_fit)
        shap_values = explainer.shap_values(X_fit)
    else:
        # Tree-based models (Random Forest or XGBoost)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_fit)

    # Mean absolute SHAP value per feature = global importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = dict(zip(X.columns, mean_abs_shap))
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    cv_metrics = {
        "model_name": model_name,
        "n_samples": n,
        "cv_r2_mean": float(np.mean(cv_r2)),
        "cv_r2_std": float(np.std(cv_r2)),
        "cv_mae_mean": float(-np.mean(cv_mae)),
        "cv_mae_std": float(np.std(cv_mae)),
    }

    return model, scaler, shap_values, feature_importance, cv_metrics


def score_new_campaign(
    model,
    scaler,
    feature_columns: list,
    headline: str,
    body: str,
    cta: str,
    traffic_source: str = "email",
    cause_category: str = "environment",
) -> dict:
    """Score a new campaign draft and return predicted conversion rate.

    Constructs a single-row DataFrame that matches the training schema,
    extracts features using the same pipeline, then predicts.

    Args:
        model: Trained model.
        scaler: StandardScaler or None.
        feature_columns: List of feature column names from training.
        headline: Draft headline text.
        body: Draft body text.
        cta: Draft CTA text.
        traffic_source: Expected traffic source.
        cause_category: Cause category.

    Returns:
        Dict with predicted_conversion_rate (float) and features (dict).
    """
    from pipeline.feature_extraction import extract_features

    draft = pd.DataFrame([{
        "headline": headline,
        "body_text": body,
        "cta_text": cta,
        "unique_visitors": 1000,  # Placeholder — not used in prediction
        "signatures": 0,
        "traffic_source": traffic_source,
        "cause_category": cause_category,
        "campaign_duration_days": 14,
        "launch_date": "2024-06-01",
        "has_image": True,
        "has_video": False,
    }])

    X_draft = extract_features(draft)

    # Align columns with training data (handle any column order mismatch)
    for col in feature_columns:
        if col not in X_draft.columns:
            X_draft[col] = 0
    X_draft = X_draft[feature_columns]

    if scaler:
        X_draft_scaled = scaler.transform(X_draft)
        pred = model.predict(X_draft_scaled)[0]
    else:
        pred = model.predict(X_draft.values)[0]

    return {
        "predicted_conversion_rate": round(float(np.clip(pred, 0.5, 45.0)), 2),
        "features": X_draft.iloc[0].to_dict(),
    }
