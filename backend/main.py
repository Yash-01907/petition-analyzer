# backend/main.py
# Phase 6 + Phase 9: Full API with post-MVP enhancements.
# Includes: analyze, score-draft, PDF export, model persistence, retraining.

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import pandas as pd
import numpy as np
import io
import logging
import os

from pipeline.ingestion import load_and_validate_csv
from pipeline.feature_extraction import extract_features
from pipeline.modeling import train_and_explain, score_new_campaign
from pipeline.recommender import (
    generate_recommendations, compute_campaign_score, compute_archetypes,
    FEATURE_LABELS,
)
from api.schemas import DraftScoreRequest
from utils.export import generate_analysis_pdf
from utils.model_store import save_model, load_latest_model, list_saved_models

logger = logging.getLogger("petition-analyzer")

ACTIVE_DATASET_PATH = "data/user_campaigns.csv"
SAMPLE_DATASET_PATH = "data/sample_campaigns.csv"

app = FastAPI(title="Petition Effectiveness Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state — persists across requests within the same process.
# Phase 9: On startup, attempt to restore from disk.
_model_state = {}
_last_analysis_result = {}  # Cache for PDF export

# Try to restore model from disk on startup
_restored = load_latest_model()
if _restored:
    _model_state.update(_restored)
    logger.info("Restored saved model from disk (%d campaigns)", _restored.get("n_campaigns", 0))


def _load_active_dataset() -> pd.DataFrame:
    """Load the current active dataset for retraining.

    Priority:
    1) model state's dataset_path
    2) ACTIVE_DATASET_PATH
    3) SAMPLE_DATASET_PATH (legacy fallback)
    """
    preferred = _model_state.get("dataset_path") or ACTIVE_DATASET_PATH
    if os.path.exists(preferred):
        return pd.read_csv(preferred)
    if os.path.exists(ACTIVE_DATASET_PATH):
        return pd.read_csv(ACTIVE_DATASET_PATH)
    if os.path.exists(SAMPLE_DATASET_PATH):
        return pd.read_csv(SAMPLE_DATASET_PATH)
    return pd.DataFrame()


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Petition Analyzer API running"}


@app.get("/api/health")
def health():
    model_ready = bool(_model_state.get("model"))
    return {"status": "healthy", "model_trained": model_ready}


# ── Analyze ───────────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_campaigns(file: UploadFile = File(...)):
    """Accept a CSV of past campaigns, run the full analysis pipeline.

    Returns feature importance, model metrics, campaign scores,
    archetypes, and source breakdown.
    """
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Invalid CSV: {str(e)}")

    # Validate and clean
    try:
        df, validation_errors = load_and_validate_csv(df)
    except ValueError as e:
        raise HTTPException(400, str(e))
        
    if len(df) < 5:
        raise HTTPException(400, "Need at least 5 campaigns to analyze")

    # Persist the uploaded dataset as the active retraining base.
    try:
        df.to_csv(ACTIVE_DATASET_PATH, index=False)
        _model_state["dataset_path"] = ACTIVE_DATASET_PATH
    except Exception as e:
        logger.warning("Failed to persist active dataset: %s", e)

    # Feature extraction
    X = extract_features(df)
    y = df["conversion_rate"]

    # Model training + SHAP
    model, scaler, shap_values, feature_importance, cv_metrics = train_and_explain(X, y)

    # Store model state for subsequent scoring requests
    _model_state["model"] = model
    _model_state["scaler"] = scaler
    _model_state["X_columns"] = list(X.columns)
    _model_state["avg_rate"] = float(y.mean())
    _model_state["std_rate"] = float(y.std())
    _model_state["campaign_averages"] = X.mean().to_dict()
    _model_state["feature_importance"] = feature_importance
    _model_state["n_campaigns"] = len(df)

    # Top 15 features for display
    top_features = [
        {
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "importance": round(float(imp), 4),
            "importance_pct": 0.0,
        }
        for feat, imp in list(feature_importance.items())[:15]
    ]
    total_imp = sum(f["importance"] for f in top_features) or 1
    for f in top_features:
        f["importance_pct"] = round(f["importance"] / total_imp * 100, 1)

    # Campaign-level scores
    predictions = model.predict(
        scaler.transform(X) if scaler else X.values
    )
    campaign_scores = []
    for i, (_, row) in enumerate(df.iterrows()):
        score = compute_campaign_score(
            predictions[i], _model_state["avg_rate"], _model_state["std_rate"]
        )
        campaign_scores.append({
            "campaign_id": row.get("campaign_id", f"C{i}"),
            "headline": row.get("headline", ""),
            "traffic_source": row.get("traffic_source", ""),
            "actual_conversion": round(float(row.get("conversion_rate", 0)), 2),
            "predicted_conversion": round(float(predictions[i]), 2),
            "grade": score["grade"],
            "grade_label": score["label"],
        })

    # Archetypes
    archetypes = compute_archetypes(X, df)

    # Source breakdown
    source_stats = (
        df.groupby("traffic_source")["conversion_rate"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "avg_conversion",
            "std": "std_conversion",
            "count": "n_campaigns",
        })
    )
    source_stats = source_stats.fillna(0).round(2)
    source_breakdown = source_stats.to_dict("records")

    result = {
        "status": "success",
        "cv_metrics": cv_metrics,
        "feature_importance": top_features,
        "campaign_scores": campaign_scores,
        "archetypes": archetypes,
        "source_breakdown": source_breakdown,
        "summary": {
            "n_campaigns": len(df),
            "avg_conversion_rate": round(float(y.mean()), 2),
            "best_campaign": df.loc[y.idxmax(), "headline"],
            "worst_campaign": df.loc[y.idxmin(), "headline"],
            "validation_errors": validation_errors,
        },
    }

    # Phase 9: Persist model to disk + cache result for export
    try:
        save_path = save_model(_model_state)
        logger.info("Model saved to %s", save_path)
    except Exception as e:
        logger.warning("Failed to persist model: %s", e)

    global _last_analysis_result
    _last_analysis_result = result

    return result


# ── Score Draft ───────────────────────────────────────────────────────────────

@app.post("/api/score-draft")
async def score_draft(request: DraftScoreRequest):
    """Score a new campaign draft in real-time.

    Returns predicted conversion rate, letter grade, and top recommendations.
    Includes content quality warnings and prediction penalties for low-quality inputs.
    Requires a model to be trained first via /api/analyze.
    """
    if not _model_state.get("model"):
        raise HTTPException(400, "No model trained yet. Upload campaign data first.")

    # ── Content quality gates ─────────────────────────────────────────────
    quality_warnings = []
    quality_penalty = 1.0  # Multiplier: 1.0 = no penalty, 0.0 = zeroed out

    hl_words = len(request.headline.strip().split())
    body_words = len(request.body_text.strip().split())
    cta_words = len(request.cta_text.strip().split())

    # Headline checks
    if hl_words < 3:
        quality_warnings.append({
            "field": "headline",
            "issue": f"Too short ({hl_words} word{'s' if hl_words != 1 else ''}). Effective headlines are 5–15 words.",
            "severity": "high",
        })
        quality_penalty *= 0.3
    elif hl_words < 5:
        quality_warnings.append({
            "field": "headline",
            "issue": f"Short ({hl_words} words). Best-performing headlines use 5–15 words.",
            "severity": "medium",
        })
        quality_penalty *= 0.7

    # Body text checks
    if body_words < 10:
        quality_warnings.append({
            "field": "body_text",
            "issue": f"Far too short ({body_words} words). Petition bodies need at least 80 words to be persuasive.",
            "severity": "high",
        })
        quality_penalty *= 0.2
    elif body_words < 40:
        quality_warnings.append({
            "field": "body_text",
            "issue": f"Very short ({body_words} words). High-converting petitions average 120–250 words.",
            "severity": "medium",
        })
        quality_penalty *= 0.5

    # CTA checks
    if cta_words < 2:
        quality_warnings.append({
            "field": "cta_text",
            "issue": f"Too short ({cta_words} word{'s' if cta_words != 1 else ''}). CTAs should clearly state what action to take.",
            "severity": "high",
        })
        quality_penalty *= 0.5

    # ── Run model prediction ──────────────────────────────────────────────
    result = score_new_campaign(
        model=_model_state["model"],
        scaler=_model_state["scaler"],
        feature_columns=_model_state["X_columns"],
        headline=request.headline,
        body=request.body_text,
        cta=request.cta_text,
        traffic_source=request.traffic_source or "email",
        cause_category=request.cause_category or "environment",
    )

    # Apply quality penalty to the raw prediction
    raw_rate = result["predicted_conversion_rate"]
    penalized_rate = round(raw_rate * quality_penalty, 2)

    score = compute_campaign_score(
        penalized_rate,
        _model_state["avg_rate"],
        _model_state["std_rate"],
    )

    recommendations = generate_recommendations(
        feature_values=result["features"],
        feature_importance=_model_state.get("feature_importance", {}),
        campaign_averages=_model_state.get("campaign_averages", {}),
        headline=request.headline,
        body=request.body_text,
        cta=request.cta_text,
    )

    return {
        **score,
        "features": result["features"],
        "recommendations": recommendations,
        "quality_warnings": quality_warnings,
        "quality_penalty_applied": quality_penalty < 1.0,
    }


# ── Sample Data ───────────────────────────────────────────────────────────────

@app.get("/api/sample-data")
async def get_sample_data():
    """Return the pre-generated synthetic dataset for demo purposes."""
    try:
        df = pd.read_csv("data/sample_campaigns.csv")
        return {
            "data": df.head(10).to_dict("records"),
            "total_rows": len(df),
            "columns": list(df.columns),
        }
    except FileNotFoundError:
        return {"error": "Sample data not generated yet. Run synthetic_generator.py"}

@app.get("/api/sample-csv")
async def get_sample_csv():
    """Return the raw sample CSV file to be uploaded by the UI mock data button."""
    from fastapi.responses import FileResponse
    import os
    file_path = "data/sample_campaigns.csv"
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="text/csv",
            filename="sample_campaigns.csv",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    raise HTTPException(404, "Sample data not found. Please run synthetic_generator.py first.")


# ── Phase 9: PDF Export ───────────────────────────────────────────────────────

@app.get("/api/export-pdf")
async def export_pdf():
    """Generate and download a PDF report of the last analysis.

    Requires /api/analyze to have been called first.
    """
    if not _last_analysis_result:
        raise HTTPException(400, "No analysis results available. Run /api/analyze first.")

    try:
        pdf_bytes = generate_analysis_pdf(_last_analysis_result)
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {str(e)}")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=petition_analysis_report.pdf"},
    )


# ── Phase 9: Model Management ────────────────────────────────────────────────

@app.get("/api/models")
async def list_models():
    """List all saved model versions."""
    versions = list_saved_models()
    return {
        "models": versions,
        "current_loaded": bool(_model_state.get("model")),
    }


# ── Phase 9: Retraining (Append Data) ────────────────────────────────────────

@app.post("/api/retrain")
async def retrain_with_new_data(file: UploadFile = File(...)):
    """Append new campaign data and retrain the model.

    Accepts a CSV of new campaigns, merges with the active analyzed dataset,
    and retrains. This implements the POST-CAMPAIGN LOOP from the spec:
    new results feed back into the model, sharpening recommendations.
    """
    contents = await file.read()
    try:
        new_df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Invalid CSV: {str(e)}")

    # Load active baseline data (latest analyzed user dataset when available).
    existing_df = _load_active_dataset()

    # Merge datasets
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.drop_duplicates(
        subset=["headline", "body_text"], keep="last", inplace=True
    )

    # Validate and clean
    try:
        df, validation_errors = load_and_validate_csv(combined_df)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if len(df) < 5:
        raise HTTPException(400, "Combined dataset has fewer than 5 valid campaigns.")

    # Retrain
    X = extract_features(df)
    y = df["conversion_rate"]
    model, scaler, shap_values, feature_importance, cv_metrics = train_and_explain(X, y)

    # Save validated dataset only on success as the new active retraining base.
    df.to_csv(ACTIVE_DATASET_PATH, index=False)


    # Update model state
    _model_state["model"] = model
    _model_state["scaler"] = scaler
    _model_state["X_columns"] = list(X.columns)
    _model_state["avg_rate"] = float(y.mean())
    _model_state["std_rate"] = float(y.std())
    _model_state["campaign_averages"] = X.mean().to_dict()
    _model_state["feature_importance"] = feature_importance
    _model_state["n_campaigns"] = len(df)
    _model_state["dataset_path"] = ACTIVE_DATASET_PATH

    # Persist
    try:
        save_model(_model_state)
    except Exception as e:
        logger.warning("Failed to persist retrained model: %s", e)

    return {
        "status": "success",
        "message": f"Retrained on {len(df)} campaigns (added {len(new_df)} new).",
        "cv_metrics": cv_metrics,
        "total_campaigns": len(df),
        "validation_errors": validation_errors,
    }
