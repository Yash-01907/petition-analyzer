# utils/model_store.py — Persistent model storage with joblib.
# Phase 9: Post-MVP Enhancement #3.
#
# Saves and loads trained models + metadata to/from disk so they
# survive server restarts. In production, swap with Redis or S3.

import os
import json
from datetime import datetime
import joblib


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def ensure_model_dir():
    """Create the models directory if it doesn't exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)


def save_model(model_state: dict) -> str:
    """Persist model state to disk.

    Saves:
      - model.joblib     — the trained estimator
      - scaler.joblib    — the fitted scaler (or None)
      - metadata.json    — feature columns, averages, importance, metrics

    Args:
        model_state: The in-memory _model_state dict from main.py.

    Returns:
        Path to the saved model directory.
    """
    ensure_model_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(MODEL_DIR, f"model_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Save sklearn/xgboost model
    joblib.dump(model_state["model"], os.path.join(save_dir, "model.joblib"))

    # Save scaler (may be None for tree-based models)
    if model_state.get("scaler") is not None:
        joblib.dump(model_state["scaler"], os.path.join(save_dir, "scaler.joblib"))

    # Save metadata as JSON
    metadata = {
        "X_columns": model_state.get("X_columns", []),
        "avg_rate": model_state.get("avg_rate", 0),
        "std_rate": model_state.get("std_rate", 0),
        "campaign_averages": model_state.get("campaign_averages", {}),
        "feature_importance": {
            k: float(v)
            for k, v in model_state.get("feature_importance", {}).items()
        },
        "saved_at": datetime.now().isoformat(),
        "n_campaigns": model_state.get("n_campaigns", 0),
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Write a "latest" symlink/pointer
    latest_path = os.path.join(MODEL_DIR, "latest.txt")
    with open(latest_path, "w") as f:
        f.write(save_dir)

    return save_dir


def load_latest_model() -> dict | None:
    """Load the most recently saved model from disk.

    Returns:
        Reconstructed model_state dict, or None if no saved model exists.
    """
    latest_path = os.path.join(MODEL_DIR, "latest.txt")
    if not os.path.exists(latest_path):
        return None

    with open(latest_path, "r") as f:
        save_dir = f.read().strip()

    if not os.path.isdir(save_dir):
        return None

    model_path = os.path.join(save_dir, "model.joblib")
    if not os.path.exists(model_path):
        return None

    model = joblib.load(model_path)

    scaler = None
    scaler_path = os.path.join(save_dir, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    with open(os.path.join(save_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    return {
        "model": model,
        "scaler": scaler,
        "X_columns": metadata.get("X_columns", []),
        "avg_rate": metadata.get("avg_rate", 0),
        "std_rate": metadata.get("std_rate", 0),
        "campaign_averages": metadata.get("campaign_averages", {}),
        "feature_importance": metadata.get("feature_importance", {}),
        "n_campaigns": metadata.get("n_campaigns", 0),
    }


def list_saved_models() -> list:
    """List all saved model versions with timestamps.

    Returns:
        List of dicts with version info, sorted newest first.
    """
    ensure_model_dir()
    versions = []
    for entry in os.listdir(MODEL_DIR):
        meta_path = os.path.join(MODEL_DIR, entry, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            versions.append({
                "version": entry,
                "saved_at": meta.get("saved_at", ""),
                "n_campaigns": meta.get("n_campaigns", 0),
                "avg_rate": meta.get("avg_rate", 0),
            })
    versions.sort(key=lambda v: v["saved_at"], reverse=True)
    return versions
