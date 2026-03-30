# backend/tests/test_phase9.py
# Phase 9: Tests for post-MVP enhancements.
# Covers: PDF export, model persistence, integrations adapter, retraining.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import tempfile
import shutil

from pipeline.ingestion import load_and_validate_csv
from pipeline.feature_extraction import extract_features
from pipeline.modeling import train_and_explain
from pipeline.recommender import compute_campaign_score, compute_archetypes, FEATURE_LABELS

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name}")
        failed += 1


# ── Prepare: load data, train model ──────────────────────────────────────────
print("\n⏳ Loading data and training model for Phase 9 tests...")
csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_campaigns.csv")
df = pd.read_csv(csv_path)
df_clean, _ = load_and_validate_csv(df)
X = extract_features(df_clean)
y = df_clean["conversion_rate"]
model, scaler, shap_values, feature_importance, cv_metrics = train_and_explain(X, y)
print("  Model trained.\n")


# ── TEST 1: PDF Export ────────────────────────────────────────────────────────
print("🧪 TEST 1: PDF export generates valid bytes")

from utils.export import generate_analysis_pdf

# Build a mock analysis result matching what /api/analyze returns
predictions = model.predict(scaler.transform(X) if scaler else X.values)
campaign_scores = []
avg_rate = float(y.mean())
std_rate = float(y.std())

for i, (_, row) in enumerate(df_clean.iterrows()):
    score = compute_campaign_score(predictions[i], avg_rate, std_rate)
    campaign_scores.append({
        "campaign_id": f"C{i}",
        "headline": row.get("headline", ""),
        "traffic_source": row.get("traffic_source", ""),
        "actual_conversion": round(float(row.get("conversion_rate", 0)), 2),
        "predicted_conversion": round(float(predictions[i]), 2),
        "grade": score["grade"],
        "grade_label": score["label"],
    })

archetypes = compute_archetypes(X, df_clean)

top_features = [
    {"feature": feat, "label": FEATURE_LABELS.get(feat, feat),
     "importance": round(float(imp), 4), "importance_pct": round(float(imp) * 100, 1)}
    for feat, imp in list(feature_importance.items())[:10]
]

source_stats = (
    df_clean.groupby("traffic_source")["conversion_rate"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "avg_conversion", "std": "std_conversion", "count": "n_campaigns"})
)
source_breakdown = source_stats.fillna(0).round(2).to_dict("records")

mock_result = {
    "status": "success",
    "cv_metrics": cv_metrics,
    "feature_importance": top_features,
    "campaign_scores": campaign_scores,
    "archetypes": archetypes,
    "source_breakdown": source_breakdown,
    "summary": {
        "n_campaigns": len(df_clean),
        "avg_conversion_rate": round(avg_rate, 2),
        "best_campaign": df_clean.loc[y.idxmax(), "headline"],
        "worst_campaign": df_clean.loc[y.idxmin(), "headline"],
        "validation_errors": [],
    },
}

pdf_bytes = generate_analysis_pdf(mock_result)
check("PDF bytes are not empty", len(pdf_bytes) > 0)
check("PDF starts with %PDF header", pdf_bytes[:5] == b"%PDF-")
check("PDF is at least 1KB", len(pdf_bytes) > 1024)

# Verify it's writable to a file
with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
    f.write(pdf_bytes)
    temp_pdf = f.name
check("PDF file created successfully", os.path.exists(temp_pdf))
os.unlink(temp_pdf)

# Test with empty result (edge case)
empty_result = {"summary": {}, "cv_metrics": {}}
empty_pdf = generate_analysis_pdf(empty_result)
check("Empty result still generates valid PDF", empty_pdf[:5] == b"%PDF-")


# ── TEST 2: Model Persistence ────────────────────────────────────────────────
print("\n🧪 TEST 2: Model save/load roundtrip")

from utils.model_store import save_model, load_latest_model, list_saved_models

# Create a temp models directory to avoid polluting the real one
original_model_dir = None
import utils.model_store as ms
original_model_dir = ms.MODEL_DIR
temp_model_dir = tempfile.mkdtemp(prefix="test_models_")
ms.MODEL_DIR = temp_model_dir

model_state = {
    "model": model,
    "scaler": scaler,
    "X_columns": list(X.columns),
    "avg_rate": avg_rate,
    "std_rate": std_rate,
    "campaign_averages": X.mean().to_dict(),
    "feature_importance": feature_importance,
    "n_campaigns": len(df_clean),
}

# Save
save_path = save_model(model_state)
check("Save returns a path", os.path.isdir(save_path))
check("model.joblib exists", os.path.exists(os.path.join(save_path, "model.joblib")))
check("metadata.json exists", os.path.exists(os.path.join(save_path, "metadata.json")))

# Load
loaded = load_latest_model()
check("Load returns a dict", isinstance(loaded, dict))
check("Loaded model is not None", loaded["model"] is not None)
check("X_columns preserved", loaded["X_columns"] == list(X.columns))
check("avg_rate preserved", abs(loaded["avg_rate"] - avg_rate) < 0.001)
check("n_campaigns preserved", loaded["n_campaigns"] == len(df_clean))

# Verify loaded model produces same predictions
loaded_pred = loaded["model"].predict(
    loaded["scaler"].transform(X) if loaded["scaler"] else X.values
)
orig_pred = model.predict(scaler.transform(X) if scaler else X.values)
check("Loaded model predictions match original",
      np.allclose(loaded_pred, orig_pred, atol=0.01))

# List models
versions = list_saved_models()
check("List returns at least 1 version", len(versions) >= 1)
check("Version has saved_at", "saved_at" in versions[0])

# Cleanup
ms.MODEL_DIR = original_model_dir
shutil.rmtree(temp_model_dir, ignore_errors=True)


# ── TEST 3: Platform Integrations ─────────────────────────────────────────────
print("\n🧪 TEST 3: Platform adapter pattern")

from utils.integrations import get_adapter, PlatformAdapter, PLATFORM_ADAPTERS

check("ActionKit adapter exists", "actionkit" in PLATFORM_ADAPTERS)
check("NationBuilder adapter exists", "nationbuilder" in PLATFORM_ADAPTERS)

# ActionKit adapter
ak = get_adapter("actionkit")
check("ActionKit is a PlatformAdapter", isinstance(ak, PlatformAdapter))
check("ActionKit name correct", ak.platform_name() == "ActionKit")
ak.authenticate("test-key", base_url="https://act.example.org")
campaigns_df = ak.fetch_campaigns()
check("ActionKit returns DataFrame", isinstance(campaigns_df, pd.DataFrame))
check("ActionKit has correct columns",
      all(c in campaigns_df.columns for c in ["headline", "body_text", "cta_text"]))

# NationBuilder adapter
nb = get_adapter("nationbuilder")
check("NationBuilder is a PlatformAdapter", isinstance(nb, PlatformAdapter))
check("NationBuilder name correct", nb.platform_name() == "NationBuilder")

# Invalid platform
try:
    get_adapter("invalid_platform")
    check("Invalid platform raises error", False)
except ValueError:
    check("Invalid platform raises ValueError", True)


# ── TEST 5: API Endpoints (retrain & export-pdf) ─────────────────────────────
print("\n🧪 TEST 5: API Endpoints (retrain & export-pdf)")

from fastapi.testclient import TestClient
import main
from main import app

orig_state = dict(main._model_state)
orig_result = dict(main._last_analysis_result)

client = TestClient(app)

# Test PDF export success
main._last_analysis_result = mock_result
r_pdf = client.get("/api/export-pdf")
check("GET /api/export-pdf returns 200", r_pdf.status_code == 200)
check("PDF response has correct content type", r_pdf.headers.get("content-type") == "application/pdf")
check("PDF response contains bytes", len(r_pdf.content) > 1024)

# Test PDF export without analysis
main._last_analysis_result.clear()
r_pdf_fail = client.get("/api/export-pdf")
check("GET /api/export-pdf without analysis returns 400", r_pdf_fail.status_code == 400)

# Back up sample_campaigns.csv to avoid modifying it permanently
real_csv = os.path.join(os.path.dirname(__file__), "..", "data", "sample_campaigns.csv")
backup_csv = real_csv + ".bak"
if os.path.exists(real_csv):
    shutil.copy2(real_csv, backup_csv)

# Test retrain success
retrain_csv = b"headline,body_text,cta_text,unique_visitors,signatures,traffic_source\n" \
              b"Another test,This is the body text which needs to have enough words to process properly and not fail due to word count limits on empty text,Click here,1000,50,email\n"

try:
    r_retrain = client.post("/api/retrain", files={"file": ("new.csv", retrain_csv, "text/csv")})
    check("POST /api/retrain returns 200", r_retrain.status_code == 200)
    retrain_data = r_retrain.json()
    check("Retrain status is success", retrain_data.get("status") == "success")
    check("Retrain payload has cv_metrics", "cv_metrics" in retrain_data)
    check("Retrain payload has analysis", "analysis" in retrain_data)
    check("Retrain analysis has summary", "summary" in retrain_data.get("analysis", {}))

    # Test retrain failure
    r_retrain_fail = client.post("/api/retrain", files={"file": ("bad.csv", b"", "text/csv")})
    check("POST /api/retrain with bad data returns 400", r_retrain_fail.status_code == 400)
finally:
    # Restore sample_campaigns.csv
    if os.path.exists(backup_csv):
        shutil.move(backup_csv, real_csv)

main._model_state.clear()
main._model_state.update(orig_state)
main._last_analysis_result.clear()
main._last_analysis_result.update(orig_result)
print("\n🧪 TEST 4: Unauthenticated adapter raises error")

fresh_ak = get_adapter("actionkit")
try:
    fresh_ak.fetch_campaigns()
    check("Unauthenticated fetch raises error", False)
except ConnectionError:
    check("Unauthenticated fetch raises ConnectionError", True)


# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} checks")
if failed == 0:
    print("🎉 All Phase 9 tests passed!")
else:
    print("⚠️  Some tests failed.")
    sys.exit(1)
