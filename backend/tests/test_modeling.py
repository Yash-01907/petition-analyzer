# backend/tests/test_modeling.py
# Phase 4: Validates modeling layer against exit criteria.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from pipeline.ingestion import load_and_validate_csv
from pipeline.feature_extraction import extract_features
from pipeline.modeling import select_model, train_and_explain, score_new_campaign

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


# ── Load and prepare full dataset ────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_campaigns.csv")
df_raw = pd.read_csv(csv_path)
df, _ = load_and_validate_csv(df_raw.copy())
X = extract_features(df)
y = df["conversion_rate"]


# ── TEST 1: Adaptive model selection ─────────────────────────────────────────
print("\n🧪 TEST 1: Adaptive model selection")

m1, n1, s1 = select_model(30)
check("< 50 samples → Ridge", n1 == "Ridge Regression")
check("Ridge needs scaling", s1 is True)

m2, n2, s2 = select_model(100)
check("50-200 samples → Random Forest", n2 == "Random Forest")
check("RF does not need scaling", s2 is False)

m3, n3, s3 = select_model(500)
check("200+ samples → XGBoost", n3 == "XGBoost")
check("XGBoost does not need scaling", s3 is False)


# ── TEST 2: Train/analyze flow returns predictions + explainability ──────────
print("\n🧪 TEST 2: Train/analyze returns predictions + explainability")

model, scaler, shap_values, feature_importance, cv_metrics = train_and_explain(X, y)

check("Model is not None", model is not None)
check("SHAP values returned", shap_values is not None)
check("SHAP shape matches data", shap_values.shape == X.shape)
check("Feature importance is a dict", isinstance(feature_importance, dict))
check("Feature importance has entries", len(feature_importance) > 0)
check("Feature importance sorted descending",
      list(feature_importance.values()) == sorted(feature_importance.values(), reverse=True))

# CV metrics
check("cv_metrics has model_name", "model_name" in cv_metrics)
check("cv_metrics has n_samples", cv_metrics["n_samples"] == 120)
check("cv_metrics has cv_r2_mean", "cv_r2_mean" in cv_metrics)
check("cv_metrics has cv_mae_mean", "cv_mae_mean" in cv_metrics)
check("R² is reasonable (> -1)", cv_metrics["cv_r2_mean"] > -1.0)
check("MAE is positive", cv_metrics["cv_mae_mean"] > 0)

print(f"\n  Model: {cv_metrics['model_name']}")
print(f"  CV R²: {cv_metrics['cv_r2_mean']:.3f} ± {cv_metrics['cv_r2_std']:.3f}")
print(f"  CV MAE: {cv_metrics['cv_mae_mean']:.2f} ± {cv_metrics['cv_mae_std']:.2f}")
print(f"  Top 5 features: {list(feature_importance.keys())[:5]}")

# Predictions from trained model
if scaler:
    preds = model.predict(scaler.transform(X))
else:
    preds = model.predict(X.values)

check("Predictions have correct length", len(preds) == 120)
check("Predictions are finite", np.isfinite(preds).all())
check("Predictions are plausible (0-50 range)",
      (preds > -5).all() and (preds < 50).all())


# ── TEST 3: Draft scoring returns bounded conversion rate ────────────────────
print("\n🧪 TEST 3: Draft scoring returns bounded predicted conversion rate")

result = score_new_campaign(
    model=model,
    scaler=scaler,
    feature_columns=list(X.columns),
    headline="Tell Mayor Chen: Stop the Dam Before It's Too Late",
    body="Our river is under immediate threat. Mayor Chen has approved a dam project "
         "that will destroy the ecosystem. We have 14 days before the final decision.",
    cta="Tell Mayor Chen: Protect Our River",
    traffic_source="email",
    cause_category="environment",
)

check("Result is a dict", isinstance(result, dict))
check("Has predicted_conversion_rate", "predicted_conversion_rate" in result)
check("Has features dict", "features" in result)
check("Prediction is a float", isinstance(result["predicted_conversion_rate"], float))
check("Prediction >= 0.5", result["predicted_conversion_rate"] >= 0.5)
check("Prediction <= 45.0", result["predicted_conversion_rate"] <= 45.0)
check("Features dict is non-empty", len(result["features"]) > 0)

print(f"\n  Predicted conversion rate: {result['predicted_conversion_rate']}%")


# ── TEST 4: Draft scoring with different traffic sources ─────────────────────
print("\n🧪 TEST 4: Different traffic sources produce different predictions")

scores_by_source = {}
for src in ["email", "social", "organic", "paid"]:
    r = score_new_campaign(
        model=model, scaler=scaler, feature_columns=list(X.columns),
        headline="Save Our Park", body="The park is threatened by development. "
                 "We need your help to stop this plan before it is too late.",
        cta="Sign Now", traffic_source=src, cause_category="environment",
    )
    scores_by_source[src] = r["predicted_conversion_rate"]
    check(f"  {src}: bounded [{r['predicted_conversion_rate']}%]",
          0.5 <= r["predicted_conversion_rate"] <= 45.0)

print(f"\n  Scores by source: {scores_by_source}")


# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} checks")
if failed == 0:
    print("🎉 All tests passed!")
else:
    print("⚠️  Some tests failed.")
    sys.exit(1)
