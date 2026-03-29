# backend/tests/test_feature_extraction.py
# Phase 3: Validates feature extraction against exit criteria.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from pipeline.ingestion import load_and_validate_csv
from pipeline.feature_extraction import extract_features

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


# ── TEST 1: Single campaign row → complete feature vector ─────────────────────
print("\n🧪 TEST 1: Single campaign row → complete feature vector")

single_df = pd.DataFrame([{
    "headline": "Tell Mayor Chen: Stop the Dam Before It's Too Late",
    "body_text": (
        "Our river is under immediate threat. Mayor Chen has approved a dam project "
        "that will destroy the ecosystem. 2,500 residents have signed a community letter "
        "but the Mayor has not responded. We have 14 days before the final decision. "
        "Your signature today could change everything."
    ),
    "cta_text": "Tell Mayor Chen: Protect Our River",
    "unique_visitors": 5000,
    "signatures": 750,
    "traffic_source": "email",
    "cause_category": "environment",
    "campaign_duration_days": 21,
    "launch_date": "2024-03-15",
    "has_image": True,
    "has_video": False,
}])

single_clean, _ = load_and_validate_csv(single_df.copy())
X_single = extract_features(single_clean)

check("Returns a DataFrame", isinstance(X_single, pd.DataFrame))
check("Exactly 1 row", len(X_single) == 1)
check("Has 30+ features", len(X_single.columns) >= 30)

# Check all feature groups are present
check("Group A: headline_word_count exists", "headline_word_count" in X_single.columns)
check("Group A: body_word_count exists", "body_word_count" in X_single.columns)
check("Group A: has_image exists", "has_image" in X_single.columns)
check("Group B: headline_is_question exists", "headline_is_question" in X_single.columns)
check("Group B: headline_is_imperative exists", "headline_is_imperative" in X_single.columns)
check("Group B: headline_has_named_entity exists", "headline_has_named_entity" in X_single.columns)
check("Group B: headline_has_deadline exists", "headline_has_deadline" in X_single.columns)
check("Group C: headline_sentiment exists", "headline_sentiment" in X_single.columns)
check("Group C: body_anger_score exists", "body_anger_score" in X_single.columns)
check("Group C: body_fear_score exists", "body_fear_score" in X_single.columns)
check("Group D: body_reading_grade exists", "body_reading_grade" in X_single.columns)
check("Group D: body_flesch_score exists", "body_flesch_score" in X_single.columns)
check("Group D: first_person_density exists", "first_person_density" in X_single.columns)
check("Group E: cta_is_specific exists", "cta_is_specific" in X_single.columns)
check("Group E: cta_has_named_official exists", "cta_has_named_official" in X_single.columns)
check("Group F: launch_month exists", "launch_month" in X_single.columns)
check("Group F: source_email exists", "source_email" in X_single.columns)

# Check all values are numeric (no NaN, no strings)
check("All values are numeric", X_single.dtypes.apply(lambda d: np.issubdtype(d, np.number)).all())
check("No NaN values", X_single.isna().sum().sum() == 0)

# Sanity check feature values
row = X_single.iloc[0]
check("headline_is_imperative = 1 (starts with Tell)", row["headline_is_imperative"] == 1)
check("headline_has_deadline = 1 (contains urgency)", row["headline_has_deadline"] == 1)
check("cta_is_specific = 1 (specific CTA with name)", row["cta_is_specific"] == 1)
check("has_image = 1", row["has_image"] == 1)
check("has_video = 0", row["has_video"] == 0)
check("launch_month = 3 (March)", row["launch_month"] == 3)

print(f"\n  Feature count: {len(X_single.columns)}")
print(f"  Feature names: {list(X_single.columns)}")


# ── TEST 2: Full dataset converts without runtime failure ─────────────────────
print("\n🧪 TEST 2: Full dataset (120 rows) converts without runtime failure")

csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_campaigns.csv")
df_full = pd.read_csv(csv_path)
df_clean, _ = load_and_validate_csv(df_full.copy())
X_full = extract_features(df_clean)

check("Returns a DataFrame", isinstance(X_full, pd.DataFrame))
check("120 rows preserved", len(X_full) == 120)
check("Same column count as single row", len(X_full.columns) == len(X_single.columns))
check("All values are numeric", X_full.dtypes.apply(lambda d: np.issubdtype(d, np.number)).all())
check("No NaN values in full dataset", X_full.isna().sum().sum() == 0)
check("No Inf values", np.isfinite(X_full.values).all())

# Check value ranges are plausible
check("headline_word_count > 0 for all", (X_full["headline_word_count"] > 0).all())
check("body_word_count > 0 for all", (X_full["body_word_count"] > 0).all())
check("Sentiment in [-1, 1]",
      (X_full["headline_sentiment"] >= -1).all() and (X_full["headline_sentiment"] <= 1).all())
check("Reading grade plausible (0-20)",
      (X_full["body_reading_grade"] >= 0).all() and (X_full["body_reading_grade"] <= 20).all())

print(f"\n  Feature matrix shape: {X_full.shape}")
print(f"  Memory usage: {X_full.memory_usage(deep=True).sum() / 1024:.1f} KB")


# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} checks")
if failed == 0:
    print("🎉 All tests passed!")
else:
    print("⚠️  Some tests failed.")
    sys.exit(1)
