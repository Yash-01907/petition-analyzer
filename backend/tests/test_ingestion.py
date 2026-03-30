# backend/tests/test_ingestion.py
# Phase 2: Validates the ingestion pipeline against all exit criteria.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from pipeline.ingestion import load_and_validate_csv, REQUIRED_COLUMNS

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


# ── TEST 1: Valid CSV (sample_campaigns.csv) transforms deterministically ─────
print("\n🧪 TEST 1: Valid CSV transforms deterministically")
df_raw = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "sample_campaigns.csv"))
df, errors = load_and_validate_csv(df_raw.copy())

check("Returns a DataFrame", isinstance(df, pd.DataFrame))
check("Returns a list of errors", isinstance(errors, list))
check("Input row count preserved", len(df) == len(df_raw))
check("conversion_rate column exists", "conversion_rate" in df.columns)
check("All conversion rates >= 0", (df["conversion_rate"] >= 0).all())
check("All conversion rates <= 100", (df["conversion_rate"] <= 100).all())
check("No null campaign_ids", df["campaign_id"].notna().all())
check("No null headlines", (df["headline"] != "").all())
check("Traffic sources are valid", df["traffic_source"].isin({"email", "social", "organic", "paid"}).all())

# Run again — deterministic
df2, errors2 = load_and_validate_csv(df_raw.copy())
check("Deterministic: same row count", len(df) == len(df2))
check("Deterministic: same conversion rates", (df["conversion_rate"].values == df2["conversion_rate"].values).all())


# ── TEST 2: Missing required columns raise ValueError ─────────────────────────
print("\n🧪 TEST 2: Missing required columns raise ValueError")
try:
    bad_df = pd.DataFrame({"headline": ["test"], "body_text": ["test"]})
    load_and_validate_csv(bad_df)
    check("Should have raised ValueError", False)
except ValueError as e:
    check(f"ValueError raised: {e}", True)
    check("Error mentions missing columns",
          "cta_text" in str(e) or "Missing required" in str(e))


# ── TEST 3: Optional columns are filled with defaults ─────────────────────────
print("\n🧪 TEST 3: Optional columns filled with defaults")
minimal_df = pd.DataFrame({
    "headline": ["Stop the Dam", "Save the River"],
    "body_text": ["The dam will destroy our river.", "Our river needs protection now."],
    "cta_text": ["Sign Now", "Tell the Mayor"],
    "unique_visitors": [1000, 2000],
    "signatures": [100, 300],
    "traffic_source": ["email", "social"],
})
df_min, errs_min = load_and_validate_csv(minimal_df.copy())

check("cause_category filled", "cause_category" in df_min.columns)
check("campaign_duration_days filled", "campaign_duration_days" in df_min.columns)
check("has_image filled", "has_image" in df_min.columns)
check("has_video filled", "has_video" in df_min.columns)
check("campaign_id generated", df_min["campaign_id"].notna().all())
check("Warnings mention missing columns", len(errs_min) > 0)
check("conversion_rate computed correctly",
      abs(df_min.loc[0, "conversion_rate"] - 10.0) < 0.01)  # 100/1000*100 = 10%


# ── TEST 4: Invalid traffic sources are corrected ─────────────────────────────
print("\n🧪 TEST 4: Invalid traffic sources corrected")
bad_traffic = pd.DataFrame({
    "headline": ["Test headline"],
    "body_text": ["Test body text for the petition."],
    "cta_text": ["Sign it"],
    "unique_visitors": [1000],
    "signatures": [50],
    "traffic_source": ["INVALID_SOURCE"],
})
df_bt, errs_bt = load_and_validate_csv(bad_traffic.copy())

check("Invalid source defaulted to email", df_bt.loc[0, "traffic_source"] == "email")
check("Warning issued about invalid source",
      any("invalid traffic_source" in e for e in errs_bt))


# ── TEST 5: Non-numeric visitors/signatures are coerced ───────────────────────
print("\n🧪 TEST 5: Non-numeric values coerced safely")
messy_df = pd.DataFrame({
    "headline": ["Test"],
    "body_text": ["Test body for petition."],
    "cta_text": ["Sign"],
    "unique_visitors": ["not_a_number"],
    "signatures": ["also_bad"],
    "traffic_source": ["email"],
})
df_messy, _ = load_and_validate_csv(messy_df.copy())

check("Visitors coerced to default 1000", df_messy.loc[0, "unique_visitors"] == 1000)
check("Signatures coerced to default 0", df_messy.loc[0, "signatures"] == 0)
check("Conversion rate is 0% (0 sigs / 1000 visitors)",
      df_messy.loc[0, "conversion_rate"] == 0.0)


# ── TEST 6: Empty text rows are removed ───────────────────────────────────────
print("\n🧪 TEST 6: Empty text rows removed")
empty_text_df = pd.DataFrame({
    "headline": ["Good headline", "", "Another"],
    "body_text": ["Good body text here.", "Body", ""],
    "cta_text": ["Sign", "Sign", "Sign"],
    "unique_visitors": [1000, 1000, 1000],
    "signatures": [100, 50, 50],
    "traffic_source": ["email", "email", "email"],
})
df_et, errs_et = load_and_validate_csv(empty_text_df.copy())

check("Empty-text rows removed (1 valid row)",  len(df_et) == 1)
check("Warning about removed rows",
      any("empty" in e.lower() for e in errs_et))


# ── TEST 7: Zero visitors handled (no division by zero) ──────────────────────
print("\n🧪 TEST 7: Zero visitors handled")
zero_vis = pd.DataFrame({
    "headline": ["Test headline"],
    "body_text": ["Test body text content."],
    "cta_text": ["Sign now"],
    "unique_visitors": [0],
    "signatures": [10],
    "traffic_source": ["organic"],
})
df_zv, _ = load_and_validate_csv(zero_vis.copy())

check("Zero visitors coerced to 1000", df_zv.loc[0, "unique_visitors"] == 1000)
check("No NaN in conversion_rate", df_zv["conversion_rate"].notna().all())


# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} checks")
if failed == 0:
    print("🎉 All tests passed!")
else:
    print("⚠️  Some tests failed.")
    sys.exit(1)
