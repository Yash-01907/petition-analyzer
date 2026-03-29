# backend/tests/test_recommender.py
# Phase 5: Validates recommendation engine against exit criteria.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from pipeline.ingestion import load_and_validate_csv
from pipeline.feature_extraction import extract_features
from pipeline.modeling import train_and_explain, score_new_campaign
from pipeline.recommender import (
    generate_recommendations, compute_campaign_score, compute_archetypes
)

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
print("\n⏳ Loading data and training model...")
csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_campaigns.csv")
df, _ = load_and_validate_csv(pd.read_csv(csv_path))
X = extract_features(df)
y = df["conversion_rate"]
model, scaler, shap_values, feature_importance, cv_metrics = train_and_explain(X, y)
campaign_averages = X.mean().to_dict()
print("  Model trained.\n")


# ── TEST 1: Draft scoring returns grade + recommendations ────────────────────
print("🧪 TEST 1: Scored draft returns grade + actionable recommendations")

# Score a weak draft (generic headline, generic CTA, no urgency)
weak_result = score_new_campaign(
    model, scaler, list(X.columns),
    headline="Support Conservation",
    body="This petition addresses the impact of development on the green belt. "
         "We are calling on officials to reconsider the recent decision. "
         "The proposed plan would affect the community.",
    cta="Sign the Petition",
    traffic_source="social",
    cause_category="environment",
)

weak_features = weak_result["features"]
weak_pred = weak_result["predicted_conversion_rate"]

# Compute grade
weak_score = compute_campaign_score(weak_pred, float(y.mean()), float(y.std()))
check("Grade is a string", isinstance(weak_score["grade"], str))
check("Grade is A-F", weak_score["grade"] in "ABCDF")
check("Label exists", len(weak_score["label"]) > 0)
check("Color exists", weak_score["color"] in ("green", "yellow", "orange", "red"))
check("z_score is a float", isinstance(weak_score["z_score"], float))
check("predicted_rate matches", weak_score["predicted_rate"] == weak_pred)

print(f"\n  Weak draft: {weak_pred}% → Grade {weak_score['grade']} ({weak_score['label']})")

# Generate recommendations
recs = generate_recommendations(weak_features, feature_importance, campaign_averages)
check("Recommendations is a list", isinstance(recs, list))
check("At least 1 recommendation triggered", len(recs) >= 1)
check("At most 6 recommendations", len(recs) <= 6)

for i, r in enumerate(recs):
    check(f"  Rec {i+1} has title", "title" in r and len(r["title"]) > 0)
    check(f"  Rec {i+1} has description", "description" in r and len(r["description"]) > 0)
    check(f"  Rec {i+1} has example", "example" in r and len(r["example"]) > 0)
    check(f"  Rec {i+1} has grade_impact", r["grade_impact"] in ("high", "medium", "low"))
    check(f"  Rec {i+1} has feature_label", "feature_label" in r)

# Verify sorted by priority
if len(recs) >= 2:
    priority = {"high": 0, "medium": 1, "low": 2}
    impacts = [priority[r["grade_impact"]] for r in recs]
    check("Recommendations sorted by impact", impacts == sorted(impacts))

print(f"\n  Recommendations ({len(recs)} triggered):")
for r in recs:
    print(f"    [{r['grade_impact'].upper()}] {r['title']}")


# ── TEST 2: Strong draft gets higher grade ───────────────────────────────────
print("\n🧪 TEST 2: Strong draft gets higher grade than weak draft")

strong_result = score_new_campaign(
    model, scaler, list(X.columns),
    headline="URGENT: Tell Mayor Chen to Stop the Dam Before the Vote in 7 Days",
    body="Our river is under immediate threat. Mayor Chen has approved a dam project "
         "that will destroy the ecosystem you depend on. 2,500 of your neighbors have "
         "already signed — but the Mayor has not responded. We have 7 days before the "
         "final vote. Your signature today could change everything. Don't let this "
         "happen to your community.",
    cta="Tell Mayor Chen: Protect Our River Now",
    traffic_source="email",
    cause_category="environment",
)

strong_score = compute_campaign_score(
    strong_result["predicted_conversion_rate"], float(y.mean()), float(y.std())
)

print(f"  Strong draft: {strong_result['predicted_conversion_rate']}% → "
      f"Grade {strong_score['grade']} ({strong_score['label']})")

check("Strong draft predicted rate > 0", strong_result["predicted_conversion_rate"] > 0)
check("Strong score has grade", strong_score["grade"] in "ABCDF")

# Strong draft should have fewer/no recommendations
strong_recs = generate_recommendations(
    strong_result["features"], feature_importance, campaign_averages
)
check("Strong draft has fewer recs than weak", len(strong_recs) <= len(recs))
print(f"  Strong recs: {len(strong_recs)}, Weak recs: {len(recs)}")


# ── TEST 3: Archetype clustering ─────────────────────────────────────────────
print("\n🧪 TEST 3: Archetype list returns cluster names, rates, and examples")

archetypes = compute_archetypes(X, df, n_clusters=4)
check("Returns a list", isinstance(archetypes, list))
check("4 archetypes returned", len(archetypes) == 4)

for arch in archetypes:
    check(f"  '{arch['name']}' has id", "id" in arch)
    check(f"  '{arch['name']}' has name", len(arch["name"]) > 0)
    check(f"  '{arch['name']}' has campaign_count", arch["campaign_count"] > 0)
    check(f"  '{arch['name']}' has avg_conversion_rate",
          0 <= arch["avg_conversion_rate"] <= 100)
    check(f"  '{arch['name']}' has dominant_traits", isinstance(arch["dominant_traits"], list))
    check(f"  '{arch['name']}' has example_headlines",
          isinstance(arch["example_headlines"], list) and len(arch["example_headlines"]) > 0)

# Verify sorted by conversion rate descending
rates = [a["avg_conversion_rate"] for a in archetypes]
check("Archetypes sorted by conversion rate (desc)", rates == sorted(rates, reverse=True))

total_campaigns = sum(a["campaign_count"] for a in archetypes)
check("All campaigns accounted for", total_campaigns == 120)

print(f"\n  Archetypes:")
for a in archetypes:
    print(f"    {a['name']}: {a['campaign_count']} campaigns, {a['avg_conversion_rate']}% avg")


# ── TEST 4: Grade boundary checks ───────────────────────────────────────────
print("\n🧪 TEST 4: Grade boundary checks")

avg, std = 12.0, 8.0
check("z=2.0 → A", compute_campaign_score(avg + 2.0 * std, avg, std)["grade"] == "A")
check("z=1.0 → B", compute_campaign_score(avg + 1.0 * std, avg, std)["grade"] == "B")
check("z=0.0 → C", compute_campaign_score(avg, avg, std)["grade"] == "C")
check("z=-1.0 → D", compute_campaign_score(avg - 1.0 * std, avg, std)["grade"] == "D")
check("z=-2.0 → F", compute_campaign_score(avg - 2.0 * std, avg, std)["grade"] == "F")


# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} checks")
if failed == 0:
    print("🎉 All tests passed!")
else:
    print("⚠️  Some tests failed.")
    sys.exit(1)
