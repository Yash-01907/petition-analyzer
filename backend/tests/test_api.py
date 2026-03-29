# backend/tests/test_api.py
# Phase 6: End-to-end API tests using FastAPI TestClient.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from main import app

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


client = TestClient(app)

# ── TEST 1: Health and root endpoints ────────────────────────────────────────
print("\n🧪 TEST 1: Health and root endpoints")

r = client.get("/")
check("GET / returns 200", r.status_code == 200)
check("Root status ok", r.json()["status"] == "ok")

r = client.get("/api/health")
check("GET /api/health returns 200", r.status_code == 200)
check("Health model_trained is false initially", r.json()["model_trained"] == False)


# ── TEST 2: Sample data endpoint ─────────────────────────────────────────────
print("\n🧪 TEST 2: Sample data endpoint")

r = client.get("/api/sample-data")
check("GET /api/sample-data returns 200", r.status_code == 200)
body = r.json()
check("Has data key", "data" in body)
check("Has total_rows", body.get("total_rows", 0) > 0)
check("Has columns", len(body.get("columns", [])) > 0)
check("Data has 10 preview rows", len(body.get("data", [])) == 10)


# ── TEST 3: Score-draft before analyze should fail ───────────────────────────
print("\n🧪 TEST 3: Score-draft before analyze should fail")

r = client.post("/api/score-draft", json={
    "headline": "Test Headline for Scoring",
    "body_text": "This is a draft body text that needs to be long enough to pass validation.",
    "cta_text": "Sign Now",
})
check("Returns 400 without model", r.status_code == 400)
check("Error mentions no model", "No model" in r.json()["detail"])


# ── TEST 4: Analyze endpoint with CSV upload ─────────────────────────────────
print("\n🧪 TEST 4: Analyze endpoint — full pipeline")

csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_campaigns.csv")
with open(csv_path, "rb") as f:
    r = client.post("/api/analyze", files={"file": ("campaigns.csv", f, "text/csv")})

check("POST /api/analyze returns 200", r.status_code == 200)
body = r.json()

check("Status is success", body.get("status") == "success")

# CV metrics
cv = body.get("cv_metrics", {})
check("Has cv_metrics", len(cv) > 0)
check("Has model_name", "model_name" in cv)
check("Has cv_r2_mean", "cv_r2_mean" in cv)
check("Has cv_mae_mean", "cv_mae_mean" in cv)

# Feature importance
fi = body.get("feature_importance", [])
check("Feature importance is a list", isinstance(fi, list))
check("At least 5 features returned", len(fi) >= 5)
check("Each has feature, label, importance_pct", all(
    "feature" in f and "label" in f and "importance_pct" in f for f in fi
))

# Campaign scores
cs = body.get("campaign_scores", [])
check("Campaign scores is a list", isinstance(cs, list))
check("Has scores for all campaigns", len(cs) >= 5)
check("Each has grade", all("grade" in c for c in cs))
check("Each has actual_conversion", all("actual_conversion" in c for c in cs))
check("Each has predicted_conversion", all("predicted_conversion" in c for c in cs))

# Archetypes
arch = body.get("archetypes", [])
check("Archetypes returned", isinstance(arch, list) and len(arch) > 0)
check("Each archetype has name", all("name" in a for a in arch))
check("Each archetype has avg_conversion_rate", all("avg_conversion_rate" in a for a in arch))

# Source breakdown
sb = body.get("source_breakdown", [])
check("Source breakdown returned", isinstance(sb, list) and len(sb) > 0)
check("Each source has avg_conversion", all("avg_conversion" in s for s in sb))

# Summary
summary = body.get("summary", {})
check("Summary has n_campaigns", "n_campaigns" in summary)
check("Summary has avg_conversion_rate", "avg_conversion_rate" in summary)
check("Summary has best_campaign", "best_campaign" in summary)
check("Summary has worst_campaign", "worst_campaign" in summary)

print(f"\n  Model: {cv.get('model_name')}")
print(f"  CV R²: {cv.get('cv_r2_mean', 0):.3f}")
print(f"  Campaigns analyzed: {summary.get('n_campaigns')}")
print(f"  Avg conversion: {summary.get('avg_conversion_rate')}%")


# ── TEST 5: Score-draft after analyze should succeed ─────────────────────────
print("\n🧪 TEST 5: Score-draft endpoint — after model training")

r = client.post("/api/score-draft", json={
    "headline": "Tell Mayor Chen: Stop the Downtown Demolition Before Friday",
    "body_text": (
        "Our historic Main Street buildings are scheduled for demolition this Friday. "
        "Mayor Chen has approved this plan despite overwhelming community opposition. "
        "Your voice matters — 1,200 neighbors have already signed. "
        "We need 500 more signatures before the council meeting on Thursday. "
        "Don't let them erase our community's history."
    ),
    "cta_text": "Tell Mayor Chen: Save Our Historic Main Street",
    "traffic_source": "email",
    "cause_category": "environment",
})

check("POST /api/score-draft returns 200", r.status_code == 200)
body = r.json()
check("Has grade", body.get("grade") in "ABCDF")
check("Has label", "label" in body and len(body["label"]) > 0)
check("Has color", "color" in body)
check("Has predicted_rate", "predicted_rate" in body)
check("Has z_score", "z_score" in body)
check("Has recommendations", isinstance(body.get("recommendations"), list))
check("Has features dict", isinstance(body.get("features"), dict))

print(f"\n  Predicted: {body.get('predicted_rate')}%")
print(f"  Grade: {body.get('grade')} ({body.get('label')})")
print(f"  Recommendations: {len(body.get('recommendations', []))}")


# ── TEST 6: Health endpoint reflects trained model ───────────────────────────
print("\n🧪 TEST 6: Health endpoint reflects trained model")

r = client.get("/api/health")
check("model_trained is true after analyze", r.json()["model_trained"] == True)


# ── TEST 7: Invalid inputs ──────────────────────────────────────────────────
print("\n🧪 TEST 7: Invalid inputs")

# Invalid CSV
r = client.post("/api/analyze", files={
    "file": ("bad.csv", b"not,a,valid\ncsv,at,all", "text/csv")
})
check("Invalid CSV returns 400", r.status_code == 400)

# Too-short body text
r = client.post("/api/score-draft", json={
    "headline": "Test",
    "body_text": "short",
    "cta_text": "Go",
})
check("Short fields return 422 validation error", r.status_code == 422)


# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} checks")
if failed == 0:
    print("🎉 All tests passed!")
else:
    print("⚠️  Some tests failed.")
    sys.exit(1)
