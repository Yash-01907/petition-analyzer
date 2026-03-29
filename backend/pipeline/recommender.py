# backend/pipeline/recommender.py
# Phase 5: Insight and Recommendation Engine.
#
# Three main functions:
#   generate_recommendations() — Rule-based, prioritized plain-English advice
#   compute_campaign_score()   — Grade/label/z-score from predicted conversion rate
#   compute_archetypes()       — KMeans clustering with dominant-trait summaries

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── Feature human-readable labels ────────────────────────────────────────────

FEATURE_LABELS = {
    "headline_is_imperative": "Imperative headline framing (Tell/Stop/Demand)",
    "headline_is_question": "Question headline framing",
    "headline_has_named_entity": "Named person/organization in headline",
    "headline_has_number": "Specific number in headline",
    "headline_has_deadline": "Urgency/deadline language in headline",
    "headline_power_word_count": "Power word density in headline",
    "cta_is_specific": "Specific CTA (vs. generic 'Sign Now')",
    "cta_has_named_official": "Named official in CTA",
    "cta_has_deadline": "Deadline language in CTA",
    "cta_collective_framing": "Collective framing in CTA ('Join 500 others')",
    "body_anger_score": "Anger/injustice framing in body",
    "body_fear_score": "Fear/urgency framing in body",
    "body_joy_score": "Hope/positive framing in body",
    "body_reading_grade": "Body text reading grade level",
    "body_flesch_score": "Body text reading ease",
    "body_word_count": "Body text length (word count)",
    "first_person_density": "First-person language density (we/our/I)",
    "second_person_density": "Second-person language density (you/your)",
    "has_image": "Hero image present",
    "has_video": "Video present",
    "campaign_duration_days": "Campaign duration",
    "source_email": "Traffic source: Email",
    "source_social": "Traffic source: Social",
    "source_organic": "Traffic source: Organic",
    "source_paid": "Traffic source: Paid",
    "category_environment": "Cause: Environment",
    "category_housing": "Cause: Housing",
    "category_healthcare": "Cause: Healthcare",
    "category_education": "Cause: Education",
    "category_transit": "Cause: Transit",
    "category_food_security": "Cause: Food Security",
    "category_civil_rights": "Cause: Civil Rights",
    "category_climate": "Cause: Climate",
}

# ── Recommendation rule templates ─────────────────────────────────────────────

RECOMMENDATION_RULES = [
    {
        "feature": "headline_is_imperative",
        "direction": "low",
        "threshold": 0.5,
        "grade_impact": "high",
        "title": "Reframe Your Headline with a Direct Imperative",
        "description": (
            "Your headline uses passive or declarative framing. "
            "Your top-converting past campaigns almost all open with a direct verb: "
            "'Tell', 'Stop', 'Demand', 'Protect'. "
            "Imperative headlines create immediate psychological ownership — the reader "
            "knows exactly what action they're being invited to take."
        ),
        "fallback_example": "Instead of: 'The New Development Plan Will Harm Our River'\nTry: 'Tell the Mayor: Stop the Riverside Development Now'",
    },
    {
        "feature": "headline_has_named_entity",
        "direction": "low",
        "threshold": 0.5,
        "grade_impact": "medium",
        "title": "Name a Specific Villain or Decision-Maker in the Headline",
        "description": (
            "Campaigns with named people or organizations in the headline convert significantly better. "
            "Anonymous threats feel abstract. A named official makes the petition feel urgent and actionable. "
            "Who specifically is making this decision? Put their name in the headline."
        ),
        "fallback_example": "Instead of: 'Stop the Harmful Rezoning Plan'\nTry: 'Tell Planning Director Walsh: Reject the Riverside Rezoning'",
    },
    {
        "feature": "headline_has_deadline",
        "direction": "low",
        "threshold": 0.5,
        "grade_impact": "high",
        "title": "Add a Deadline or Urgency Signal to the Headline",
        "description": (
            "Urgency language in headlines is one of the strongest predictors of conversion in your historical data. "
            "When readers feel the window is closing, they act immediately instead of returning later (and forgetting). "
            "Real deadlines work best — but even 'before it's too late' provides a lift."
        ),
        "fallback_example": "Instead of: 'Protect the Riverside Green Belt'\nTry: 'Only 10 Days Left: Protect the Riverside Green Belt Before the Vote'",
    },
    {
        "feature": "cta_is_specific",
        "direction": "low",
        "threshold": 0.5,
        "grade_impact": "high",
        "title": "Replace Generic CTA with a Specific, Outcome-Focused Ask",
        "description": (
            "Your current CTA is generic ('Sign Now', 'Add Your Name'). "
            "Your best-converting campaigns use CTAs that name the decision-maker and the specific outcome: "
            "'Tell Mayor Chen: Don't Approve This Plan'. "
            "This framing reduces the psychological distance between clicking and making a difference."
        ),
        "fallback_example": "Instead of: 'Sign the Petition'\nTry: 'Tell Mayor Chen: Our Community Says No to This Plan'",
    },
    {
        "feature": "body_reading_grade",
        "direction": "high",
        "threshold": 10.0,
        "grade_impact": "medium",
        "title": "Simplify Your Body Text Reading Level",
        "description": (
            "Your petition body reads at a Grade {value:.0f} level. "
            "Your highest-converting campaigns average Grade 7–8. "
            "Petitions get scanned, not read. Long, complex sentences lose people before they reach the signature button. "
            "Break up your longest paragraph. Replace any 3-syllable words you can simplify."
        ),
        "fallback_example": "Instead of: 'The proposed infrastructure modification will precipitate irreversible environmental consequences'\nTry: 'This plan will permanently damage our river — and can't be undone'",
    },
    {
        "feature": "body_word_count",
        "direction": "high",
        "threshold": 350,
        "grade_impact": "low",
        "title": "Shorten Your Body Text",
        "description": (
            "Your body text is {value:.0f} words — longer than your average high-converting campaign. "
            "Petitions are not essays. Readers decide to sign in 15–30 seconds. "
            "Aim for 150–250 words: one paragraph of context, one paragraph of impact, one call to urgency."
        ),
        "fallback_example": "Target structure:\nP1 (50 words): What's happening and who is responsible\nP2 (80 words): What happens if we do nothing\nP3 (40 words): Why your signature matters right now",
    },
    {
        "feature": "has_image",
        "direction": "low",
        "threshold": 0.5,
        "grade_impact": "medium",
        "title": "Add a Hero Image to the Petition Page",
        "description": (
            "Petitions with hero images convert better in your historical data. "
            "The image should show what's at stake (the river, the community center, affected families) "
            "— not a stock photo. Emotional imagery anchors the reader before they read a single word."
        ),
        "fallback_example": "Best image types:\n• The place/thing being threatened (river, park, building)\n• A face: an affected resident, a child, an elder\n• A clear visual contrast: before/after, healthy/damaged",
    },
    {
        "feature": "second_person_density",
        "direction": "low",
        "threshold": 0.02,
        "grade_impact": "low",
        "title": "Increase Direct 'You' Language in Body Text",
        "description": (
            "Your body text uses very little second-person language ('you', 'your'). "
            "Addressing the reader directly increases perceived personal relevance and conversion. "
            "Readers sign petitions when they feel personally implicated — not when they're being told facts."
        ),
        "fallback_example": "Instead of: 'The community will be affected by this decision'\nTry: 'Your neighborhood will change forever if this passes — and your signature can stop it'",
    },
]


def _build_contextual_example(
    rule_feature: str, headline: str, body: str, cta: str
) -> str:
    """Generate a contextual example that references the user's actual input.

    Takes the user's headline/body/CTA and produces a 'You wrote X → Try Y'
    rewrite suggestion specific to the triggered rule.
    """
    hl_short = headline[:80] if headline else "your headline"
    body_first_sentence = (body.split(".")[0].strip() + ".") if body else "your body text"
    # Truncate body sentence if too long
    if len(body_first_sentence) > 120:
        body_first_sentence = body_first_sentence[:117] + "..."

    if rule_feature == "headline_is_imperative":
        # Extract the core topic from the headline
        return (
            f"You wrote: '{hl_short}'\n"
            f"Try: 'Tell [Decision-Maker]: {hl_short.rstrip('!.?')} — Act Now'"
        )

    elif rule_feature == "headline_has_named_entity":
        return (
            f"You wrote: '{hl_short}'\n"
            f"Try: 'Tell [Name the responsible official]: {hl_short.rstrip('!.?')}'"
        )

    elif rule_feature == "headline_has_deadline":
        return (
            f"You wrote: '{hl_short}'\n"
            f"Try: 'Last Chance: {hl_short.rstrip('!.?')} — Before the Vote on [Date]'"
        )

    elif rule_feature == "cta_is_specific":
        cta_short = cta[:60] if cta else "Sign the Petition"
        return (
            f"You wrote: '{cta_short}'\n"
            f"Try: 'Tell [Official Name]: [Specific outcome you want them to take]'"
        )

    elif rule_feature == "body_reading_grade":
        return (
            f"Your opening: '{body_first_sentence}'\n"
            f"Try rewriting with shorter words and simpler sentences. "
            f"Replace multi-syllable words and break long sentences into two."
        )

    elif rule_feature == "body_word_count":
        word_count = len(body.split()) if body else 0
        return (
            f"Your body is {word_count} words. Target 150–250 words:\n"
            f"P1 (~50 words): What's happening and who's responsible\n"
            f"P2 (~80 words): What happens if we don't act\n"
            f"P3 (~40 words): Why your signature matters right now"
        )

    elif rule_feature == "has_image":
        return (
            "Your petition has no hero image. Add a photo of:\n"
            "• The place or thing being threatened\n"
            "• An affected person (a face creates empathy)\n"
            "• A visual contrast: before vs. after"
        )

    elif rule_feature == "second_person_density":
        return (
            f"Your opening: '{body_first_sentence}'\n"
            f"Try rewriting to address the reader directly: "
            f"'Your [community/neighborhood/family] will be affected — and your signature can stop it'"
        )

    return ""  # Unknown feature — no contextual example possible


def generate_recommendations(
    feature_values: dict,
    feature_importance: dict,
    campaign_averages: dict,
    headline: str = "",
    body: str = "",
    cta: str = "",
) -> list:
    """Generate prioritized, plain-English recommendations for a campaign.

    Evaluates each rule template against the campaign's feature values.
    Only triggered rules are included, sorted by grade impact and SHAP importance.
    Examples reference the user's actual input text when available.

    Args:
        feature_values: Dict of {feature_name: value} for the scored campaign.
        feature_importance: Dict of {feature_name: mean_abs_shap} from training.
        campaign_averages: Dict of {feature_name: dataset_mean} for comparison.
        headline: The user's draft headline (for contextual examples).
        body: The user's draft body text (for contextual examples).
        cta: The user's draft CTA text (for contextual examples).

    Returns:
        List of up to 6 recommendation dicts, sorted by impact.
    """
    recommendations = []
    has_user_text = bool(headline or body or cta)

    for rule in RECOMMENDATION_RULES:
        feat = rule["feature"]
        val = feature_values.get(feat)
        if val is None:
            continue

        triggered = False
        if rule["direction"] == "low" and val < rule["threshold"]:
            triggered = True
        elif rule["direction"] == "high" and val > rule["threshold"]:
            triggered = True

        if triggered:
            importance = feature_importance.get(feat, 0)
            avg = campaign_averages.get(feat, rule["threshold"])
            description = rule["description"].replace("{value:.0f}", f"{val:.0f}")

            # Generate contextual example if user text is available,
            # otherwise fall back to generic example
            if has_user_text:
                example = _build_contextual_example(feat, headline, body, cta)
            else:
                example = rule.get("fallback_example", "")

            # If contextual generation returned empty, use fallback
            if not example:
                example = rule.get("fallback_example", "")

            recommendations.append({
                "title": rule["title"],
                "description": description,
                "example": example,
                "grade_impact": rule["grade_impact"],
                "feature": feat,
                "feature_label": FEATURE_LABELS.get(feat, feat),
                "current_value": round(float(val), 3),
                "average_value": round(float(avg), 3),
                "importance_score": round(float(importance), 4),
            })

    # Sort: high impact first, then by SHAP importance
    priority = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(
        key=lambda r: (priority[r["grade_impact"]], -r["importance_score"])
    )

    return recommendations[:6]


def compute_campaign_score(
    predicted_rate: float, avg_rate: float, std_rate: float
) -> dict:
    """Convert a predicted conversion rate into a letter grade and percentile.

    Uses z-score bucketing:
        z >= 1.5  → A (Excellent)
        z >= 0.5  → B (Above Average)
        z >= -0.5 → C (Average)
        z >= -1.5 → D (Below Average)
        z < -1.5  → F (Needs Significant Work)

    Args:
        predicted_rate: Model-predicted conversion rate for the draft.
        avg_rate: Mean conversion rate from the training dataset.
        std_rate: Std deviation of conversion rates from training data.

    Returns:
        Dict with grade, label, color, predicted_rate, avg_rate, z_score.
    """
    z = (predicted_rate - avg_rate) / (std_rate + 1e-6)

    if z >= 1.5:
        grade, label, color = "A", "Excellent", "green"
    elif z >= 0.5:
        grade, label, color = "B", "Above Average", "green"
    elif z >= -0.5:
        grade, label, color = "C", "Average", "yellow"
    elif z >= -1.5:
        grade, label, color = "D", "Below Average", "orange"
    else:
        grade, label, color = "F", "Needs Significant Work", "red"

    return {
        "grade": grade,
        "label": label,
        "color": color,
        "predicted_rate": round(predicted_rate, 2),
        "avg_rate": round(avg_rate, 2),
        "z_score": round(z, 2),
    }


def compute_archetypes(
    X: pd.DataFrame, df: pd.DataFrame, n_clusters: int = 4
) -> list:
    """Cluster past campaigns into archetypes using KMeans on NLP features.

    Args:
        X: Feature matrix from extract_features.
        df: Original campaign DataFrame (needs headline and conversion_rate).
        n_clusters: Number of archetypes to produce.

    Returns:
        List of archetype dicts sorted by avg_conversion_rate descending.
        Each has: id, name, campaign_count, avg_conversion_rate,
                  dominant_traits, example_headlines.
    """
    cluster_features = [
        "headline_is_imperative", "headline_is_question",
        "headline_has_named_entity", "headline_has_deadline",
        "body_fear_score", "body_anger_score", "body_joy_score",
        "cta_is_specific", "body_word_count", "body_reading_grade",
    ]

    available = [f for f in cluster_features if f in X.columns]
    X_cluster = X[available].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    archetypes = []
    df_clustered = df.copy()
    df_clustered["archetype"] = labels

    archetype_names = {
        0: "The Activist Call",
        1: "The Informer",
        2: "The Hope Builder",
        3: "The Urgent Alarm",
    }

    for cluster_id in range(n_clusters):
        mask = df_clustered["archetype"] == cluster_id
        cluster_campaigns = df_clustered[mask]

        avg_conv = cluster_campaigns["conversion_rate"].mean()
        count = mask.sum()

        # Dominant features for this cluster
        traits = []
        centroid = km.cluster_centers_[cluster_id]
        for i, feat in enumerate(available):
            if centroid[i] > 0.7:
                traits.append(FEATURE_LABELS.get(feat, feat))

        archetypes.append({
            "id": cluster_id,
            "name": archetype_names.get(cluster_id, f"Archetype {cluster_id + 1}"),
            "campaign_count": int(count),
            "avg_conversion_rate": round(float(avg_conv), 2),
            "dominant_traits": traits[:4],
            "example_headlines": cluster_campaigns["headline"].head(2).tolist(),
        })

    archetypes.sort(key=lambda a: -a["avg_conversion_rate"])
    return archetypes
