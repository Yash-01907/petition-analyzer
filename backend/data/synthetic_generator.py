# backend/data/synthetic_generator.py
# Phase 1: Synthetic campaign data generator with signal-injected conversion rates.
# Generates realistic petition campaign records for training and demos.

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import faker

fake = faker.Faker()
random.seed(42)
np.random.seed(42)

# ── Template pools ────────────────────────────────────────────────────────────

HEADLINE_TEMPLATES = {
    "imperative_villain": [
        "Tell {official} to Stop {threat} Now",
        "Demand {official} Protect {asset} from {threat}",
        "Stop {official}'s Plan to {action} Our {asset}",
        "Tell {body}: {asset} Is Not for Sale",
        "{official} Must Resign After {scandal}",
    ],
    "question": [
        "Should {official} Be Allowed to {action} Our {asset}?",
        "Why Is {body} Ignoring {issue}?",
        "How Long Will We Let {official} Destroy {asset}?",
    ],
    "declarative_stat": [
        "{number} Families Will Lose Their Homes. We Must Act.",
        "{number} Children Affected — And {official} Is Silent",
        "Only {days} Days Left to Stop {threat}",
    ],
    "declarative_neutral": [
        "Support {asset} Conservation",
        "A Petition for {issue} Reform",
        "Citizens Against {threat}",
        "Save Our {asset}",
    ],
    "urgency_deadline": [
        "URGENT: Decision on {issue} Is {days} Days Away",
        "Final Call: Protect {asset} Before {date}",
        "Last Chance to Stop {threat} — Vote in {days} Days",
    ],
}

OFFICIALS = [
    "Mayor Chen", "Governor Mills", "Councillor Park", "Minister Davies",
    "Senator Torres", "Director Walsh", "Planning Board", "City Council",
]
THREATS = [
    "the new dam", "deforestation", "toxic dumping", "rezoning",
    "budget cuts", "school closures", "rate hikes", "the bypass road",
]
ASSETS = [
    "our river", "the community center", "public schools", "the green belt",
    "affordable housing", "our coastline", "the urban forest", "the park",
]
ISSUES = [
    "housing", "clean water", "healthcare access", "climate policy",
    "education funding", "transit", "air quality", "food security",
]
BODIES = ["the EPA", "City Hall", "the Planning Commission", "the School Board"]

CTA_TEMPLATES = {
    "specific_official": [
        "Tell {official}: Protect {asset}",
        "Sign to Tell {official}: We Say No to {threat}",
        "Demand {official} Reverse This Decision Now",
        "Tell {official}: {asset} Belongs to the Community",
    ],
    "generic_action": [
        "Sign the Petition",
        "Add Your Name",
        "Sign Now",
        "Join the Fight",
        "Take Action",
    ],
    "collective_we": [
        "Join {number} Neighbors Standing Up",
        "Add Your Voice to {number} Others",
        "We Are {number} Strong — Will You Join?",
    ],
    "deadline_cta": [
        "Sign Before {date} — Every Signature Counts",
        "We Have {days} Days. Sign Now.",
        "Deadline: {date}. Sign Today.",
    ],
}

BODY_INTROS = {
    "fear": [
        "If we do nothing, {threat} will devastate {asset} within {days} days.",
        "Right now, {official} is moving forward with a plan that will destroy {asset} forever.",
        "Our {asset} is under immediate threat — and most people don't even know it yet.",
    ],
    "anger": [
        "{official} has ignored our community for the last time.",
        "Despite {number} letters and {number} public comments, {official} is pushing ahead anyway.",
        "This is not about policy — it's about {official} choosing corporate interests over our families.",
    ],
    "hope": [
        "Together, we can stop {threat} and protect {asset} for future generations.",
        "Communities like ours have won before. With your signature, we can win again.",
        "Every signature sends a message: {asset} is worth fighting for.",
    ],
    "neutral": [
        "The proposed {action} plan would affect {asset} in the following ways.",
        "We are calling on {official} to reconsider the recent decision regarding {asset}.",
        "This petition addresses the impact of {action} on {asset}.",
    ],
}

TRAFFIC_SOURCES = ["email", "social", "organic", "paid"]
CAUSE_CATEGORIES = [
    "environment", "housing", "healthcare", "education",
    "transit", "food_security", "civil_rights", "climate",
]

# ── Conversion rate distributions by traffic source ───────────────────────────
# Based on realistic advocacy benchmarks
CONVERSION_PARAMS = {
    "email":   {"base": 18.0, "std": 6.0},   # 12–24% typical
    "social":  {"base": 3.5,  "std": 2.0},   # 1.5–5.5%
    "organic": {"base": 6.0,  "std": 3.0},   # 3–9%
    "paid":    {"base": 4.5,  "std": 2.5},   # 2–7%
}

# ── Feature-to-conversion lift table (used to inject signal) ─────────────────
FEATURE_LIFTS = {
    "imperative_villain": 3.5,        # headline type lift in pp
    "question": 0.5,
    "declarative_stat": 2.0,
    "declarative_neutral": -2.5,
    "urgency_deadline": 4.0,
    "specific_official_cta": 2.5,
    "generic_action_cta": -2.0,
    "deadline_cta": 3.0,
    "collective_we_cta": 1.5,
    "fear_body": 1.5,
    "anger_body": 2.0,
    "hope_body": 1.0,
    "neutral_body": -3.0,
    "short_body": 1.0,               # < 150 words
    "long_body": -1.5,               # > 400 words
    "has_image": 1.2,
    "has_video": 2.5,
    "has_deadline_in_text": 2.0,
    "has_named_person": 1.5,
    "high_reading_level": -2.0,      # Grade > 10
}


def pick(templates, **kwargs):
    """Pick a random template from a list and fill in kwargs."""
    tpl = random.choice(templates)
    fills = {
        "official": random.choice(OFFICIALS),
        "threat": random.choice(THREATS),
        "asset": random.choice(ASSETS),
        "issue": random.choice(ISSUES),
        "body": random.choice(BODIES),
        "action": random.choice([
            "rezone", "demolish", "defund", "privatize",
            "approve", "build", "expand", "cut",
        ]),
        "number": random.randint(100, 50000),
        "days": random.randint(3, 30),
        "date": fake.date_this_year().strftime("%B %d"),
        "scandal": random.choice([
            "the leaked memo", "last month's vote",
            "the corruption inquiry", "this decision",
        ]),
    }
    fills.update(kwargs)
    try:
        return tpl.format(**fills)
    except KeyError:
        return tpl


def generate_body(body_type, word_length="medium"):
    """Generate petition body text of varying lengths."""
    lengths = {"short": 80, "medium": 200, "long": 420}
    target_words = lengths.get(word_length, 200)

    intro = pick(BODY_INTROS[body_type])
    sentences = [intro]

    # Add middle paragraphs
    middles = [
        f"The impact on local families would be severe. {random.randint(50, 5000)} residents "
        f"have already signed a community letter — but {random.choice(OFFICIALS)} has not responded.",
        f"Independent experts warn that {random.choice(THREATS)} will cost the community "
        f"an estimated ${random.randint(1, 20)}M over the next decade.",
        f"We have {random.randint(5, 60)} days before the final decision is made. "
        f"Your signature today could change everything.",
        f"This is not the first time {random.choice(OFFICIALS)} has ignored our community. "
        f"In {fake.year()}, a similar proposal was stopped by grassroots action — just like this.",
        f"Across the country, communities facing the same challenge have succeeded when they "
        f"spoke up together. We can do the same.",
        f"Our children will inherit the consequences of this decision. "
        f"Let's make sure they inherit {random.choice(ASSETS)} — not {random.choice(THREATS)}.",
    ]

    while sum(len(s.split()) for s in sentences) < target_words:
        sentences.append(random.choice(middles))

    return " ".join(sentences[:6])


def generate_campaign(idx):
    """Generate a single realistic campaign row with signal-injected conversion rate."""
    # Choose headline type
    headline_type = random.choices(
        list(HEADLINE_TEMPLATES.keys()),
        weights=[30, 15, 20, 20, 15],
        k=1,
    )[0]
    headline = pick(HEADLINE_TEMPLATES[headline_type])

    # Choose CTA type
    cta_type = random.choices(
        list(CTA_TEMPLATES.keys()),
        weights=[25, 35, 20, 20],
        k=1,
    )[0]
    cta = pick(CTA_TEMPLATES[cta_type])

    # Choose body type and length
    body_type = random.choices(
        list(BODY_INTROS.keys()),
        weights=[25, 25, 25, 25],
        k=1,
    )[0]
    body_length = random.choices(
        ["short", "medium", "long"], weights=[25, 50, 25], k=1
    )[0]
    body = generate_body(body_type, body_length)

    # Traffic and cause
    traffic_source = random.choices(
        TRAFFIC_SOURCES, weights=[40, 30, 20, 10], k=1
    )[0]
    cause = random.choice(CAUSE_CATEGORIES)

    # Compute conversion rate with signal injection
    params = CONVERSION_PARAMS[traffic_source]
    base_rate = np.random.normal(params["base"], params["std"])

    # Add feature lifts
    lift = 0
    lift += FEATURE_LIFTS.get(headline_type, 0)
    lift += FEATURE_LIFTS.get(f"{cta_type}_cta", 0)
    lift += FEATURE_LIFTS.get(f"{body_type}_body", 0)
    lift += FEATURE_LIFTS.get(
        body_length == "short" and "short_body"
        or body_length == "long" and "long_body"
        or "",
        0,
    )

    has_image = random.random() > 0.4
    has_video = random.random() > 0.8
    if has_image:
        lift += FEATURE_LIFTS["has_image"]
    if has_video:
        lift += FEATURE_LIFTS["has_video"]

    conversion_rate = max(0.5, min(45.0, base_rate + lift + np.random.normal(0, 1.5)))

    # Compute signatures from conversion rate
    visitors = random.randint(500, 15000)
    signatures = int(visitors * conversion_rate / 100)

    # Launch date
    launch = fake.date_between(start_date="-2y", end_date="-7d")
    duration = random.randint(7, 60)

    return {
        "campaign_id": f"CAMP-{2022 + idx // 60}-{idx:03d}",
        "headline": headline,
        "body_text": body,
        "cta_text": cta,
        "unique_visitors": visitors,
        "signatures": signatures,
        "conversion_rate": round(conversion_rate, 2),
        "traffic_source": traffic_source,
        "cause_category": cause,
        "campaign_duration_days": duration,
        "launch_date": launch.strftime("%Y-%m-%d"),
        "has_image": has_image,
        "has_video": has_video,
        # Ground truth labels (for debugging/validation only)
        "_headline_type": headline_type,
        "_cta_type": cta_type,
        "_body_type": body_type,
        "_body_length": body_length,
    }


def generate_dataset(n=120, output_path=None):
    """Generate n synthetic campaigns and save to CSV.

    Args:
        n: Number of campaigns to generate (default 120).
        output_path: File path for. Defaults to data/sample_campaigns.csv
                     relative to this file's directory.

    Returns:
        DataFrame with all campaigns (including ground-truth debug columns).
    """
    import os

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "sample_campaigns.csv")

    campaigns = [generate_campaign(i) for i in range(n)]
    df = pd.DataFrame(campaigns)

    # Drop ground-truth labels before saving (they won't be in real data)
    user_columns = [c for c in df.columns if not c.startswith("_")]
    df[user_columns].to_csv(output_path, index=False)

    print(f"Generated {n} campaigns → {output_path}")
    print(f"Conversion rate range: {df['conversion_rate'].min():.1f}% – {df['conversion_rate'].max():.1f}%")
    print(f"Mean conversion rate:  {df['conversion_rate'].mean():.1f}%")
    print(f"Columns: {list(df[user_columns].columns)}")

    return df


if __name__ == "__main__":
    generate_dataset(120)
