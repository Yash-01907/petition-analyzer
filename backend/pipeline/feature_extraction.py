# backend/pipeline/feature_extraction.py
# Phase 3: Complete NLP + structural feature extraction engine.
#
# Extracts 30+ numeric features from each campaign's text fields:
#   Group A — Structural / Metadata
#   Group B — Headline type classification
#   Group C — Sentiment & Emotion (VADER + NRCLex)
#   Group D — Linguistic quality / readability
#   Group E — CTA quality features
#   Group F — Temporal & contextual features

import re
import spacy
import textstat
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
from datetime import datetime

# Load spaCy model (run: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()

# ── Constants ─────────────────────────────────────────────────────────────────

URGENCY_WORDS = [
    "urgent", "deadline", "last chance", "final", "act now", "today only",
    "hours left", "days left", "before it's too late", "running out of time",
    "critical", "emergency", "immediately", "expires", "cutoff",
]

IMPERATIVE_STARTERS = [
    "tell", "stop", "demand", "protect", "save", "end", "fight", "join",
    "sign", "act", "help", "stand", "support", "reject", "prevent", "urge",
]

POWER_WORDS = [
    "demand", "destroy", "protect", "urgent", "crisis", "fight", "now",
    "stop", "save", "justice", "right", "wrong", "threat", "fear", "hope",
    "future", "children", "community", "families", "together", "stand",
]

FIRST_PERSON = re.compile(r"\b(i|we|our|us|my)\b", re.IGNORECASE)
SECOND_PERSON = re.compile(r"\b(you|your|yours|yourself)\b", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"\b\d[\d,]*\b")
QUESTION_PATTERN = re.compile(r"\?$")


# ── Group A: Structural features ─────────────────────────────────────────────

def extract_structural_features(row):
    """Extract word counts, media presence, and campaign duration."""
    features = {}
    headline = str(row.get("headline", ""))
    body = str(row.get("body_text", ""))
    cta = str(row.get("cta_text", ""))

    features["headline_word_count"] = len(headline.split())
    features["body_word_count"] = len(body.split())
    features["cta_word_count"] = len(cta.split())
    features["has_image"] = int(bool(row.get("has_image", False)))
    features["has_video"] = int(bool(row.get("has_video", False)))
    features["campaign_duration_days"] = int(row.get("campaign_duration_days", 14))
    return features


# ── Group B: Headline type features ──────────────────────────────────────────

def extract_headline_features(headline):
    """Classify headline type and extract entity/urgency signals."""
    features = {}
    hl = str(headline).strip()
    doc = nlp(hl)

    # Type classification
    features["headline_is_question"] = int(bool(QUESTION_PATTERN.search(hl)))

    first_token_lemma = doc[0].lemma_.lower() if doc else ""
    features["headline_is_imperative"] = int(
        first_token_lemma in IMPERATIVE_STARTERS
        or (doc and doc[0].pos_ == "VERB")
    )

    # Entity checks
    has_person_org = any(
        ent.label_ in ("PERSON", "ORG", "GPE") for ent in doc.ents
    )
    features["headline_has_named_entity"] = int(has_person_org)
    features["headline_has_number"] = int(bool(NUMBER_PATTERN.search(hl)))

    # Urgency
    hl_lower = hl.lower()
    features["headline_has_deadline"] = int(
        any(word in hl_lower for word in URGENCY_WORDS)
    )

    # Power word count
    words_lower = set(hl_lower.split())
    features["headline_power_word_count"] = len(words_lower & set(POWER_WORDS))

    return features


# ── Group C: Sentiment & Emotion features ────────────────────────────────────

def extract_sentiment_features(text, prefix):
    """Extract VADER sentiment and NRC emotion scores."""
    features = {}
    text = str(text)

    # VADER sentiment
    scores = vader.polarity_scores(text)
    features[f"{prefix}_sentiment"] = scores["compound"]
    features[f"{prefix}_sentiment_neg"] = scores["neg"]
    features[f"{prefix}_sentiment_pos"] = scores["pos"]

    # NRC emotion lexicon
    try:
        nrc = NRCLex(text)
        emotion_freqs = nrc.affect_frequencies
        for emotion in [
            "anger", "fear", "joy", "disgust", "trust",
            "anticipation", "sadness", "surprise",
        ]:
            features[f"{prefix}_{emotion}_score"] = emotion_freqs.get(emotion, 0.0)
    except Exception:
        for emotion in [
            "anger", "fear", "joy", "disgust", "trust",
            "anticipation", "sadness", "surprise",
        ]:
            features[f"{prefix}_{emotion}_score"] = 0.0

    return features


# ── Group D: Readability & Linguistic quality ────────────────────────────────

def extract_readability_features(body):
    """Extract reading level, sentence stats, and pronoun densities."""
    features = {}
    body = str(body)

    try:
        features["body_reading_grade"] = textstat.flesch_kincaid_grade(body)
        features["body_flesch_score"] = textstat.flesch_reading_ease(body)
        features["body_sentence_count"] = textstat.sentence_count(body)
        features["body_avg_sentence_length"] = textstat.avg_sentence_length(body)
    except Exception:
        features["body_reading_grade"] = 8.0
        features["body_flesch_score"] = 60.0
        features["body_sentence_count"] = 5
        features["body_avg_sentence_length"] = 15.0

    word_count = len(body.split())
    if word_count > 0:
        features["first_person_density"] = len(FIRST_PERSON.findall(body)) / word_count
        features["second_person_density"] = len(SECOND_PERSON.findall(body)) / word_count
    else:
        features["first_person_density"] = 0.0
        features["second_person_density"] = 0.0

    return features


# ── Group E: CTA quality features ───────────────────────────────────────────

def extract_cta_features(cta):
    """Assess specificity, urgency, and framing of the call-to-action."""
    features = {}
    cta = str(cta)
    doc = nlp(cta)
    cta_lower = cta.lower()

    # Named official in CTA
    has_entity = any(ent.label_ in ("PERSON", "ORG") for ent in doc.ents)
    features["cta_has_named_official"] = int(has_entity)

    # Deadline / urgency in CTA
    features["cta_has_deadline"] = int(
        any(word in cta_lower for word in URGENCY_WORDS)
    )

    # Specificity heuristic: more than 5 words AND not a generic phrase
    generic_ctas = {
        "sign the petition", "sign now", "add your name",
        "take action", "join the fight", "sign",
    }
    is_specific = len(cta.split()) > 5 and cta_lower not in generic_ctas
    features["cta_is_specific"] = int(is_specific)

    # Collective framing
    collective_pattern = re.compile(
        r"join\s+\d+|we\s+are\s+\d+|\d+\s+(others|neighbors|people)",
        re.IGNORECASE,
    )
    features["cta_collective_framing"] = int(bool(collective_pattern.search(cta)))

    return features


# ── Group F: Temporal & contextual features ──────────────────────────────────

def extract_temporal_features(launch_date):
    """Extract month, day of week, and weekday flag from launch date."""
    features = {}
    try:
        if isinstance(launch_date, str):
            dt = datetime.strptime(launch_date, "%Y-%m-%d")
        else:
            dt = launch_date
        features["launch_month"] = dt.month
        features["launch_dayofweek"] = dt.weekday()
        features["launch_is_weekday"] = int(dt.weekday() < 5)
    except Exception:
        features["launch_month"] = 6
        features["launch_dayofweek"] = 1
        features["launch_is_weekday"] = 1
    return features


# ── Known categorical values (fixed vocabulary) ──────────────────────────────

TRAFFIC_SOURCES = ["email", "social", "organic", "paid"]
CAUSE_CATEGORIES = [
    "environment", "housing", "healthcare", "education",
    "transit", "food_security", "civil_rights", "climate",
]


def encode_categoricals(row):
    """One-hot encode traffic_source and cause_category with a fixed vocabulary.

    Unlike LabelEncoder, this produces identical encodings whether called on
    a 120-row training DataFrame or a single-row scoring request.
    """
    features = {}
    source = str(row.get("traffic_source", "email")).lower().strip()
    category = str(row.get("cause_category", "environment")).lower().strip()

    for s in TRAFFIC_SOURCES:
        features[f"source_{s}"] = int(source == s)

    for c in CAUSE_CATEGORIES:
        features[f"category_{c}"] = int(category == c)

    return features


# ── Master feature extraction function ───────────────────────────────────────

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the full numeric feature matrix from a campaign DataFrame.

    Runs all six feature groups on every row and one-hot encodes categoricals.

    Args:
        df: Cleaned campaign DataFrame (output of ingestion pipeline).

    Returns:
        DataFrame where each row is an all-numeric feature vector.
        Column order is deterministic.
    """
    feature_rows = []

    for _, row in df.iterrows():
        features = {}
        features.update(extract_structural_features(row))
        features.update(extract_headline_features(row.get("headline", "")))
        features.update(extract_sentiment_features(row.get("headline", ""), "headline"))
        features.update(extract_sentiment_features(row.get("body_text", ""), "body"))
        features.update(extract_readability_features(row.get("body_text", "")))
        features.update(extract_cta_features(row.get("cta_text", "")))
        features.update(extract_temporal_features(row.get("launch_date", "")))
        features.update(encode_categoricals(row))
        feature_rows.append(features)

    return pd.DataFrame(feature_rows)

