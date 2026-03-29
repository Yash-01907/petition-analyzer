# backend/pipeline/ingestion.py
# Phase 2: CSV loading, validation, cleaning, and conversion rate computation.
# Handles missing columns, invalid values, and outlier flagging.

import pandas as pd
from typing import Tuple, List

REQUIRED_COLUMNS = [
    "headline", "body_text", "cta_text",
    "unique_visitors", "signatures", "traffic_source",
]

OPTIONAL_COLUMNS_DEFAULTS = {
    "cause_category": "general",
    "campaign_duration_days": 14,
    "launch_date": "2024-01-01",
    "has_image": False,
    "has_video": False,
    "campaign_id": None,
}

VALID_TRAFFIC_SOURCES = {"email", "social", "organic", "paid"}


def load_and_validate_csv(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Validate and clean a campaign DataFrame.

    Ensures required columns exist, fills optional defaults,
    normalizes text, coerces numerics, computes conversion_rate,
    validates traffic sources, and flags statistical outliers.

    Args:
        df: Raw DataFrame from CSV upload.

    Returns:
        Tuple of (cleaned DataFrame, list of user-facing warning strings).

    Raises:
        ValueError: If any required columns are missing.
    """
    errors = []

    # ── 1. Check required columns ─────────────────────────────────────────
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── 2. Fill optional columns with defaults ────────────────────────────
    for col, default in OPTIONAL_COLUMNS_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
            errors.append(f"Column '{col}' not found — using default: {default}")

    # ── 3. Generate campaign_id if missing or all null ────────────────────
    if df["campaign_id"].isna().all():
        df["campaign_id"] = [f"CAMP-{i:04d}" for i in range(len(df))]

    # ── 4. Clean and normalize text fields ────────────────────────────────
    for col in ["headline", "body_text", "cta_text"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # ── 5. Coerce numeric columns safely ──────────────────────────────────
    df["unique_visitors"] = pd.to_numeric(
        df["unique_visitors"], errors="coerce"
    ).fillna(1000)
    df["signatures"] = pd.to_numeric(
        df["signatures"], errors="coerce"
    ).fillna(0)

    # Ensure visitors > 0 to avoid division by zero
    df.loc[df["unique_visitors"] <= 0, "unique_visitors"] = 1000

    # ── 6. Compute conversion rate ────────────────────────────────────────
    df["conversion_rate"] = (
        df["signatures"] / df["unique_visitors"] * 100
    ).clip(0, 100)

    # ── 7. Validate and fix traffic source values ─────────────────────────
    df["traffic_source"] = df["traffic_source"].astype(str).str.lower().str.strip().fillna("email")
    invalid_sources = df[~df["traffic_source"].isin(VALID_TRAFFIC_SOURCES)]
    if len(invalid_sources) > 0:
        df.loc[
            ~df["traffic_source"].isin(VALID_TRAFFIC_SOURCES), "traffic_source"
        ] = "email"
        errors.append(
            f"{len(invalid_sources)} rows had invalid traffic_source — defaulted to 'email'"
        )

    # ── 8. Coerce boolean columns ─────────────────────────────────────────
    truthy_vals = {"true", "1", "yes", "t", "y"}
    for col in ["has_image", "has_video"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: str(x).lower().strip() in truthy_vals if pd.notnull(x) else False
            )

    # ── 9. Coerce campaign_duration_days ──────────────────────────────────
    df["campaign_duration_days"] = pd.to_numeric(
        df["campaign_duration_days"], errors="coerce"
    ).fillna(14).astype(int)

    # ── 10. Flag outliers ─────────────────────────────────────────────────
    if len(df) > 5:
        q99 = df["conversion_rate"].quantile(0.99)
        outliers = df[df["conversion_rate"] > q99]
        if len(outliers) > 0:
            errors.append(
                f"{len(outliers)} campaigns had unusually high conversion rates (>{q99:.1f}%) "
                f"and were flagged as potential viral outliers. Review before analysis."
            )

    # ── 11. Drop rows with empty required text ───────────────────────────
    empty_text = df[
        (df["headline"] == "") | (df["body_text"] == "") | (df["cta_text"] == "")
    ]
    if len(empty_text) > 0:
        errors.append(
            f"{len(empty_text)} rows had empty headline, body, or CTA text and were removed."
        )
        df = df[
            (df["headline"] != "") & (df["body_text"] != "") & (df["cta_text"] != "")
        ].reset_index(drop=True)

    return df, errors
