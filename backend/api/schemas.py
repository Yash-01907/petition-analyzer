# backend/api/schemas.py
# Pydantic request/response models for the API layer.

from pydantic import BaseModel, Field
from typing import Optional


class DraftScoreRequest(BaseModel):
    """Request body for POST /api/score-draft."""
    headline: str = Field(..., min_length=5, max_length=300)
    body_text: str = Field(..., min_length=20, max_length=5000)
    cta_text: str = Field(..., min_length=2, max_length=200)
    traffic_source: Optional[str] = "email"
    cause_category: Optional[str] = "environment"
