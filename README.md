# Petition Effectiveness Analyzer

A data-driven tool that helps advocacy organizations understand what textual and structural features of their petition campaigns correlate with higher signature conversion rates. Upload your campaign history, train an ML model, and score new drafts before publishing.

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+

### Backend

```bash
cd petition-analyzer/backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Generate sample dataset (optional — one is pre-included)
python data/synthetic_generator.py

# Start the API server
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### Frontend

```bash
cd petition-analyzer/frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

App: http://localhost:5173

### Running Tests

```bash
cd petition-analyzer/backend

# Activate venv first, then run all test suites:
python tests/test_ingestion.py
python tests/test_feature_extraction.py
python tests/test_modeling.py
python tests/test_recommender.py
python tests/test_api.py
```

All 5 suites (210 total checks) should pass with `🎉 All tests passed!`.

## Demo Walkthrough

1. **Open the app** at http://localhost:5173 (backend must be running on :8000).
2. **Click "Generate & Load Mock Data"** — this fetches the pre-generated 120-row synthetic dataset from the backend, uploads it to `/api/analyze`, trains the model, and transitions to the Analysis Dashboard.
3. **Explore the dashboard** — review the summary cards (campaigns analyzed, average conversion rate, top traffic source), the SHAP feature importance bar chart, the archetype clusters, and the campaign leaderboard.
4. **Switch to the "✏️ Score Draft" tab** — enter a headline, body text, and CTA. The system will predict a conversion rate, assign a letter grade (A–F), and return actionable recommendations ranked by impact.

## Project Structure

```
petition-analyzer/
├── backend/
│   ├── main.py                       # FastAPI app (5 endpoints)
│   ├── requirements.txt
│   ├── data/
│   │   ├── synthetic_generator.py    # Generates realistic campaign data
│   │   └── sample_campaigns.csv      # 120 pre-generated campaigns
│   ├── pipeline/
│   │   ├── ingestion.py              # CSV validation & cleaning
│   │   ├── feature_extraction.py     # 49 NLP/structural features
│   │   ├── modeling.py               # Adaptive ML (Ridge/RF/XGBoost) + SHAP
│   │   └── recommender.py           # Grading, recommendations, archetypes
│   ├── api/
│   │   └── schemas.py                # Pydantic request models
│   └── tests/                        # 5 test suites, 210 checks
├── frontend/
│   ├── src/
│   │   ├── App.jsx                   # Shell with tab routing
│   │   └── components/
│   │       ├── UploadPanel.jsx       # CSV upload + mock data loader
│   │       ├── AnalysisDashboard.jsx # Charts, tables, archetypes
│   │       ├── DraftScorer.jsx       # Score form + grade card
│   │       └── RecommendationCard.jsx
│   ├── package.json
│   └── vite.config.js                # Proxy /api → localhost:8000
├── .env.example
└── README.md
```

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, spaCy, VADER, NRCLex, Textstat |
| ML | XGBoost / RandomForest / Ridge (adaptive by dataset size) |
| Explainability | SHAP (directional feature impact per campaign) |
| Frontend | React 18, Vite, Tailwind CSS v4, Recharts |

## How It Works

1. **Upload** a CSV with columns: `headline`, `body_text`, `cta_text`, `unique_visitors`, `signatures`, `traffic_source`.
2. **Ingestion** validates, cleans, and normalizes the data. Outliers are flagged.
3. **Feature extraction** computes 49 numeric features across 6 groups: structural, headline, sentiment/emotion, readability, CTA quality, and contextual.
4. **Modeling** adaptively selects the best estimator based on dataset size, trains with cross-validation, and computes SHAP feature importance.
5. **Scoring** predicts conversion rate for new drafts, assigns a letter grade (A–F via z-score), and generates prioritized recommendations.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check + model status |
| `POST` | `/api/analyze` | Upload CSV → full pipeline analysis |
| `POST` | `/api/score-draft` | Score a new campaign draft (requires trained model) |
| `GET` | `/api/sample-data` | Preview the synthetic dataset (JSON) |
| `GET` | `/api/sample-csv` | Download the synthetic dataset (CSV file) |

## Known Limitations

1. **Causation vs. correlation** — The model identifies statistical patterns in your historical data. A recommendation is a hypothesis to test via A/B experimentation, not a guaranteed rule.
2. **Confounding variables** — Traffic quality, email list engagement rates, current events, and seasonal effects all influence conversion independently of copy quality.
3. **Cross-category generalization** — Environment petitions attract different audiences than housing petitions. The model controls for category as a feature, but within-category models would be more accurate given sufficient data.
4. **Temporal drift** — What converted in 2022 may not convert in 2025. The system should be retrained quarterly with fresh campaign data.
5. **In-memory model** — The trained model lives in process memory. Restarting the backend server clears it. A production system would persist to disk or Redis.
6. **Synthetic data** — The included sample dataset is synthetically generated. Model accuracy will improve substantially with real campaign data.

## Modeling Decisions

| Decision | Rationale |
|---|---|
| XGBoost over deep learning | Small dataset (30–200 samples). XGBoost is purpose-built for tabular data. |
| Regression over classification | Conversion rate is continuous — regression preserves "how much better" comparisons. |
| SHAP over `.feature_importances_` | SHAP gives *directional* impact (hurts vs helps), enabling personalized recommendations. |
| Ridge for < 50 samples | L2 regularization prevents overfitting on tiny datasets. |
| KFold cross-validation | Prevents overoptimistic in-sample accuracy — gives an honest reliability estimate. |
| Outlier flagging | Viral campaigns corrupt the model's view of what drives conversion. |

## License

MIT
