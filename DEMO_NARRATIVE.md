# Recruiter Demo Narrative

> **Purpose:** Talking points for walking a recruiter or interviewer through the Petition Effectiveness Analyzer. Ties every feature to a deliberate technical decision.

---

## Opening (30 seconds)

> "I built a tool that helps advocacy organizations write more effective petition campaigns. You upload your campaign history, the system trains an ML model on your data, and then you can paste in a new draft and get a predicted conversion rate plus specific, prioritized recommendations — before you hit publish."

---

## Live Demo Script

### Step 1: Load Data (show Upload tab)

> "I click 'Generate & Load Mock Data' — this sends a 120-row synthetic dataset through the full pipeline. In production, you'd upload your own CSV."

**If asked about the data:**
> "The synthetic generator injects realistic signal — urgency words, named targets, and question headlines all get measurable conversion lifts. This ensures the model has something real to learn from, even in a demo."

### Step 2: Analysis Dashboard

**Point to the summary cards:**
> "120 campaigns analyzed, average conversion rate 12.2%. The model used Random Forest because the dataset is between 50 and 200 rows."

**Point to the feature importance chart:**
> "This is SHAP-based feature importance, not just `.feature_importances_`. The difference matters — SHAP gives directional impact. It tells you this feature *hurts* this specific campaign, not just that it matters globally. That's what enables campaign-level recommendations."

**Point to archetypes:**
> "The system clusters campaigns using KMeans and names them by their dominant traits. 'The Hope Builder' has different conversion patterns than 'The Urgent Alarm.' This helps a campaign manager understand which playbook they're running."

### Step 3: Score a Draft (show Score Draft tab)

> "Now I paste in a new draft. The system extracts the same 49 features, runs it through the trained model, and returns a grade plus recommendations."

**Paste in this example:**
- Headline: "Tell Mayor Chen: Stop the Downtown Demolition Before Friday"
- Body: "Our historic Main Street buildings are scheduled for demolition this Friday. Mayor Chen has approved this plan despite overwhelming community opposition. Your voice matters — 1,200 neighbors have already signed. We need 500 more signatures before the council meeting on Thursday."
- CTA: "Tell Mayor Chen: Save Our Historic Main Street"

> "It scored this as a B — 'Above Average.' The recommendations are sorted by impact. The top one says to add more direct 'you' language. That's not a generic tip — it's based on what *your* historical data says drives conversion."

---

## Anticipated Questions & Answers

### "Why not deep learning?"

> "This is a small-data problem. Organizations typically have 30–200 past campaigns. Deep learning needs 10,000+ samples to generalize on tabular data. XGBoost dominates Kaggle tabular competitions precisely because it handles feature interactions with small datasets. I used Ridge regression as a fallback for datasets under 50 rows — it can't overfit when you have regularization."

### "How do you handle different dataset sizes?"

> "The model selection is adaptive. Under 50 campaigns: Ridge with L2 regularization. 50–200: Random Forest. Over 200: XGBoost with SHAP. The architecture is identical in all cases — only the estimator changes. You can drop in real data without modifying any code."

### "Is this just correlation?"

> "Yes, and I'm explicit about that. Every recommendation card says the model identified a *pattern*, not a *cause*. A recommendation is a hypothesis to A/B test, not a rule to follow blindly. The system also flags confounding variables — traffic source quality, external news events, and seasonal effects all affect conversion independently of copy."

### "What are the 49 features?"

> "Six groups: structural features (word count, paragraph count, media presence), headline features (question marks, imperative verbs, named entities, urgency words), sentiment and emotion (VADER compound score, NRCLex emotion proportions), readability (Flesch-Kincaid grade level), CTA quality (specificity, urgency, collective framing), and contextual features (traffic source encoding, cause category, launch timing)."

### "How do you prevent overfitting?"

> "Three layers: (1) KFold cross-validation during training, so the reported accuracy is honest. (2) Adaptive model selection — simpler models for smaller datasets. (3) Outlier flagging — viral campaigns get flagged so they don't corrupt the model's view of what drives conversion."

### "What would you improve with more time?"

> "Three things: (1) Within-category models — environment petitions attract different audiences than housing petitions. With enough data per category, separate models would be more precise. (2) A/B test integration — the recommendation engine generates hypotheses, but it doesn't yet close the loop by tracking which recommendations actually improved conversion. (3) Multi-worker model sharing — the model is persisted to disk via joblib and auto-restores on startup, but in a multi-worker deployment I'd add a shared model store like Redis."

---

## Key Metrics to Quote

- **49 NLP features** extracted per campaign across 6 feature groups
- **210 automated checks** across 5 test suites, all passing
- **3-tier adaptive model selection** (Ridge / Random Forest / XGBoost)
- **SHAP explainability** for campaign-level personalized recommendations
- **8 recommendation rules** with impact grading (high/medium/low)
- **4 archetype clusters** via KMeans with dominant-trait labeling
