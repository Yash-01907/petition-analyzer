# How I Actually Approached This

This file isn't formal documentation. It's an honest account of how I thought through this problem — including the conversations, the dead ends, and the decisions I made along the way.

---

## Getting the Problem

The brief came from OpenPAWS — an org working at the intersection of AI and animal advocacy. The problem statement was clear on what to build but deliberately open on the details: take past petition campaigns, figure out what makes some convert better than others, and give a non-technical campaigns manager something they can actually act on.

The first thing I noticed was that the problem statement said "signature count" — not conversion rate. That felt like a trap worth thinking about.

---

## The First Real Decision: What Are We Actually Predicting?

Before touching any code, I spent time just thinking about what "effectiveness" means here. Raw signature count is the obvious answer. It's also wrong.

A petition with 500 signatures from 600 visitors is dramatically more effective than one with 2,000 signatures from 80,000 visitors. If I trained a model on raw signature count, it would mostly learn which campaigns had more traffic — not which campaigns had better copy.

So the real target had to be:

```
conversion_rate = (signatures / unique_visitors) × 100
```

That one shift changed the entire pipeline. It meant I needed visitor data in the input. It meant I had to think about traffic source as a confounding variable — because an email sent to existing supporters converts at 15–25% regardless of how the headline is written, while cold social traffic converts at 1–4%. If I didn't control for that, the model would just learn "email good, social bad" and give useless recommendations.

---

## The Dataset Size Problem

Here's something the problem statement didn't tell me: how many past campaigns would this tool actually have to work with?

This matters a lot technically. A dataset of 30 campaigns needs a completely different model than a dataset of 500. I could have just assumed and moved forward — but I emailed Divyaansh directly to ask:

> *"Could you share an estimate of the number of past petitions/campaigns we expect to have available? This will help determine the most appropriate modeling strategy. Since the dataset size influences whether we use a simpler regularized model or move toward more complex ensemble methods, having a rough range would allow me to proceed more confidently."*

I didn't wait for a reply before building. But I used the uncertainty itself as a design constraint — I built the system to adapt based on whatever data size it encounters:

- **Under 50 campaigns** → Ridge Regression. Simple, regularized, won't overfit to noise.
- **50–200 campaigns** → Random Forest. Handles feature interactions, robust to small-n.
- **200+ campaigns** → XGBoost. Full power, best for tabular data at scale.

The model selection happens automatically at runtime. Plug in 40 campaigns or 400 — the system picks the right approach without any code changes. This felt like the honest engineering answer to an open question rather than just picking one model and hoping the data cooperates.

---

## Thinking Through the Product Side

The problem statement asked me to think about who this is for and how it would work in the real world. That question shaped a lot of the technical decisions.

The person using this tool isn't a data scientist. She's a campaigns manager who runs 10–40 petitions a year, lives in Google Docs and email, and has a gut feeling that emotional headlines work better but can't prove it. What she needs isn't model accuracy — it's a recommendation she can act on in the next 20 minutes.

That realization pushed me toward two things: explicit interpretable features (instead of embeddings) and a recommendation engine that produces rewrite examples, not just scores.

There's also a copywriter use case — someone drafting a new petition who wants to paste their headline and CTA before publishing and get instant feedback. That's where the live draft scorer comes in. The campaigns manager uses it post-analysis; the copywriter uses it at draft time. One tool, two workflows.

---

## Why I Chose Explicit Features Over Embeddings

When I first mapped out the approach, the "modern" answer would have been to pass the petition text through a language model and use the embeddings as features. BERT, sentence transformers — there's a real case for them.

I decided against it, and not because they're hard to implement.

The problem is that embeddings produce features no human can interpret or change. If the model says a petition will convert poorly because "embedding dimension 312 is low" — that tells the campaigns manager exactly nothing. They can't edit dimension 312.

The whole point of this tool is that someone should be able to read the recommendations and immediately know what to rewrite. That only works if every feature maps to something a person can actually change:

- Reading grade level → rewrite complex sentences
- Whether the CTA names a specific official → add their name
- Urgency language in headline → add a deadline
- Second person density → use more "you/your"

So I built 30 explicit, interpretable features across six categories — structural, headline type, sentiment, emotion, readability, CTA quality — and accepted that I'd miss some nuance that embeddings might catch. For this use case, that's the right tradeoff.

---

## The Recommendation Engine Was the Hardest Part

Getting the model to predict conversion rates was the easier half. The harder problem was: how do you turn a SHAP value into something a campaigns manager will actually read and use?

My first instinct produced output that was too technical. Things like "your `body_reading_grade` value of 11.4 exceeds the dataset mean of 8.2" — accurate, but useless. Nobody is rewriting their petition body because of a number they don't understand.

I rebuilt the recommendation layer as a rule system with human language hardcoded in. Each rule checks a feature threshold and fires a specific, plain-English recommendation with a concrete rewrite example:

- If the headline has no imperative verb → *"Your headline reads as a statement. Try starting with Tell, Stop, or Demand — for example: 'Tell Mayor Chen: Stop the Dam Before Friday's Vote'"*
- If the CTA is generic → *"You're using a generic call to action. Name the decision-maker and the specific outcome: 'Tell the Planning Board: Reject the Riverside Development'"*

The example rewrites were non-negotiable. Without them, the recommendations are suggestions. With them, they're something a campaigns manager can directly copy-edit into their draft.

---

## Why SHAP Instead of Just Feature Importances

I could have used `.feature_importances_` from Random Forest and stopped there. It's simpler, fast, and tells you what matters globally across all campaigns.

But global importance doesn't help with one specific draft. SHAP gives local explanations — for this exact piece of text, which features are dragging down the predicted conversion and by how much. That's what powers the personalized recommendations. Not "reading level matters in general" but "your specific campaign's Grade 11 reading level is estimated to be costing it about 2 percentage points against your historical average."

---

## Since I Had No Real Data

The problem statement didn't come with a dataset, which is realistic — you'd need an organization's actual campaign history to use this tool. So I built a synthetic data generator that creates 120 realistic fake campaigns.

The important thing is that the synthetic data has real signal injected into it. Imperative headlines convert ~3.5 percentage points higher. Urgency language adds ~4 points. Generic CTAs subtract ~2 points. These lifts are grounded in actual advocacy benchmarks, not random numbers.

This means when you test the tool with sample data, the model learns something meaningful, SHAP outputs make intuitive sense, and scoring a strong draft vs. a weak draft produces noticeably different grades. It behaves like it would with real campaign data.

When real data comes in, you replace the CSV. Nothing else changes.

---

## What I Know This Doesn't Do Well

I'd rather say this upfront:

**It finds correlations, not causes.** A headline structure that worked 18 months ago might have worked because of who was on the email list that week, not the headline itself. Every recommendation is a hypothesis worth testing — not a guarantee.

**Viral campaigns will skew the model.** If a petition gets picked up by national media or a known influencer shares it, the conversion rate will spike for reasons completely unrelated to copy quality. The ingestion pipeline flags statistical outliers, but the user has to decide whether to exclude them before training.

**Cross-category generalization is imperfect.** An environment petition and a housing petition attract different audiences with different sensitivities. The model encodes cause category as a feature, which helps — but a model trained mostly on environment campaigns will give weaker guidance on a healthcare campaign until it sees more of that data.

---

*— Yash*