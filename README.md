# FoodFirst Customer Churn Prediction

> An end-to-end customer churn analysis project built around a synthetic food-delivery platform, combining restaurant data, synthetic customer behavior, order simulation, SQL feature engineering, exploratory analysis, and XGBoost churn prediction.

---

`Python` · `SQL / PostgreSQL` · `XGBoost` · `SHAP` · `Statistical Simulation`

📄 Full methodology, math, and design decisions → [`METHODOLOGY.md`](./METHODOLOGY.md)

---

## TL;DR

| | |
|---|---|
| **Business problem** | Identify customers likely to stop ordering so retention effort can be prioritized before they churn |
| **Approach** | No real customer-level dataset existed for this problem, so one was built: simulated customers, preferences, and 180 days of orders on top of a real restaurant dataset, then engineered features and trained a classifier |
| **Prediction window** | Features from days 0–90 → predict churn in days 91–180 |
| **Model** | XGBoost (`binary:logistic`) |

| Metric | Score |
|---|---:|
| ROC-AUC | **93.20%** |
| Accuracy | **87.19%** |
| Precision | **84.27%** |
| Recall | **84.86%** |
| F1 Score | **84.56%** |

> ⚠️ **Read before trusting these numbers:** customers, orders, and churn outcomes are all simulated. These metrics show the model recovered patterns *built into the simulation* — they are not a claim about real-world FoodFirst performance. See [Limitations](#limitations).

<!-- INSERT: metric comparison bar chart (ROC-AUC / Accuracy / Precision / Recall / F1) -->

<details>
<summary><b>📑 Table of Contents</b></summary>

- [Pipeline](#pipeline)
- [Why I Built This](#why-i-built-this)
- [Key Business Insights](#key-business-insights)
- [How the Data Was Built](#how-the-data-was-built)
- [Churn Definition](#churn-definition)
- [Feature Engineering (SQL)](#feature-engineering-sql)
- [Modeling](#modeling)
- [Limitations](#limitations)
- [Planned: Power BI Dashboard](#planned-power-bi-dashboard)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Setup](#setup)
- [Key Takeaway](#key-takeaway)

</details>

---

## Pipeline

```text
Restaurant Data
       ↓
Restaurant EDA & Cleaning
       ↓
Restaurant Feature Engineering
       ↓
Synthetic Customer Generation
       ↓
Customer Preferences
       ↓
Order Simulation
       ↓
90-Day Observation  →  90-Day Future Outcome
       ↓
SQL Feature Engineering
       ↓
Churn EDA
       ↓
XGBoost  →  SHAP Interpretation
       ↓
Business Insights
```

<!-- INSERT: pipeline diagram (visual version of the flow above) -->

---

## Why I Built This

Most churn projects start with a ready-made customer-level dataset and jump straight to modeling. I wanted to work through the part that usually happens *before* the model:

> How do you construct realistic customer behavior, define churn without leaking future information, and turn that into predictive features — starting from nothing but restaurant data?

I couldn't find a suitable customer-level food-delivery dataset, so I built the environment myself: restaurant data → simulated customers → simulated orders → engineered features → churn label → model → business interpretation. The goal was never to produce a "real" dataset — it was to exercise the full pipeline end to end, including the parts (data generation, label design) that most portfolio projects skip.

---

## Key Business Insights

<!-- INSERT: SHAP summary plot -->

**1. Recent activity outweighs historical value.** Recency and recent ordering behavior were the strongest churn signals — a customer who has recently gone quiet is a higher-priority retention target than one who simply has lower lifetime spend.

**2. High historical spend ≠ low risk.** A customer with strong past spending isn't automatically safe. Lifetime value alone misses the disengagement signal that recent behavior captures.

**3. Behavioral features beat static ones.** Features describing what a customer actually *does* (order gaps, restaurant/cuisine loyalty, spending trend) outperformed static demographic-style characteristics.

**4. Suggested retention path:**

```text
Customer Behavior → Churn Probability → Risk Segment → Retention Strategy
```

Candidate interventions: personalized offers, free-delivery incentives, re-engagement campaigns, restaurant/cuisine recommendations, loyalty incentives. The right threshold for triggering these isn't a modeling decision alone — it depends on contact cost, expected customer value, cost of a missed churner, and expected retention uplift, and should ultimately be validated with controlled experiments rather than model metrics alone.

---

## How the Data Was Built

The full statistical reasoning (why Poisson, why Gamma, why Bayesian-adjusted ratings, why log-transformed votes) lives in [`METHODOLOGY.md`](./METHODOLOGY.md). Short version:

| Component | Approach | Why |
|---|---|---|
| Order frequency | Poisson-based, rate varies by customer profile | Order counts are non-negative events over a fixed period |
| Spending | Gamma distribution | Spending is positive, continuous, and right-skewed |
| Restaurant choice | Customer-specific **Appeal Score** (cuisine match, price match, rating, preferences) — not raw popularity | A popular restaurant isn't attractive to *every* customer; popularity asks "how popular overall," appeal asks "how attractive to this customer" |
| Restaurant quality | Bayesian-adjusted rating | A 5.0 rating from 5 reviews and a 4.6 from 5,000 reviews aren't equally trustworthy — low-evidence ratings are shrunk toward the global mean |
| Review volume | `log(1 + votes)` | Raw vote counts were heavily right-skewed and would otherwise dominate the appeal score |
| Churn | Time-based (not order-triggered) | Checking churn around individual orders unintentionally gave frequent customers more chances to trigger it, coupling the label to order frequency itself |

<!-- INSERT: customer persona / spending distribution plots -->

---

## Churn Definition

```text
Days 0–90 (Observation)  →  Features only
Days 91–180 (Outcome)    →  Did the customer return?
```

- Customers with **zero activity** in the observation window are excluded entirely — they're a "never engaged" problem, not churn
- `churned = 1` if active in the observation window but did not return in the outcome window

This observation/outcome split avoids right-censoring bias and mirrors how real subscription/usage-based businesses define churn, rather than penalizing naturally slower-paced customers with a single fixed lookback point.

---

## Feature Engineering (SQL)

Built entirely in PostgreSQL using CTEs, aggregating **how behavior was changing**, not just static snapshots:

| Category | Examples |
|---|---|
| Recency | Days since last order |
| Frequency | Orders in recent periods |
| Monetary | Recent vs. historical spending |
| Order behavior | Average order gap, gap volatility (`LAG()` window functions) |
| Restaurant/cuisine behavior | Diversity, loyalty ratios |
| Price behavior | Spend relative to preferred price range |
| Preference matching | Deviation between preferred and actual restaurant type/price |

---

## Modeling

1. Feature preparation → train/test split
2. Baseline XGBoost
3. Manual hyperparameter sweeps (`n_estimators`, `gamma`) to build intuition, then `RandomizedSearchCV` — chosen over `GridSearchCV` for better parameter-space coverage at lower cost
4. Diagnosed and corrected an overfitting pattern via train/test accuracy curves across `n_estimators`
5. Evaluated on ROC-AUC, precision, recall, F1, and accuracy — not accuracy alone, given class imbalance
6. SHAP for interpretation

<!-- INSERT: train/test accuracy curve across n_estimators (overfitting diagnostic) -->
<!-- INSERT: ROC curve -->

---

## Limitations

- **Synthetic everything** — customers, orders, preferences, and churn outcomes are all simulated
- **Simulated target** — the churn label comes from assumptions, not observed outcomes
- **Performance may be optimistic** — the model is learning patterns within a controlled simulation, so metrics aren't comparable to a model trained on real customer data
- **No causal retention evidence** — this predicts churn, it doesn't establish that any specific intervention prevents it
- **Persona/tier assumptions are generative, not derived** — in a real pipeline these would come from clustering on actual transactional data, not be assumed upfront
- **Churn hazard compounded with order frequency more than intended**, compressing the separation between engagement tiers — a real finding about hazard-based simulation design, documented rather than hidden

**With real customer data next, I'd add:** time-based validation, probability calibration, model monitoring, cost-sensitive threshold selection, A/B testing, and retraining based on behavioral drift.

---

## Planned: Power BI Dashboard

Aggregated SQL outputs (segment summaries, churn rate by tier/segment, order trend lines) will feed an interactive dashboard surfacing churn drivers and at-risk segments for stakeholder reporting.

---

## Repository Structure

```text
FoodFirst-Customer-Churn-Prediction/
│
├── 01_cleaned.ipynb                          Data cleaning
├── 02_restaurants_eda.ipynb                  Restaurant EDA
├── 03_restaurants_feature_engineering.ipynb  Restaurant feature engineering
├── 04_Synthetic_Customer_Generation.ipynb    Customer persona simulation
├── 05_order_simulation.ipynb                 Order timeline simulation
├── 06_Preprocessing_and_Model_training.ipynb Feature prep + model training
├── 07_Churn_EDA.ipynb                        Churn exploratory analysis
│
├── restaurants.csv
├── EDA_restaurants.csv
├── customers_dummy.csv
├── order_simulation(1).csv
├── customer_churn_features_clean.csv
│
├── requirements.txt
├── METHODOLOGY.md
└── README.md
```

## Tech Stack

**Data & Analysis** — Python, Pandas, NumPy, Matplotlib, Seaborn, SQL, PostgreSQL
**Machine Learning** — Scikit-learn, XGBoost
**Interpretability** — SHAP
**Statistical Simulation** — Poisson, Gamma, Bayesian rating adjustment

## Setup

```bash
git clone https://github.com/PrikshitSingh230/FoodFirst-Customer-Churn-Prediction.git
cd FoodFirst-Customer-Churn-Prediction
pip install -r requirements.txt
```

Run notebooks `01` → `07` in order to reproduce the full pipeline, from raw restaurant data to a trained, interpreted churn model.

---

## Key Takeaway

This project isn't a claim of a high-performing production churn model. It's a walkthrough of the full analytical process behind a churn problem: turning raw restaurant data into customer behavior, defining a prediction target that doesn't leak the future, engineering behavioral features, validating assumptions along the way, and connecting model output to business decisions.

The biggest lesson: modeling is one part of the problem. The assumptions used to generate and structure the data matter just as much as the algorithm applied afterward.
