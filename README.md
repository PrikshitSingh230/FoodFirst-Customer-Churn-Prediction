# FoodFirst Customer Churn Prediction

> An end-to-end customer churn analysis project built around a synthetic food-delivery platform, combining restaurant data, synthetic customer behavior, order simulation, SQL feature engineering, exploratory analysis, and XGBoost churn prediction.

---

`Python` · `SQL / PostgreSQL` · `XGBoost` · `SHAP` · `Statistical Simulation`

📄 Full methodology, math, and design decisions → [`METHODOLOGY.md`](./METHODOLOGY.md)

---

## Overview

No customer-level dataset existed for this problem, so one was built from scratch: simulated customers, preferences, and 180 days of orders layered on top of a real restaurant dataset, then engineered features and trained a classifier to flag customers who are about to stop ordering.

Prediction setup: features are calculated from days 0-90 (observation window), and used to predict churn in days 91-180 (outcome window). Model: XGBoost (`binary:logistic`).

| Metric | Score |
|---|---:|
| ROC-AUC | 93.20% |
| Accuracy | 87.19% |
| Precision | 84.27% |
| Recall | 84.86% |
| F1 Score | 84.56% |

![ROC Curve](https://raw.githubusercontent.com/PrikshitSingh230/FoodFirst-Customer-Churn-Prediction/main/Images/ROC_AUC%20graph.png)

These numbers come from a simulated environment: customers, orders, and churn outcomes are all synthetic. They show the model recovered patterns that exist inside the simulation, not that it would perform this well on real FoodFirst data. More on this under Limitations.

<details>
<summary>Table of Contents</summary>

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
       |
Restaurant EDA & Cleaning
       |
Restaurant Feature Engineering
       |
Synthetic Customer Generation
       |
Customer Preferences
       |
Order Simulation
       |
90-Day Observation  ->  90-Day Future Outcome
       |
SQL Feature Engineering
       |
Churn EDA
       |
XGBoost  ->  SHAP Interpretation
       |
Business Insights
```

---

## Why I Built This

Most churn projects start with a ready-made customer-level dataset and go straight to modeling. I wanted to work through the part that usually happens before the model: how do you construct realistic customer behavior, define churn without leaking future information, and turn that into predictive features, starting from nothing but restaurant data?

There was no suitable customer-level food delivery dataset available, so I built the environment myself. Restaurant data, simulated customers, simulated orders, engineered features, a churn label, a model, and finally a business interpretation. The point was never to produce a "real" dataset. It was to work through the full pipeline end to end, including the parts most portfolio projects skip: data generation and label design.

---

## Key Business Insights

### 1. Customers who front-load their activity are the real risk, not customers who order less overall

![Order trajectory by churn status](https://raw.githubusercontent.com/PrikshitSingh230/FoodFirst-Customer-Churn-Prediction/main/Images/order_trajectory.png)

Retained customers order at a steady pace across the whole 90-day window, averaging around 4 orders every 30 days from start to finish. Churned customers show the opposite shape. They start out more active than retained customers (6.1 orders in the first 30 days), then drop to 2.9, then to 1.0 by the final 30 days. The customers who looked like the best customers early on turned out to be the ones about to leave.

This also explains a feature that looked wrong at first. `mean_order_gap`, the average time between a customer's orders, was smaller for churned customers than for retained ones, which is backwards from what you'd expect if churn meant "ordering less." It turns out `mean_order_gap` was averaging over the whole window and couldn't tell a customer who was consistently active from a customer who burned through a lot of orders early and then vanished. Both produce a small average gap. The full breakdown, with the numbers that confirm it, is in [METHODOLOGY.md, "Why mean_order_gap Looked Backwards"](./METHODOLOGY.md#23-why-mean_order_gap-looked-backwards).

### 2. Recency is still the single strongest signal

![SHAP feature importance](https://raw.githubusercontent.com/PrikshitSingh230/FoodFirst-Customer-Churn-Prediction/main/Images/shap_barplot.png)

`recency_days`, days since the last order, dominates the SHAP importance ranking by a wide margin, roughly five times the impact of the next feature. This matches intuition (a customer who has not ordered in a while is a better churn signal than any single historical trait), but it also gives a clear priority order for a real deployment: recency-based alerts alone would catch most of the signal this model relies on.

### 3. High historical spending does not mean a customer is safe

A customer with strong past spending isn't automatically low risk. `avg_order_value` looks almost identical between retained and churned customers, and `total_spend` distributions overlap heavily. What predicts churn is a customer's *recent* trajectory, not their lifetime value.

### 4. Tuning traded some false alarms for far fewer missed churners

| | Baseline | Tuned |
|---|---:|---:|
| False positives (predicted churn, stayed) | 119 | 182 |
| False negatives (missed actual churn) | 243 | 174 |

Hyperparameter tuning caught 69 more real churners (975 vs. 906) at the cost of 63 more false alarms. In a churn problem this is usually the right trade: a missed churner is a lost customer, a false alarm is just an unnecessary retention offer. Full confusion matrices for both models are in Methodology.

### 5. Luxury-tier customers show noticeably lower churn (worth confirming before treating as a finding)

<!-- PASTE IMAGE LINK HERE: cropped price-tier panel from churn_categorical.png -->

Customers whose preferred price tier is Luxury retain at roughly 75%, well above Budget, Mid-range, and Premium, which all sit around 58-61%. This is a real gap in the data, but price tier was assigned as part of persona generation alongside other traits, so this may partly reflect how personas were designed rather than an emergent pattern. Worth treating as a lead to investigate further, not a confirmed insight.

### Suggested retention path

```text
Customer Behavior -> Churn Probability -> Risk Segment -> Retention Strategy
```

Candidate interventions: personalized offers, free-delivery incentives, re-engagement campaigns, restaurant or cuisine recommendations, loyalty incentives. The right threshold for triggering these depends on contact cost, expected customer value, the cost of a missed churner, and expected retention uplift. It should ultimately be set with controlled experiments, not model metrics alone.

---

## How the Data Was Built

Full statistical reasoning (why Poisson, why Gamma, why a Bayesian-adjusted rating, why log-transformed votes) is in [METHODOLOGY.md](./METHODOLOGY.md). Short version:

| Component | Approach | Why |
|---|---|---|
| Order frequency | Poisson-based, rate varies by customer profile | Order counts are non-negative events over a fixed period |
| Spending | Gamma distribution | Spending is positive, continuous, and right-skewed |
| Restaurant choice | Customer-specific Appeal Score (cuisine match, price match, rating, preferences), not raw popularity | A popular restaurant isn't attractive to every customer. Popularity asks how popular overall, appeal asks how attractive to this customer |
| Restaurant quality | Bayesian-adjusted rating | A 5.0 rating from 5 reviews and a 4.6 from 5,000 reviews aren't equally trustworthy, so low-evidence ratings are pulled toward the global mean |
| Review volume | `log(1 + votes)` | Raw vote counts were heavily right-skewed and would otherwise dominate the appeal score |
| Churn | Time-based, not order-triggered | Checking churn around individual orders gave frequent customers more chances to trigger it, which coupled the label to order frequency itself |

---

## Churn Definition

```text
Days 0-90 (Observation)  ->  Features only
Days 91-180 (Outcome)    ->  Did the customer return?
```

Customers with zero activity in the observation window are excluded entirely. They're a "never engaged" problem, not churn. `churned = 1` if a customer was active in the observation window but did not return in the outcome window.

This split avoids right-censoring bias and mirrors how real subscription or usage-based businesses define churn, rather than penalizing naturally slower-paced customers with a single fixed lookback point.

---

## Feature Engineering (SQL)

Built in PostgreSQL using CTEs, aggregating how behavior was changing over the window, not just static snapshots:

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

1. Feature preparation, train/test split
2. Baseline XGBoost
3. Manual hyperparameter sweeps (`n_estimators`, `gamma`) to build intuition, then `RandomizedSearchCV`, chosen over `GridSearchCV` for better parameter-space coverage at a lower cost
4. Diagnosed and corrected an overfitting pattern using train/test accuracy curves across `n_estimators`
5. Evaluated on ROC-AUC, precision, recall, F1, and accuracy rather than accuracy alone, given the class imbalance
6. SHAP for interpretation

---

## Limitations

- Synthetic everything: customers, orders, preferences, and churn outcomes are all simulated
- The churn label comes from assumptions, not observed outcomes
- Performance may be optimistic, since the model is learning patterns inside a controlled simulation and isn't directly comparable to a model trained on real customer data
- No causal evidence for retention: this predicts churn, it doesn't establish that any specific intervention prevents it
- Persona and engagement-tier assumptions are generative, not derived. In a real pipeline these would come from clustering on actual transactional data, not be assumed upfront
- Churn hazard compounded with order frequency more than intended during simulation design, which compressed the separation between engagement tiers. This is a real finding about the simulation's own mechanics, documented rather than hidden
- One engineered feature, `std_group_size`, turned out to be constant across the dataset and had no correlation with anything, including churn. It wasn't dropped from the pipeline but also contributed nothing to the model

With real customer data, next steps would include time-based validation, probability calibration, model monitoring, cost-sensitive threshold selection, A/B testing, and retraining based on behavioral drift.

---

## Planned: Power BI Dashboard

Aggregated SQL outputs (segment summaries, churn rate by tier and segment, order trend lines) will feed an interactive dashboard surfacing churn drivers and at-risk segments for stakeholder reporting.

---

## Repository Structure

```text
FoodFirst-Customer-Churn-Prediction/
|
|-- 01_cleaned.ipynb                          Data cleaning
|-- 02_restaurants_eda.ipynb                  Restaurant EDA
|-- 03_restaurants_feature_engineering.ipynb  Restaurant feature engineering
|-- 04_Synthetic_Customer_Generation.ipynb    Customer persona simulation
|-- 05_order_simulation.ipynb                 Order timeline simulation
|-- 06_Preprocessing_and_Model_training.ipynb Feature prep and model training
|-- 07_Churn_EDA.ipynb                        Churn exploratory analysis
|
|-- data/processed/customer_churn_features_clean.csv
|-- Images/                                   Charts referenced in this README and Methodology
|
|-- requirements.txt
|-- METHODOLOGY.md
|-- README.md
```

## Tech Stack

Data & Analysis: Python, Pandas, NumPy, Matplotlib, Seaborn, SQL, PostgreSQL
Machine Learning: Scikit-learn, XGBoost
Interpretability: SHAP
Statistical Simulation: Poisson, Gamma, Bayesian rating adjustment

## Setup

```bash
git clone https://github.com/PrikshitSingh230/FoodFirst-Customer-Churn-Prediction.git
cd FoodFirst-Customer-Churn-Prediction
pip install -r requirements.txt
```

Run notebooks `01` through `07` in order to reproduce the full pipeline, from raw restaurant data to a trained, interpreted churn model.

---

## Key Takeaway

This isn't a claim of a production-ready churn model. It's a walkthrough of the full analytical process behind a churn problem: turning raw restaurant data into customer behavior, defining a prediction target that doesn't leak the future, engineering behavioral features, catching where those features were misleading, and connecting the output back to business decisions.

The biggest lesson was that modeling is one part of the problem. The assumptions used to generate and structure the data mattered just as much as the algorithm applied afterward, and in a few places (the churn mechanism, `mean_order_gap`) those assumptions showed up directly in the results in ways that took real digging to explain.


