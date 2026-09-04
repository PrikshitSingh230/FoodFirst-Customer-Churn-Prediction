# FoodFirst — Customer Churn Prediction

## Overview

An end-to-end churn analytics project for **FoodFirst**, a food delivery platform. Since no real transactional data was available, realistic customer and order behavior was simulated from first principles — persona-driven customer segments, discrete-choice restaurant selection weighted by popularity and price fit, and time-based order generation — then used to engineer features, define a proper time-aware churn label, and train a predictive model.

## Project Goals

- Simulate realistic customer ordering behavior on a real restaurant dataset
- Engineer a statistically sound churn label that avoids look-ahead bias (right-censoring)
- Build customer-level features using SQL aggregation
- Train and tune a churn prediction model (Logistic Regression baseline → XGBoost)
- Present findings via a Power BI dashboard

## Data & Simulation

- **Restaurant data**: real dataset (Zomato-style: rating, votes, cost, cuisine, location, online_order, rest_type), cleaned and enriched with a Bayesian-adjusted rating (shrinks low-vote restaurants toward the global mean) and a log-scaled popularity score to correct for right-skewed vote counts.
- **Customer simulation**: two independent persona dimensions —
  - **Identity/occasion segment** (Student, Office Worker, Family, Couple, Food Explorer) — drives cost preference, group size, cuisine/restaurant-type preference
  - **Engagement tier** (power_user, regular, occasional, trial_only) — drives ordering frequency and churn hazard
- **Order generation**: each customer's order timeline is simulated using gamma-distributed inter-order gaps (centered on their tier's typical pace) and a discrete-choice restaurant selection model weighted by appeal score, cost match, and cuisine/type loyalty vs. exploration.

## Churn Label Definition

Churn is defined using an **observation/outcome window split** to avoid right-censoring bias:
- **Observation window** (day 0–90): used only to compute features
- **Outcome window** (day 90–180): used only to check whether the customer returned
- Customers with zero activity in the observation window are excluded from the modeling population entirely (they represent a separate "never engaged" problem, not churn)
- `churned = 1` if active in the observation window but did not return in the outcome window

This mirrors how real subscription/usage-based businesses define churn, and avoids the common mistake of using a single fixed lookback point that unfairly penalizes naturally slower-paced customers.

## Feature Engineering (SQL)

Customer-level features were built entirely via SQL (PostgreSQL) using CTEs, including:
- RFM basics: total spend, average/median/std order value, recency
- Order gap statistics (mean/std/min/max via `LAG()` window functions)
- 30-day period activity and spend trends (first/middle/last third of observation window)
- Restaurant and cuisine loyalty ratios (share of orders going to top restaurant/cuisine)
- Preference-alignment features (deviation between preferred and actual price/restaurant type)

## Modeling

- Baseline: Logistic Regression
- Primary model: XGBoost (`binary:logistic`, evaluated on PR-AUC given class imbalance)
- Hyperparameter exploration: manual sweeps (`n_estimators`, `gamma`) to build intuition before `RandomizedSearchCV`, chosen over `GridSearchCV` for better coverage of the parameter space at lower computational cost
- Diagnosed and corrected an overfitting pattern via train/test accuracy curves across `n_estimators`

## Planned: Power BI Dashboard

Aggregated SQL outputs (customer segment summaries, churn rate by tier/segment, order trend lines) exported for an interactive dashboard surfacing churn drivers and at-risk customer segments for stakeholder reporting.

## Key Learnings / Honest Caveats

- Personas and engagement tiers were used as *generative* assumptions to produce realistic synthetic behavior; in a real pipeline these would instead be *derived* from clustering/segmentation on raw transactional data, not assumed upfront.
- Cumulative churn hazard compounds with order frequency, which compressed the intended separation between engagement tiers more than initially designed — a useful finding about hazard-based simulation design, documented rather than hidden.
- All data is simulated; findings demonstrate methodology, not real-world FoodFirst behavior.
