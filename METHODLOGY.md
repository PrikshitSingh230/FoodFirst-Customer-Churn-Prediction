# FoodFirst Customer Churn Prediction, Methodology

Back to [README](./README.md)

This document covers the reasoning, assumptions, math, and modeling decisions behind the project. The goal was never just to train a classifier. It was to build a plausible customer-behavior environment and understand how the assumptions made during data generation shape what the eventual model learns.

<details>
<summary>Table of Contents</summary>

1. [Problem Formulation](#1-problem-formulation)
2. [Why Synthetic Data?](#2-why-synthetic-data)
3. [Restaurant Data Preparation](#3-restaurant-data-preparation)
4. [Restaurant Rating](#4-restaurant-rating)
5. [Bayesian Rating Adjustment](#5-bayesian-rating-adjustment)
6. [Review Volume Transformation](#6-review-volume-transformation)
7. [Why Appeal Score Instead of Popularity?](#7-why-appeal-score-instead-of-popularity)
8. [Appeal Score](#8-appeal-score)
9. [Synthetic Customer Generation](#9-synthetic-customer-generation)
10. [Poisson Distribution for Order Frequency](#10-poisson-distribution-for-order-frequency)
11. [Gamma Distribution for Spending](#11-gamma-distribution-for-spending)
12. [Customer Preferences](#12-customer-preferences)
13. [Order Simulation](#13-order-simulation)
14. [Churn Generation](#14-churn-generation)
15. [Revised Time-Based Churn](#15-revised-time-based-churn)
16. [Temporal Feature Engineering](#16-temporal-feature-engineering)
17. [SQL Feature Engineering](#17-sql-feature-engineering)
18. [Modeling](#18-modeling)
19. [Hyperparameter Tuning](#19-hyperparameter-tuning)
20. [Evaluation](#20-evaluation)
21. [Why SHAP?](#21-why-shap)
22. [Key Analytical Findings](#22-key-analytical-findings)
23. [Why mean_order_gap Looked Backwards](#23-why-mean_order_gap-looked-backwards)
24. [Synthetic Data: The Central Limitation](#24-synthetic-data-the-central-limitation)
25. [Avoiding the Simulation Trap](#25-avoiding-the-simulation-trap)
26. [Production Extension](#26-production-extension)
27. [Reproducibility](#27-reproducibility)
28. [Summary of Key Decisions](#28-summary-of-key-decisions)
29. [Final Perspective](#29-final-perspective)

</details>

---

# 1. Problem Formulation

The problem is framed as a binary customer churn prediction task. For each customer, the model receives behavioral information observed during an initial period and predicts whether the customer will churn during a subsequent period.

```text
                    OBSERVATION                 OUTCOME
                    PERIOD                      PERIOD

                 Days 0-90                    Days 91-180
                     |                            |
             Customer behavior              Future activity
                     |                            |
                     v                            v
             Feature engineering              Churn label
                     |                            |
                     +------------+---------------+
                                  |
                                  v
                          Churn Prediction
```

The key constraint: no information from the outcome period should be available to the model when creating prediction features. This prevents future information from leaking into the prediction problem.

---

# 2. Why Synthetic Data?

The project required customer-level food delivery behavior, but the available restaurant dataset had no customer or order history to build a churn problem on top of.

Rather than switching to a generic churn dataset, I built the missing customer layer myself, which raised a different question: if the customer data doesn't exist, what assumptions are required to create believable customer behavior?

```text
Restaurant Data
      |
Restaurant Characteristics
      |
Customer Profiles
      |
      +-- Ordering Frequency
      +-- Spending Behavior
      +-- Cuisine Preferences
      +-- Price Preferences
      |
Restaurant Selection
      |
Orders
      |
Future Activity
      |
Churn
```

The synthetic dataset isn't meant to reproduce real customer behavior exactly. It's a controlled environment where assumptions can be explicitly defined, tested, and changed.

---

# 3. Restaurant Data Preparation

The restaurant dataset is the foundation of the simulation. Relevant attributes include restaurant rating, review/vote volume, cost, cuisine, restaurant type, and location-related fields. These were cleaned and transformed into features that influence customer preferences and restaurant selection before any customer simulation happened.

Source: [Zomato Restaurants Dataset, Kaggle](https://www.kaggle.com/datasets/rajeshrampure/zomato-dataset)
---

# 4. Restaurant Rating

A raw restaurant rating doesn't capture how much evidence sits behind it. For example:

```text
Restaurant A: Rating 5.0, Reviews 5
Restaurant B: Rating 4.6, Reviews 5,000
```

Treating these ratings the same assumes five reviews carry the same confidence as five thousand. I didn't want to use the raw rating directly for that reason.

---

# 5. Bayesian Rating Adjustment

The restaurant rating was adjusted using a Bayesian-style approach combining the restaurant's observed rating with the overall dataset average:

$$
AdjustedRating = \frac{v}{v+m}R + \frac{m}{v+m}C
$$

where R is the restaurant's observed rating, v is its number of votes, C is the overall average rating, and m is a minimum-review threshold that controls how strongly low-evidence ratings get pulled toward the average.

With few reviews, the rating is pulled toward the overall average. With many reviews, the observed rating dominates. This is meant to reflect that restaurant quality depends on both the rating itself and how much evidence backs it.

![Popularity score vs raw rating](https://raw.githubusercontent.com/PrikshitSingh230/FoodFirst-Customer-Churn-Prediction/main/Images/popularity_score_rate.png)

The scatter shows this in practice. At low ratings, points sit tightly along a line, since most low-rated restaurants also have few reviews and little room for adjustment. At high ratings (4.0 and above), the spread widens noticeably: restaurants with the same raw rating end up with meaningfully different popularity scores depending on how many votes back that rating.

---

# 6. Review Volume Transformation

Review counts were heavily right-skewed. A small number of restaurants had extremely large numbers of reviews compared to the rest, and using raw counts would let those extreme values dominate any scoring function built on top of them.

$$
log(1+votes)
$$

The `+1` lets restaurants with zero votes still pass through the transform. This compresses large values while preserving the relative ordering of review volume. The goal wasn't to remove information from review volume, just to stop a handful of extreme values from dominating everything downstream.

---

# 7. Why Appeal Score Instead of Popularity?

A simpler restaurant-selection mechanism could have used overall popularity. But popularity is a global property: it answers "how often is this restaurant chosen overall," not "how attractive is this restaurant to this particular customer." A customer who prefers inexpensive Indian food shouldn't have the same restaurant-selection probabilities as one who prefers expensive Japanese food. That gap is why I introduced a customer-specific Appeal Score instead.

---

# 8. Appeal Score

The Appeal Score combines restaurant characteristics with an individual customer's preferences:

$$
Appeal(c,r) = w_1 \cdot CuisineMatch + w_2 \cdot PriceMatch + w_3 \cdot Rating + w_4 \cdot OtherFactors
$$

where c is the customer, r is the restaurant, and the w terms are component weights defined in the customer/order simulation notebook. The resulting score is converted into probabilities used for restaurant selection.

```text
Restaurant popularity  ->  same ordering probability for everyone

Customer preferences + Restaurant characteristics
        -> Customer-specific Appeal
        -> Restaurant choice
```

This is what makes restaurant diversity, loyalty, cuisine preference, and price sensitivity meaningful as downstream features, instead of every customer converging on the same popular restaurants.

![Appeal score distribution vs popularity score distribution](https://raw.githubusercontent.com/PrikshitSingh230/FoodFirst-Customer-Churn-Prediction/main/Images/popularity_score_vs_appeal_score.png)

This is the clearest evidence that the two scores aren't measuring the same thing. Popularity score is tightly concentrated around 0.2 to 0.3 for almost every restaurant. Appeal score is wider and flatter, spreading meaningfully from 0.4 up past 0.8. If appeal score were just a rescaled version of popularity, the two histograms would have the same shape. They don't.

![Appeal score vs rating](https://raw.githubusercontent.com/PrikshitSingh230/FoodFirst-Customer-Churn-Prediction/main/Images/appeal_score_rating.png)

One thing worth flagging honestly: there's a visible floor in this scatter, a cluster of restaurants sitting at roughly 0.34 appeal score across a range of ratings from about 3.3 to 4.3. That's most likely a minimum-value clip somewhere in the appeal score formula rather than a real behavioral pattern, and it's worth checking against the simulation notebook rather than reading into it further.

---

# 9. Synthetic Customer Generation

Customers were generated with different behavioral characteristics rather than treating every customer identically. The simulation introduces variation in ordering frequency, spending behavior, cuisine preference, price preference, restaurant preference, and exploration/loyalty tendency. The goal was a population with genuinely different types of customers, not a set of independent random draws that happen to look different.

---

# 10. Poisson Distribution for Order Frequency

The number of orders a customer places in a fixed period is a count variable, so I used a Poisson-based approach:

$$
N \sim Poisson(\lambda)
$$

where N is the number of orders and lambda is the expected order rate. Different customer profiles get different values of lambda, from lower for infrequent customers to higher for frequent ones. This keeps order counts non-negative and realistic while still producing meaningful variation across the population. A plain random integer generator wouldn't have modeled event frequency the same way.

---

# 11. Gamma Distribution for Spending

Spending-related variables were modeled with a Gamma distribution:

$$
Spend \sim Gamma(k,\theta)
$$

Gamma was chosen because spending can't be negative, is continuous, and is typically right-skewed with a small number of high spenders. A normal distribution would allow negative values and wouldn't reproduce that skew naturally. Parameters were set based on the intended customer behavior and are documented in the simulation notebook.

---

# 12. Customer Preferences

Customers weren't given identical preferences. Preference variables (preferred cuisines, price range, ordering intensity, spending tendency, exploration vs. loyalty) shape the probability that a customer interacts with a given restaurant or cuisine, which ties a customer's latent preferences to their observed orders.

---

# 13. Order Simulation

Orders combine customer behavior with restaurant characteristics:

```text
Customer (order frequency, spending tendency, preferences)
        -> Appeal Scores
        -> Restaurant selection
        -> Order
```

Each simulated order carries information about both sides of the interaction, customer and restaurant, which lets downstream features capture not just how often customers order but what they order and how their preferences shift over time.

---

# 14. Churn Generation

The first version of the simulation checked churn around customer ordering events. This created an unintended problem: customers who ordered frequently had more opportunities to trigger the churn mechanism than infrequent customers did. That meant the mechanics of the label-generation process could introduce a relationship between frequency and churn on their own, independent of any real behavioral signal.

---

# 15. Revised Time-Based Churn

I changed the mechanism to make churn a time-based outcome instead of something evaluated only when an order happens:

```text
Days 0-90: observation, feature generation, churn prediction
Days 91-180: future behavior, churn outcome
```

This gives a cleaner question to answer: what can we learn about future churn from behavior observed before the prediction point? It also stops order frequency from influencing churn simply because frequent customers get more chances to trigger the label.

---

# 16. Temporal Feature Engineering

All customer features were calculated using only the observation period. This included recency (days between the prediction date and the last order), frequency (orders in historical windows), monetary behavior (spending in historical windows), order-gap behavior (average or characteristic time between orders), restaurant behavior (diversity, loyalty, repeat ordering), cuisine behavior (diversity, loyalty, preference concentration), and price behavior (spending relative to restaurant price level). The goal was to represent behavior, not just identity.

---

# 17. SQL Feature Engineering

Customer-level features were aggregated using SQL/PostgreSQL, intentionally kept separate from the modeling stage:

```text
Raw Orders -> SQL Aggregation -> Customer-level Feature Table -> EDA -> Modeling
```

This mirrors a common analytics workflow where raw transactional data becomes an analytical dataset before it ever reaches a model.

---

# 18. Modeling

The final predictive model uses XGBoost, chosen because the dataset has nonlinear relationships, mixed feature scales, and behavioral interactions, particularly between recency, frequency, spending, and churn, that a tree-based boosting model can capture without every relationship being specified by hand.

Workflow: feature preparation, train/test split, baseline model, hyperparameter search, final model, evaluation, SHAP interpretation.

---

# 19. Hyperparameter Tuning

```text
Baseline -> Randomized Search -> Promising Parameter Region -> Grid Search -> Final Configuration
```

The idea was to explore the parameter space broadly first, then focus a grid search around the region that looked promising. Final parameters are documented in the model-training notebook.

---

# 20. Evaluation

Multiple metrics were used since churn prediction isn't well described by accuracy alone.

| Metric | Result |
|---|---:|
| ROC-AUC | 93.20% |
| Accuracy | 87.19% |
| Precision | 84.27% |
| Recall | 84.86% |
| F1 | 84.56% |

![ROC Curve](https://raw.githubusercontent.com/PrikshitSingh230/FoodFirst-Customer-Churn-Prediction/main/Images/ROC_AUC%20graph.png)

**Baseline vs. tuned model**

<table>
<tr>
<td><img src="https://github.com/PrikshitSingh230/FoodFirst-Customer-Churn-Prediction/blob/main/Images/confusionmatrix_baseline.png" alt="Baseline confusion matrix" width="400"/></td>
<td><img src="https://raw.githubusercontent.com/PrikshitSingh230/FoodFirst-Customer-Churn-Prediction/main/Images/confusionmatrix.png" alt="Tuned confusion matrix" width="400"/></td>
</tr>
</table>

Comparing the two directly:

| | Baseline | Tuned |
|---|---:|---:|
| False positives | 119 | 182 |
| False negatives | 243 | 174 |
| Churners caught | 906 | 975 |

Tuning moved the model toward catching more real churners at the cost of more false alarms. That's a deliberate trade, not just an optimization artifact: in this problem a missed churner is a lost customer, while a false alarm is an unnecessary retention offer, so shifting error toward false positives is usually the right call.

These results reflect performance on the simulated dataset and shouldn't be read as expected production performance, since both customer behavior and churn outcomes were generated synthetically.

---

# 21. Why SHAP?

A churn model is only useful if its predictions can be understood well enough to act on. SHAP was used to answer which features influence predictions most, which push churn probability up or down, and why a specific customer gets flagged as high risk. That turns the question from "who will churn" into "who will churn, and what behavior is driving it."

![SHAP summary plot](https://raw.githubusercontent.com/PrikshitSingh230/FoodFirst-Customer-Churn-Prediction/main/Images/SHAP_value_output.png)

![SHAP feature importance bar chart](https://raw.githubusercontent.com/PrikshitSingh230/FoodFirst-Customer-Churn-Prediction/main/Images/shap_barplot.png)

`recency_days` dominates both plots. In the beeswarm, high recency (red, meaning a long time since the last order) pushes SHAP values sharply positive, toward churn. The bar chart shows the same thing more bluntly: recency's mean absolute SHAP value is roughly five times the next feature, `mean_order_gap`.

Two features appear here, `age` and `income`, that aren't described anywhere in the feature engineering sections above. They came from customer profile generation rather than the SQL aggregation step, and are noted here for completeness since a plot listing undocumented features is a fair thing for a reader to notice.

---

# 22. Key Analytical Findings

Recent activity was strongly associated with churn risk. Historical spending alone wasn't enough to identify who would stay active. Recency, frequency, and ordering behavior gave more useful signal than static customer attributes. And the churn-generation issue in Section 14 showed that a model can learn patterns created by the data-generation process itself, which was one of the more important lessons of the project.

---

# 23. Why mean_order_gap Looked Backwards

One feature didn't behave the way intuition suggested it should. `mean_order_gap`, the average number of days between a customer's orders, was *smaller* for churned customers (median around 5 days) than for retained customers (median around 10 days). That's backwards from the usual assumption that churn means declining activity and growing gaps between orders.

The first hypothesis was that this came from the churn-hazard-compounding issue described in Section 14 and 15, frequent orderers accumulating more chances to trigger churn even after the mechanism was made time-based. Checking that directly against the feature table didn't support it: customers with more total orders in the observation window actually had a slightly *lower* churn rate overall (34% in the top order-frequency quintile vs. 53% in the bottom quintile), which is the opposite of what a compounding-hazard artifact would produce.

The real explanation showed up once the observation window was split into thirds. Looking only at the customers with the smallest average gaps (the group that should, by the old intuition, be the safest):

| | Days 0-30 | Days 31-60 | Days 61-90 | Recency |
|---|---:|---:|---:|---:|
| Retained | 15.3 orders | 15.7 orders | 15.8 orders | 1.9 days |
| Churned | 12.5 orders | 5.2 orders | 1.6 orders | 58.0 days |

Both groups have a small average gap, but for opposite reasons. The retained customers here are consistently active across the entire window. The churned customers front-loaded their activity, ordered heavily in the first 30 days, and then dropped off almost completely, going 58 days without an order by the end of the window. `mean_order_gap` averages over the whole 90 days, so it can't tell a steadily active customer from one who burned through a cluster of orders early and then vanished. Both produce a similarly small number.

This pattern holds at the population level too, not just within this one quintile:

![Order trajectory by churn status](https://raw.githubusercontent.com/PrikshitSingh230/FoodFirst-Customer-Churn-Prediction/main/Images/order_trajectory.png)

![mean_order_gap box plot by churn status](https://github.com/PrikshitSingh230/FoodFirst-Customer-Churn-Prediction/blob/main/Images/churn_retained_by_mean_order_gap.png)

Practically, this means `mean_order_gap` on its own is a misleading feature. A trend-based feature, something like the ratio of `orders_last_30d` to `orders_first_30d`, or a slope across the three 30-day buckets, would capture the actual decline pattern that the flat average currently hides. Those raw ingredients (`orders_first_30d`, `orders_middle_30d`, `orders_last_30d`) already exist in the feature table; the trend feature itself just hasn't been built yet.

---

# 24. Synthetic Data: The Central Limitation

The most important limitation is that the customer-level data is simulated:

```text
Assumptions -> Synthetic behavior -> Synthetic churn -> Model
```

The model can recover patterns that exist inside this simulated environment, but that doesn't guarantee the same relationships hold in real food-delivery data. A high AUC here doesn't prove that recency would produce a 93% ROC-AUC on a real platform. It shows the pipeline can find patterns that were generated by the simulation.

---

# 25. Avoiding the Simulation Trap

Since the data is synthetic, model performance shouldn't be the only measure of project quality. More useful questions: were the assumptions reasonable, were unintended relationships caught, was future information kept out of the features, were the distributions appropriate for what they were modeling, was customer heterogeneity actually introduced, can the model's learned relationships be explained, and can the workflow be adapted to real customer data. That's why this project puts real weight on data generation and validation before modeling, rather than treating the model as the whole story.

---

# 26. Production Extension

With real customer-level data, several parts of the workflow would change. Validation would move to time-based splits instead of random ones. Predicted probabilities would need calibration checks against observed churn rates. The intervention threshold would be set by business cost, not by maximizing F1 or accuracy. Retention prioritization would use something like:

$$
ExpectedValue = P(Retention) \times CustomerValue - InterventionCost
$$

And critically, a churn model predicts risk, it doesn't prove an intervention prevents churn. Retention strategies would need to be tested through controlled experiments, not assumed to work because the model flagged the right customers.

---

# 27. Reproducibility

```text
01_cleaned.ipynb
   -> 02_restaurants_eda.ipynb
   -> 03_restaurants_feature_engineering.ipynb
   -> 04_Synthetic_Customer_Generation.ipynb
   -> 05_order_simulation.ipynb
   -> 06_Preprocessing_and_Model_training.ipynb
   -> 07_Churn_EDA.ipynb
```

Each stage builds on the output of the one before it. The exact implementation, distributions, parameters, and feature calculations can be inspected directly in the corresponding notebook.

---

# 28. Summary of Key Decisions

| Problem | Decision | Reason |
|---|---|---|
| No customer-level dataset | Synthetic customer/order generation | Allows a complete end-to-end problem, not just a modeling exercise |
| Restaurant ratings | Bayesian-adjusted rating | Accounts for how much evidence backs a rating |
| Highly skewed review counts | `log(1 + votes)` | Reduces the influence of extreme values |
| Restaurant selection | Customer-specific Appeal Score | Models individual preference, not global popularity |
| Order frequency | Poisson-based simulation | Appropriate for event counts |
| Spending | Gamma-based simulation | Positive and right-skewed |
| Churn generation | Time-based mechanism | Avoids order-frequency bias in the label itself |
| Feature window | 90-day observation | Prevents future-information leakage |
| Prediction window | Following 90 days | Defines a genuine future outcome |
| Model | XGBoost | Captures nonlinear behavioral relationships |
| Interpretation | SHAP | Explains individual and global model behavior |

---

# 29. Final Perspective

The central lesson here was that the model is only one part of a data science problem. When you're working with synthetic data, the assumptions used to generate it can end up determining what the model eventually learns, and a few places in this project (the churn mechanism, `mean_order_gap`) made that very literal.

The most valuable part of the project wasn't hitting a particular AUC. It was the loop of making assumptions, generating data, checking what came out of it, finding something that didn't add up, tracing it back to its cause, and only then trusting the number. With real customer data, the next challenge would be figuring out which of these simulated relationships actually survive contact with the real world.
