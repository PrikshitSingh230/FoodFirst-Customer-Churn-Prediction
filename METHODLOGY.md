# FoodFirst Customer Churn Prediction — Methodology

This document describes the reasoning, assumptions, mathematical choices, and modeling decisions behind the FoodFirst customer churn prediction project.

The main objective was not simply to train a classifier, but to construct a plausible customer-behavior environment and understand how assumptions made during data generation affect the eventual churn model.

---

# 1. Problem Formulation

The problem is framed as a **binary customer churn prediction task**.

For each customer, the model receives behavioral information observed during an initial period and predicts whether the customer will churn during a subsequent period.

### Prediction framework

```text
                    OBSERVATION                 OUTCOME
                    PERIOD                      PERIOD

                 Days 0–90                    Days 91–180
                     │                            │
                     │                            │
             Customer behavior              Future activity
                     │                            │
                     ▼                            ▼
             Feature engineering              Churn label
                     │                            │
                     └───────────┬────────────────┘
                                 │
                                 ▼
                         Churn Prediction
```

The key constraint is:

> **No information from the outcome period should be available to the model when creating prediction features.**

This prevents future information from leaking into the prediction problem.

---

# 2. Why Synthetic Data?

The project required customer-level food-delivery behavior, but the available restaurant dataset did not contain the customer/order history needed to construct a churn problem.

Rather than switching to a generic churn dataset, I decided to build the missing customer layer myself.

This introduced a different challenge:

> **If the customer data does not exist, what assumptions are required to create believable customer behavior?**

The synthetic environment was therefore designed around several components:

```text
Restaurant Data
      │
      ▼
Restaurant Characteristics
      │
      ▼
Customer Profiles
      │
      ├── Ordering Frequency
      ├── Spending Behavior
      ├── Cuisine Preferences
      └── Price Preferences
      │
      ▼
Restaurant Selection
      │
      ▼
Orders
      │
      ▼
Future Activity
      │
      ▼
Churn
```

The synthetic dataset is not intended to reproduce real customer behavior perfectly.

Instead, it provides a controlled environment in which assumptions can be explicitly defined, tested, and modified.

---

# 3. Restaurant Data Preparation

The restaurant dataset forms the foundation of the simulation.

Relevant restaurant attributes include:

* Restaurant rating
* Review/vote volume
* Cost information
* Cuisine
* Restaurant type
* Location-related attributes

Before using these attributes in customer simulation, they were cleaned and transformed into features that could influence customer preferences and restaurant selection.

---

# 4. Restaurant Rating

## 4.1 Problem with raw ratings

A raw restaurant rating does not capture the amount of evidence behind the rating.

For example:

```text
Restaurant A
Rating = 5.0
Reviews = 5

Restaurant B
Rating = 4.6
Reviews = 5,000
```

Treating these ratings equally assumes that five reviews provide the same confidence as thousands of reviews.

I therefore decided not to use the raw rating directly.

---

# 5. Bayesian Rating Adjustment

The restaurant rating was adjusted using a Bayesian-style approach that combines the restaurant's observed rating with the overall dataset average.

The general form is:

$$
AdjustedRating =
\frac{v}{v+m}R+
\frac{m}{v+m}C
$$

where:

* \(R\) = restaurant's observed rating
* \(v\) = number of votes/reviews
* \(C\) = overall average rating
* \(m\) = minimum review threshold / prior-strength parameter

The effect is intuitive:

### Few reviews

The restaurant's rating is pulled toward the overall average.

### Many reviews

The observed restaurant rating has greater influence.

This was chosen because restaurant quality should depend on both **rating and confidence in that rating**.

---

# 6. Review Volume Transformation

Review counts were highly right-skewed.

A small number of restaurants had extremely large numbers of reviews compared with the majority.

Using raw review counts would allow these extreme values to dominate a scoring function.

I therefore used:

$$
log(1+votes)
$$

The `+1` ensures that restaurants with zero votes can still be transformed.

The transformation compresses large values while preserving the relative ordering of review volume.

The purpose was not to remove information from review volume, but to reduce the disproportionate influence of extreme values.

---

# 7. Why Appeal Score Instead of Popularity?

A simple restaurant-selection mechanism could have used overall restaurant popularity.

However, popularity is a **global property**.

It answers:

> "How often is this restaurant selected overall?"

It does not answer:

> "How attractive is this restaurant to this particular customer?"

A customer who prefers inexpensive Indian food should not necessarily have the same restaurant-selection probabilities as a customer who prefers expensive Japanese food.

I therefore introduced a **customer-specific Appeal Score**.

---

# 8. Appeal Score

The Appeal Score combines restaurant characteristics with the preferences of an individual customer.

Conceptually:

$$
Appeal(c,r)
=
w_1CuisineMatch
+
w_2PriceMatch
+
w_3Rating
+
w_4OtherFactors
$$

where:

* \(c\) = customer
* \(r\) = restaurant
* \(w_i\) = component weights

The exact implementation and weights are defined in the customer/order simulation notebook.

The score is then converted into probabilities used to determine restaurant selection.

### Why this approach?

It creates **heterogeneous customer behavior**.

Instead of:

```text
Restaurant popularity
        ↓
Same ordering probability for everyone
```

the simulation becomes:

```text
Customer preferences
        +
Restaurant characteristics
        ↓
Customer-specific Appeal
        ↓
Restaurant choice
```

This makes restaurant diversity, loyalty, cuisine preference, and price sensitivity meaningful downstream features.

---

# 9. Synthetic Customer Generation

Customers were generated with different behavioral characteristics rather than treating every customer identically.

The simulation introduces variation in:

* Ordering frequency
* Spending behavior
* Cuisine preference
* Price preference
* Restaurant preference
* Exploration/loyalty behavior

The objective was to create a population containing different types of customers rather than a collection of independent random observations.

---

# 10. Poisson Distribution for Order Frequency

The number of orders placed by a customer during a fixed time period is a count variable.

I therefore used a Poisson-based approach:

$$
N \sim Poisson(\lambda)
$$

where:

* \(N\) = number of orders
* \(\lambda\) = expected order rate

Different customer profiles can be assigned different values of \(\lambda\).

For example:

```text
Low-frequency customer
λ → lower

Medium-frequency customer
λ → moderate

High-frequency customer
λ → higher
```

This produces variation in ordering frequency while maintaining a realistic non-negative count structure.

### Why not simply generate random order counts?

Because the objective was to model **event frequency**, not just generate arbitrary integers.

---

# 11. Gamma Distribution for Spending

Spending-related variables were modeled using a Gamma distribution:

$$
Spend \sim Gamma(k,\theta)
$$

where:

* \(k\) = shape parameter
* \(\theta\) = scale parameter

Gamma was selected because spending:

* cannot be negative
* is continuous
* is typically right-skewed
* can contain a smaller number of high-spending customers

A normal distribution would allow negative values and would not naturally reproduce the desired skew.

The parameters were chosen based on the intended customer behavior and the implementation in the simulation notebook.

---

# 12. Customer Preferences

Customers were not given identical preferences.

Preference variables influence the probability that a customer interacts with a particular restaurant or cuisine.

Examples include:

```text
Customer
│
├── Preferred cuisines
├── Preferred price range
├── Ordering intensity
├── Spending tendency
└── Exploration / loyalty tendency
```

This creates a relationship between the customer's latent preferences and their observed orders.

---

# 13. Order Simulation

Orders are generated by combining customer behavior with restaurant characteristics.

At a high level:

```text
Customer
    │
    ├── Order frequency
    ├── Spending tendency
    └── Preferences
             │
             ▼
       Appeal Scores
             │
             ▼
     Restaurant selection
             │
             ▼
          Order
```

Each simulated order therefore contains information about both sides of the interaction:

**Customer ↔ Restaurant**

This allows downstream features to capture not just how often customers order, but **what they order and how their preferences evolve**.

---

# 14. Churn Generation

## Initial approach

The first version of the simulation considered churn around customer ordering events.

This created an unintended problem.

Customers who ordered frequently had more opportunities to trigger the churn mechanism than customers who ordered infrequently.

This meant the mechanics of the label-generation process could itself introduce a relationship between frequency and churn.

---

# 15. Revised Time-Based Churn

I changed the mechanism to make churn a **time-based outcome** rather than something evaluated only when an order occurs.

The final structure is:

```text
Days 0–90
Customer observation
       │
       ▼
Feature generation
       │
       ▼
Churn prediction
       │
       ▼
Days 91–180
Future customer behavior
       │
       ▼
Churn outcome
```

This creates a cleaner prediction problem:

> **What can we learn about future churn from behavior observed before the prediction point?**

This change was particularly important because it prevented order frequency from influencing churn simply through the number of opportunities available to trigger the label.

---

# 16. Temporal Feature Engineering

Customer features were calculated using only the observation period.

Examples include:

### Recency

$$
Recency =
PredictionDate - LastOrderDate
$$

### Frequency

Number of orders within specified historical windows.

### Monetary behavior

Spending during historical windows.

### Order-gap behavior

Average or characteristic time between orders.

### Restaurant behavior

* Restaurant diversity
* Restaurant loyalty
* Repeat ordering

### Cuisine behavior

* Cuisine diversity
* Cuisine loyalty
* Preference concentration

### Price behavior

Comparison between customer spending and restaurant price characteristics.

The purpose was to represent **customer behavior**, rather than simply customer identity.

---

# 17. SQL Feature Engineering

Customer-level behavioral features were aggregated using SQL/PostgreSQL.

This was intentionally separated from the machine learning stage.

The process was:

```text
Raw Orders
    ↓
SQL Aggregation
    ↓
Customer-level Feature Table
    ↓
EDA
    ↓
Modeling
```

This mirrors a common analytics workflow where raw transactional data is transformed into an analytical dataset before modeling.

---

# 18. Modeling

The final predictive model uses **XGBoost**.

XGBoost was selected because the resulting dataset contains:

* nonlinear relationships
* mixed feature scales
* behavioral interactions
* potentially complex relationships between recency, frequency, spending, and churn

A tree-based boosting model can capture these interactions without requiring every relationship to be explicitly specified.

The modeling workflow included:

1. Feature preparation
2. Train/test split
3. Baseline model
4. Hyperparameter search
5. Final model
6. Evaluation
7. SHAP interpretation

---

# 19. Hyperparameter Tuning

Hyperparameter tuning was performed in two stages:

```text
Baseline
   ↓
Randomized Search
   ↓
Promising Parameter Region
   ↓
Grid Search
   ↓
Final Configuration
```

The purpose was to first explore the parameter space efficiently and then perform a more focused search around promising configurations.

The final parameters are documented in the model-training notebook.

---

# 20. Evaluation

Multiple metrics were considered because churn prediction is not adequately described by accuracy alone.

The final model achieved:

| Metric    |     Result |
| --------- | ---------: |
| ROC-AUC   | **93.20%** |
| Accuracy  | **87.19%** |
| Precision | **84.27%** |
| Recall    | **84.86%** |
| F1        | **84.56%** |

These results represent performance on the simulated dataset.

They should not be interpreted as expected production performance because both the customer behavior and churn outcomes were generated synthetically.

---

# 21. Why SHAP?

A churn model is useful only if its predictions can be interpreted well enough to support decisions.

SHAP was used to investigate:

* Which features influence predictions most?
* Which features increase churn probability?
* Which features decrease churn probability?
* Why is an individual customer classified as high risk?

This changes the analysis from:

> "Who will churn?"

to:

> "Who will churn, and what behavior is driving that prediction?"

---

# 22. Key Analytical Findings

The analysis highlighted several behavioral patterns.

### Recent activity

Recent customer activity was strongly associated with churn risk.

### Historical spending

Historical spending alone was not sufficient to identify customers who would remain active.

### Behavioral features

Recency, frequency, and ordering behavior provided more useful signals than relying purely on static customer attributes.

### Simulation assumptions matter

The churn-generation issue demonstrated that a model can learn patterns created by the **data-generation process itself**.

This was one of the main lessons of the project.

---

# 23. Synthetic Data: The Central Limitation

The most important limitation is that the customer-level data is simulated.

The project therefore has a controlled relationship between:

```text
Assumptions
    ↓
Synthetic behavior
    ↓
Synthetic churn
    ↓
Model
```

The model can successfully recover patterns that exist in this simulated environment.

However, this does not guarantee that the same relationships exist in real food-delivery data.

For example, a high-performing model on this dataset does not prove that:

> "Recency will produce 93% ROC-AUC on a real food-delivery platform."

It demonstrates that the modeling pipeline can identify patterns generated by the simulation.

---

# 24. Avoiding the Simulation Trap

Because the data is synthetic, model performance should not be the only measure of project quality.

The more important questions are:

1. Were the assumptions reasonable?
2. Were unintended relationships identified?
3. Was future information prevented from entering the features?
4. Were distributions appropriate for the variables being simulated?
5. Was customer heterogeneity introduced?
6. Can the model's learned relationships be explained?
7. Can the workflow be adapted to real customer data?

This is why the project places significant emphasis on **data generation and validation before modeling**.

---

# 25. Production Extension

If real customer-level data were available, I would change several parts of the workflow.

### Validation

Use time-based validation rather than relying only on random splits.

### Calibration

Check whether predicted churn probabilities correspond to observed churn rates.

### Threshold selection

Choose the intervention threshold using business costs rather than simply maximizing F1 or accuracy.

### Retention economics

Estimate:

$$
ExpectedValue =
P(Retention)\times CustomerValue
-
InterventionCost
$$

and use this to prioritize customers.

### Experimentation

A churn model predicts risk.

It does **not** prove that an intervention will prevent churn.

Retention strategies should therefore be tested through controlled experiments such as A/B testing.

---

# 26. Reproducibility

The project is organized into sequential notebooks covering:

```text
01_cleaned.ipynb
        ↓
02_restaurants_eda.ipynb
        ↓
03_restaurants_feature_engineering.ipynb
        ↓
04_Synthetic_Customer_Generation.ipynb
        ↓
05_order_simulation.ipynb
        ↓
06_Preprocessing_and_Model_training.ipynb
        ↓
07_Churn_EDA.ipynb
```

Each stage builds on the output of the previous stage.

The exact implementation, distributions, parameters, and feature calculations can be inspected directly in the corresponding notebooks.

---

# 27. Summary of Key Decisions

| Problem                     | Decision                            | Reason                                                      |
| --------------------------- | ----------------------------------- | ----------------------------------------------------------- |
| No customer-level dataset   | Synthetic customer/order generation | Allows complete end-to-end problem construction             |
| Restaurant ratings          | Bayesian-adjusted rating            | Accounts for rating confidence                              |
| Highly skewed review counts | `log(1 + votes)`                    | Reduces influence of extreme values                         |
| Restaurant selection        | Customer-specific Appeal Score      | Models individual preferences rather than global popularity |
| Order frequency             | Poisson-based simulation            | Appropriate for event counts                                |
| Spending                    | Gamma-based simulation              | Positive and right-skewed                                   |
| Churn generation            | Time-based mechanism                | Avoids order-frequency bias                                 |
| Feature window              | 90-day observation                  | Prevents future-information leakage                         |
| Prediction window           | Following 90 days                   | Defines a future churn outcome                              |
| Model                       | XGBoost                             | Captures nonlinear behavioral relationships                 |
| Interpretation              | SHAP                                | Explains model behavior                                     |

---

# 28. Final Perspective

The central lesson from this project was that **the model is only one part of a data science problem**.

When working with synthetic data, the assumptions used to generate the data can determine what the model eventually learns.

The most valuable part of the project was therefore not achieving a particular AUC.

It was the iterative process of:

**making assumptions → generating data → checking the resulting behavior → identifying flaws → changing the methodology → modeling → interpreting the results.**

With real customer data, the next challenge would be determining which of these simulated relationships actually survive in the real world.
