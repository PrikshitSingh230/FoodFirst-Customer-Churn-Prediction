# FoodFirst Customer Churn Prediction

> An end-to-end customer churn analysis project built around a synthetic food-delivery platform, combining restaurant data, synthetic customer behavior, order simulation, SQL feature engineering, exploratory analysis, and XGBoost churn prediction.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Project Approach](#project-approach)
- [Data Strategy](#data-strategy)
- [Restaurant Data](#restaurant-data)
- [Restaurant Feature Engineering](#restaurant-feature-engineering)
- [Synthetic Customer Generation](#synthetic-customer-generation)
- [Order Simulation](#order-simulation)
- [Churn Definition](#churn-definition)
- [Churn Feature Engineering](#churn-feature-engineering)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Churn Modeling](#churn-modeling)
- [Model Performance](#model-performance)
- [Model Interpretability](#model-interpretability)
- [Key Findings](#key-findings)
- [Challenges and Analytical Decisions](#challenges-and-analytical-decisions)
- [Limitations](#limitations)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [How to Explore the Project](#how-to-explore-the-project)
- [Final Takeaway](#final-takeaway)

---

## Project Overview

I wanted to explore a practical question:

**How can a food-delivery platform identify customers who are likely to stop ordering?**

Instead of starting with a ready-made customer churn dataset, I wanted to understand the process of building the customer behavior data itself.

I started with restaurant data and used it as the foundation for a simulated food-delivery environment. From there, I created synthetic customers with different characteristics and behavioral tendencies, simulated their ordering activity, defined churn over a future time period, and finally built a customer-level dataset for churn prediction.

The main goal was not to create a perfect representation of a real food-delivery company.

The goal was to understand the complete analytical workflow:

**Raw restaurant data → restaurant features → synthetic customers → simulated orders → churn definition → SQL feature engineering → EDA → ML model → model interpretation**

Throughout the project, I also tried to question my own assumptions and validate whether the simulated behavior made sense.

---

## Business Problem

Customer acquisition is generally more expensive than retaining an existing customer.

For a food-delivery platform, identifying customers who are becoming inactive can allow the business to intervene before the customer completely stops ordering.

The business questions I wanted to investigate were:

- Which customer behaviors are associated with churn?
- Does recent ordering activity matter more than total historical spending?
- Does restaurant exploration relate to retention?
- Does customer price preference influence churn?
- Can customer behavior observed during one period be used to predict whether the customer will churn in a future period?
- Which features does the final model rely on most heavily?

### Analytical Objective

Build a binary classification model that predicts whether an active customer will churn during a future period using only information available during the observation period.

---

## Project Approach

The project was built in several stages.

```text
Restaurant Dataset
       ↓
Restaurant Cleaning & EDA
       ↓
Restaurant Feature Engineering
       ↓
Synthetic Customer Generation
       ↓
Order Simulation
       ↓
90-Day Observation + 90-Day Outcome
       ↓
SQL Customer-Level Feature Engineering
       ↓
Churn EDA
       ↓
Preprocessing
       ↓
XGBoost Baseline
       ↓
Hyperparameter Tuning
       ↓
Model Evaluation
       ↓
SHAP Interpretation
       ↓
Business Insights


````markdown
# FoodFirst Customer Churn Prediction

> An end-to-end customer churn analysis project for a simulated food-delivery platform, built from restaurant data, synthetic customer behavior, simulated orders, SQL feature engineering, exploratory data analysis, and XGBoost.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Project Approach](#project-approach)
- [Data Strategy](#data-strategy)
- [Restaurant Data](#restaurant-data)
- [Restaurant Feature Engineering](#restaurant-feature-engineering)
- [Synthetic Customer Generation](#synthetic-customer-generation)
- [Order Simulation](#order-simulation)
- [Churn Definition](#churn-definition)
- [Churn Feature Engineering](#churn-feature-engineering)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Churn Modeling](#churn-modeling)
- [Model Performance](#model-performance)
- [Model Interpretability](#model-interpretability)
- [Key Findings](#key-findings)
- [Challenges and Analytical Decisions](#challenges-and-analytical-decisions)
- [Limitations](#limitations)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [How to Explore the Project](#how-to-explore-the-project)
- [Final Takeaway](#final-takeaway)

---

## Project Overview

I wanted to explore a practical question:

**How can a food-delivery platform identify customers who are likely to stop ordering?**

Instead of starting with a ready-made customer churn dataset, I wanted to understand what sits behind a customer-level churn dataset and how customer behavior could be created from the ground up.

I started with restaurant data and used it as the foundation for a simulated food-delivery environment. From there, I created synthetic customers with different characteristics and behavioral tendencies, simulated their ordering activity, defined churn over a future time period, and finally built a customer-level dataset for churn prediction.

The main goal was not to create a perfect representation of a real food-delivery company.

The goal was to understand the complete analytical workflow:

```text
Restaurant Data
      ↓
Restaurant EDA
      ↓
Restaurant Feature Engineering
      ↓
Synthetic Customer Generation
      ↓
Order Simulation
      ↓
Churn Definition
      ↓
SQL Feature Engineering
      ↓
Churn EDA
      ↓
Machine Learning
      ↓
Model Interpretation
      ↓
Business Insights
````

<!-- GRAPH 1: Add overall project pipeline / workflow diagram here -->

Throughout the project, I also tried to question my own assumptions and validate whether the simulated behavior made sense.

This became an important part of the project because some of the first approaches I tried produced behavior that did not match what I was trying to model. Instead of simply accepting the output, I went back and changed the methodology.

[Back to Top](#table-of-contents)

---

## Business Problem

Customer acquisition is generally more expensive than retaining an existing customer.

For a food-delivery platform, identifying customers who are becoming inactive can help the business intervene before those customers completely stop ordering.

The questions I wanted to investigate were:

* Which customer behaviors are associated with churn?
* Does recent ordering activity matter more than total historical spending?
* Does restaurant exploration relate to retention?
* Does customer price preference influence churn?
* Can behavior observed during one period be used to predict whether a customer will churn during a future period?
* Which features does the final model rely on most heavily?

### Analytical Objective

Build a binary classification model that predicts whether an active customer will churn during a future period using only information available during the observation period.

[Back to Top](#table-of-contents)

---

## Project Approach

The project was divided into several stages.

### 1. Understand the restaurant environment

I started with restaurant-level data because customers need restaurants to interact with in the simulated food-delivery environment.

### 2. Engineer restaurant-level signals

I created restaurant features such as price tiers, popularity, rating reliability, and an overall appeal score.

### 3. Create synthetic customers

I created customer personas with different demographic characteristics, price preferences, restaurant preferences, engagement levels, and churn tendencies.

### 4. Simulate customer orders

Customers were then allowed to place orders over a 180-day period.

Their restaurant choices were influenced by their preferences, exploration behavior, restaurant price, and restaurant appeal.

### 5. Define churn using a future period

The first 90 days were treated as the observation period and the following 90 days as the outcome period.

### 6. Engineer customer-level features

I used PostgreSQL to convert the order-level data into one row per customer containing behavioral features.

### 7. Explore churn behavior

I compared retained and churned customers to understand which features appeared to have useful relationships with churn.

### 8. Train and tune XGBoost

I trained a baseline XGBoost model and then used RandomizedSearchCV followed by GridSearchCV to tune the model.

### 9. Interpret the model

Finally, I used SHAP to understand which features were contributing most to the model's predictions.

[Back to Top](#table-of-contents)

---

## Data Strategy

One of the main decisions in this project was **not to use a pre-built customer churn dataset**.

A ready-made churn dataset would have allowed me to start directly with preprocessing and modeling.

However, I wanted to understand how the customer behavior itself could be constructed.

The restaurant dataset therefore became the starting point for building a simulated food-delivery environment.

The overall data flow was:

```text
Restaurant Data
      ↓
Restaurant Features
      ↓
Synthetic Customers
      ↓
Simulated Orders
      ↓
Customer-Level Churn Features
      ↓
Machine Learning Dataset
```

The project contains three main types of data:

* Restaurant-level data
* Customer-level data
* Order-level data

The final machine learning dataset contains one row per customer.

[Back to Top](#table-of-contents)

---

# Restaurant Data

The starting point was a Zomato restaurant dataset containing information such as:

* Restaurant type
* Cuisine
* Approximate cost
* Ratings
* Number of ratings
* Online ordering availability
* Location

I first cleaned and explored the restaurant data before using it in the simulation.

Only restaurants supporting online ordering were considered eligible for simulated food-delivery orders.

**Notebook:** [View Restaurant EDA →](./02_restaurants_eda.ipynb)

<!-- GRAPH 2: Add restaurant EDA graph here -->

The purpose of this stage was not only to clean the data but also to understand what information was available and which variables could realistically influence customer restaurant selection.

[Back to Top](#table-of-contents)

---

# Restaurant Feature Engineering

After exploring the restaurant data, I needed to create signals that could be used by the order simulation.

I wanted customers to have some reason to prefer one restaurant over another instead of selecting restaurants completely randomly.

The main restaurant-level features included:

* Rating quality
* Rating reliability
* Popularity
* Price level
* Overall restaurant appeal

**Notebook:** [View Restaurant Feature Engineering →](./03_restaurants_feature_engineering.ipynb)

---

## Restaurant Price Tier

I created broader price tiers from restaurant pricing information.

The purpose was to turn the raw price information into a simpler representation that could be compared with a customer's price preference.

For example:

```text
Restaurant
    ↓
Price
    ↓
Price Tier

Budget
Mid-range
Premium
Luxury
```

This allowed the simulation to represent a simple behavioral relationship:

> A customer who generally prefers lower-cost restaurants should be more likely to select restaurants within a similar price range.

---

## Initial Approach: Normalised Rating

My first approach was to use restaurant rating as a quality signal.

Since ratings were on a bounded scale, I normalised them so that restaurants could be compared on a common 0 to 1 scale.

The general min-max normalisation formula is:

```text
Normalized Rating =
(Rating - Minimum Rating)
-------------------------
(Maximum Rating - Minimum Rating)
```

This makes the values easier to combine with other numerical signals.

However, while exploring the data, I realised that normalisation did not solve an important problem.

Consider two restaurants:

```text
Restaurant A
Rating = 5.0
Number of ratings = 5

Restaurant B
Rating = 4.6
Number of ratings = 5,000
```

A simple normalised rating would strongly favour Restaurant A.

But the 5.0 rating is based on only five observations.

That made me question whether I should treat every observed rating as equally reliable.

This led to the next approach.

---

## Why I Changed to Bayesian Rating

I used a Bayesian-style weighted rating to account for the number of ratings behind a restaurant's average score.

The general formula is:

```text
Bayesian Rating =
(v / (v + m)) × R
+
(m / (v + m)) × C
```

Where:

```text
R = restaurant's observed average rating
v = number of ratings for the restaurant
C = overall average rating across restaurants
m = minimum rating-count threshold
```

The important idea is the weighting.

A restaurant with very few ratings is pulled closer toward the overall average.

As the number of ratings increases, the restaurant's own observed rating gets more weight.

Conceptually:

```text
Few ratings
     ↓
Low confidence
     ↓
More influence from overall average


Many ratings
     ↓
Higher confidence
     ↓
More influence from restaurant's own rating
```

### Why Bayesian Rating Instead of Normalised Rating?

Normalisation answers:

> "Where does this restaurant's rating sit within the rating range?"

Bayesian weighting answers:

> "How much confidence should I place in this restaurant's observed rating?"

For this project, the second question was more useful.

The restaurant score would later influence restaurant selection during order simulation, so I wanted a restaurant with a very small number of reviews to be treated more cautiously.

This was one of the points where the project changed based on what I observed rather than simply following the first approach I implemented.

<!-- GRAPH 3: Add normalised rating vs Bayesian rating / rating reliability visual here -->

---

## Restaurant Appeal Score

After creating a more reliable rating signal, I combined it with restaurant popularity to create an overall restaurant appeal score.

The general idea was:

```text
Restaurant Appeal
        =
Rating Signal
        +
Popularity Signal
```

The individual components were scaled before being combined so that one variable would not dominate simply because of its numerical range.

The appeal score was then used during restaurant selection.

Conceptually:

```text
High restaurant appeal
        ↓
Higher selection weight

Lower restaurant appeal
        ↓
Lower selection weight
```

However, appeal was not the only factor.

The simulation also considered the customer's price preference and restaurant-type preference.

This meant that a highly appealing restaurant was not automatically selected by every customer.

[Back to Top](#table-of-contents)

---

# Synthetic Customer Generation

Once the restaurant environment was prepared, I needed customers who could interact with it.

Instead of generating completely random customers, I created different customer personas with different characteristics and behavioral tendencies.

**Notebook:** [View Synthetic Customer Generation →](./04_Synthetic_Customer_Generation.ipynb)

The customer dataset includes attributes such as:

* Age
* Income
* Cost preference
* Preferred price tier
* Group size
* Preferred restaurant type
* Exploration probability
* Order frequency
* Engagement tier
* Churn tendency
* Location

---

## Customer Personas

The purpose of personas was to introduce some structure into the synthetic population.

Different customer types were given different tendencies.

For example, some customers are expected to order frequently, while others may order only occasionally.

Some customers are more likely to explore different restaurant types, while others are more likely to stay within their preferred restaurant type.

The main engagement tiers were:

```text
Power User
Regular
Occasional
Trial Only
```

These tiers influence typical ordering frequency and baseline churn tendency.

<!-- GRAPH 4: Add customer persona / engagement distribution graph here -->

The goal was not to claim that these exact proportions or behaviors represent a real food-delivery platform.

The goal was to create a structured population in which different types of customer behavior could emerge during simulation.

[Back to Top](#table-of-contents)

---

# Order Simulation

After generating customers, I simulated their interaction with the restaurant environment.

**Notebook:** [View Order Simulation →](./05_order_simulation.ipynb)

For each customer, the simulation determines:

1. When they place an order
2. Whether they explore restaurants
3. Which restaurant type they prefer
4. How closely the restaurant matches their spending preference
5. The restaurant's appeal
6. Which restaurant is selected
7. Order-level attributes such as order value, cuisine, restaurant type, and group size

---

## Restaurant Selection Logic

Restaurant selection was not completely random.

It was influenced by three main ideas.

### 1. Customer Restaurant Preference

Customers have a preferred restaurant type.

If they are not exploring, restaurants matching that preference form the main candidate pool.

If they are exploring, they can select from the wider online-ordering restaurant pool.

### 2. Price Matching

I wanted customers to be more likely to select restaurants that fit their spending preference.

The difference between the restaurant's cost and the customer's preference is calculated as:

```text
cost_diff =
|restaurant_cost - customer_cost_preference|
```

This difference is then converted into a decreasing weight:

```text
cost_weight =
exp(-cost_diff / 300)
```

Therefore:

```text
Small price difference
        ↓
Higher weight


Large price difference
        ↓
Lower weight
```

### 3. Restaurant Appeal

The restaurant's appeal score is also incorporated into the selection probability.

The final selection weight is therefore conceptually:

```text
Selection Weight
=
Restaurant Appeal
×
Customer Price Fit
```

The weights are then normalised into probabilities, and one restaurant is selected according to those probabilities.

This produces a simulation where customers are not simply ordering from random restaurants.

Their choices are influenced by both **customer preferences** and **restaurant characteristics**.

---

# A Problem I Found in the Churn Simulation

One of the most important changes I made during the project involved the churn simulation itself.

Initially, churn risk was checked once after every order.

At first, this seemed reasonable:

```text
Customer places order
        ↓
Check whether customer churns
        ↓
Continue or stop
```

However, after looking more closely at the logic, I realised that this created an unintended relationship between order frequency and churn risk.

Imagine two customers:

```text
Riya
Power User
Orders approximately every 3 days


Aman
Trial Only
Orders approximately every 30 days
```

Over 180 days:

```text
Riya
≈ 60 orders
≈ 60 churn checks


Aman
≈ 6 orders
≈ 6 churn checks
```

Even though Riya had a much lower churn hazard per check, she received many more opportunities to churn simply because she ordered more often.

That was not the behavior I wanted.

A customer should not become more likely to churn just because they order frequently.

---

## Changing Churn Risk From Order-Based to Time-Based

I changed the approach so that churn risk accumulates according to the amount of time represented by the gap between orders.

The baseline churn hazard is converted into a daily hazard:

```python
daily_hazard = hazard / order_frequency_days
```

Then, for a gap of `n` days, the probability of churning during that gap is:

```text
P(churn during gap)
=
1 - (1 - daily_hazard)^n
```

The reasoning is based on survival probability.

If the probability of surviving one day is:

```text
1 - daily_hazard
```

then the probability of surviving `n` consecutive days is:

```text
(1 - daily_hazard)^n
```

Therefore:

```text
Probability of churn
=
1 - Probability of survival
```

### Simple Example

Suppose:

```text
daily_hazard = 0.001
```

That means approximately a 0.1% daily churn risk.

For a 10-day gap:

```text
P(churn)
=
1 - (1 - 0.001)^10
```

This is approximately:

```text
1 - 0.990
≈ 0.010
```

or about a 1% chance of churning during that 10-day period.

The important part is that the risk is now related to **time passing**, rather than simply the number of times the customer places an order.

This made the simulation more consistent with the behavior I was trying to represent.

<!-- GRAPH 5: Add example of simulated customer order timeline / churn logic here -->

[Back to Top](#table-of-contents)

---

# Churn Definition

The simulation covers 180 days.

I divided this into two separate periods:

```text
Days 0-90
Observation Period

Days 91-180
Outcome Period
```

The model only uses information from the first 90 days.

The following 90 days are used to determine whether the customer churned.

### Customer Eligibility

A customer must have placed at least one order during the observation period to be included in the churn modeling population.

Customers with no observed activity cannot meaningfully be classified under this particular churn definition.

### Churn Definition

A customer is labelled as churned if:

```text
At least one order during days 0-90
+
No order during days 91-180
=
Churned
```

This creates a temporal setup where:

```text
What the model knows
        ↓
Days 0-90

What the model predicts
        ↓
Days 91-180
```

This separation was important because using future ordering behavior as a model feature would create temporal leakage.

[Back to Top](#table-of-contents)

---

# Churn Feature Engineering

After generating the order history, I converted the order-level data into one row per customer using PostgreSQL.

The resulting feature set contains several groups of customer behavior.

---

## Customer Activity

Examples include:

* Total spending
* Average order value
* Median order value
* Maximum order value

These features capture the overall scale of a customer's activity.

---

## Recency

```text
recency_days
```

This represents the number of days since the customer's last order during the observation period.

It is calculated using only information available up to the end of day 90.

This feature became especially important later in both EDA and model interpretation.

---

## Order Gaps

I calculated features describing the time between orders:

* Mean order gap
* Minimum order gap
* Maximum order gap
* Standard deviation of order gaps

Some of these features naturally contain missing values.

For example, if a customer has only one order, there is no previous order from which to calculate a gap.

Therefore:

```text
NaN
```

does not mean:

```text
Gap = 0
```

It means there was not enough information to calculate the statistic.

---

## Recent Activity

I created several time-window features:

* Orders in first 30 days
* Orders in middle 30 days
* Orders in last 30 days
* Orders in last 7 days
* Orders in last 14 days

These features were designed to capture changes in customer engagement over the observation period.

---

## Spending Trends

I also calculated spending across different periods:

* Spending in first 30 days
* Spending in middle 30 days
* Spending in last 30 days

This allows the model to distinguish between customers who were historically active and customers who are currently active.

---

## Restaurant and Cuisine Behavior

The feature set also includes:

* Unique restaurants
* Unique cuisines
* Unique restaurant types
* Restaurant loyalty
* Cuisine loyalty

These features capture whether customers tend to repeatedly use the same restaurants or explore different parts of the platform.

---

## Preference Matching

I also included:

* Average price deviation
* Restaurant type match rate

These attempt to capture whether the restaurants customers actually selected were consistent with their stated preferences.

[Back to Top](#table-of-contents)

---

# Exploratory Data Analysis

**Notebook:** [View Churn EDA →](./07_Churn_EDA.ipynb)

The churn EDA focused on understanding which customer behaviors appeared to differ between retained and churned customers.

I intentionally kept this analysis focused instead of creating a graph for every feature.

The main areas were:

* Churn distribution
* Numerical feature distributions
* Behavioral differences between retained and churned customers
* Categorical churn rates

---

## Numerical Feature Analysis

The main numerical features explored were:

* Recency
* Orders in the last 30 days
* Total spending
* Average order value
* Mean order gap
* Unique restaurants

<!-- GRAPH 6: Add numerical feature distribution grid here -->

### Observations

**Recency**

Churned customers generally show higher recency values, meaning they had gone longer without ordering by the end of the observation period.

**Orders in the Last 30 Days**

Retained customers generally placed more recent orders, while churned customers showed lower recent engagement.

**Total Spend**

Total spending showed considerable overlap between retained and churned customers.

This suggests that historical spending alone does not strongly distinguish churners from retained customers.

**Average Order Value**

Average order value distributions were relatively similar between the two groups.

**Unique Restaurants**

Retained customers generally showed slightly higher restaurant diversity, suggesting that greater exploration may have some relationship with retention.

**Mean Order Gap**

The relationship was less straightforward than expected.

This showed that churn was not simply driven by customers having longer average gaps between orders.

---

# Recency vs Churn

One of the clearest patterns in the EDA was the relationship between recency and churn.

<!-- GRAPH 7: Add Recency vs Churn boxplot here -->

Customers with higher recency were generally more likely to churn.

From a business perspective, this makes intuitive sense.

A customer who has not ordered recently may be showing an early sign of disengagement.

This also became one of the strongest signals in the final machine learning model.

[Back to Top](#table-of-contents)

---

# Recent Orders vs Churn

Another strong relationship appeared when looking at the number of orders placed during the last 30 days of the observation period.

<!-- GRAPH 8: Add Orders Last 30 Days vs Churn graph here -->

Customers with stronger recent ordering activity were generally more likely to be retained.

This suggests that a decline in recent engagement can be a useful warning signal for potential churn.

[Back to Top](#table-of-contents)

---

# Categorical Feature Analysis

I also compared churn and retention rates across:

* Identity segment
* Preferred price tier
* Preferred restaurant type

<!-- GRAPH 9: Add categorical churn comparison graph here -->

### Identity Segment

Churn and retention rates were relatively similar across identity segments.

This suggests that identity segment has limited influence on churn in the simulated dataset.

This was not necessarily a problem.

The synthetic churn mechanism was designed primarily around engagement behavior, so it was reasonable for behavioral variables to show stronger relationships with churn than demographic identity.

### Preferred Price Tier

Churn rates varied more noticeably across price tiers.

This suggests that customer spending preference may have some relationship with the simulated churn behavior.

### Preferred Restaurant Type

Churn rates varied across restaurant preferences, although there was considerable overlap between retained and churned customers.

The differences were less pronounced than the behavioral differences seen in recent activity.

[Back to Top](#table-of-contents)

---

# Churn Modeling

**Notebook:** [View Preprocessing and Model Training →](./06_Preprocessing_and_Model_training.ipynb)

The target variable is:

```text
churned
```

where:

```text
0 = Retained
1 = Churned
```

---

## Preprocessing

The customer ID was removed because it is an identifier rather than a predictive feature.

The following categorical features were one-hot encoded:

* Identity segment
* Location
* Preferred price tier
* Preferred restaurant type

Numerical features were kept in their original form.

Scaling was not required because XGBoost is a tree-based model and does not require numerical features to be on the same scale.

The train-test split was performed before preprocessing so that information from the test set was not used during model development.

---

## Features Excluded From the Model

I avoided including variables that would directly expose the mechanism used to generate the churn label.

For example, the underlying churn hazard used during simulation is not a useful real-world predictive feature because it directly influences how churn was generated.

The objective was to see whether customer behavior could recover the churn signal rather than simply giving the model the answer.

This is an important distinction in a synthetic project.

[Back to Top](#table-of-contents)

---

# Baseline XGBoost Model

I first trained a basic XGBoost classifier to establish a baseline.

The baseline achieved:

```text
Test Accuracy = 86.97%
Test ROC-AUC  = 92.17%
```

However, the training accuracy was approximately:

```text
99.61%
```

while the test accuracy was approximately:

```text
86.97%
```

This showed a noticeable gap between training and test performance.

Rather than accepting the first high-performing model, I treated this as a sign that the model was fitting the training data too closely.

This motivated the next step:

**hyperparameter tuning and regularisation.**

[Back to Top](#table-of-contents)

---

# Hyperparameter Tuning

I used two stages of hyperparameter search.

## RandomizedSearchCV

RandomizedSearchCV was used to explore a relatively wide range of hyperparameter combinations efficiently.

The purpose was to identify a promising region rather than exhaustively test every possible combination.

## GridSearchCV

After identifying a promising region, GridSearchCV was used to test selected combinations more systematically.

This created a two-stage process:

```text
Wide Search
    ↓
Promising Parameter Region
    ↓
Focused Grid Search
    ↓
Final Model
```

The final tuned model used parameters including:

```text
max_depth = 4
learning_rate = 0.05
n_estimators = 80
subsample = 0.6
colsample_bytree = 0.6
min_child_weight = 1
scale_pos_weight = 2
gamma = 0
```

The tuned model also substantially reduced the training-test performance gap.

[Back to Top](#table-of-contents)

---

# Model Performance

The final tuned XGBoost model achieved:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 87.19% |
| Precision | 84.27% |
| Recall    | 84.86% |
| F1 Score  | 84.56% |
| ROC-AUC   | 93.20% |

---

## Accuracy

**87.19%**

The model correctly classified approximately 87% of customers in the test set.

---

## Precision

**84.27%**

Of the customers predicted as churners, approximately 84% actually churned.

This matters when a business wants to avoid spending retention resources on too many customers who were not actually at risk.

---

## Recall

**84.86%**

The model identified approximately 85% of the customers who actually churned.

For churn prediction, recall is particularly important because missing an actual churner means losing an opportunity to intervene.

---

## F1 Score

**84.56%**

F1 combines precision and recall into a single metric.

The relatively balanced precision and recall indicate that the model is not achieving high recall simply by classifying an excessive number of customers as churners.

---

## ROC-AUC

**93.20%**

The ROC-AUC measures the model's ability to distinguish churned customers from retained customers across different classification thresholds.

An AUC of 0.932 indicates strong discrimination on this synthetic test dataset.

[Back to Top](#table-of-contents)

---

# Confusion Matrix

<!-- GRAPH 10: Add confusion matrix here -->

The final confusion matrix contained:

```text
True Negatives  = 1447
False Positives = 182
False Negatives = 174
True Positives  = 975
```

One useful comparison was the reduction in false negatives:

```text
Baseline:
243 false negatives

Tuned model:
174 false negatives
```

The tuned model therefore identified more actual churners, although it also increased the number of false positives.

For a churn intervention problem, this can be a reasonable trade-off because missing a customer who is actually going to churn can be more costly than contacting a customer who ultimately stays.

[Back to Top](#table-of-contents)

---

# ROC Curve

<!-- GRAPH 11: Add ROC curve here -->

The ROC curve evaluates the model across different classification thresholds.

The final model achieved:

```text
ROC-AUC = 0.932
```

This indicates strong separation between churned and retained customers within the simulated dataset.

[Back to Top](#table-of-contents)

---

# Model Interpretability

A predictive model can perform well without making it obvious why it is making its predictions.

I therefore used SHAP to understand the model's behavior.

---

## SHAP Analysis

SHAP, or SHapley Additive exPlanations, was used to understand how individual features contribute to the model's predictions.

<!-- GRAPH 12: Add SHAP beeswarm plot here -->

In the SHAP beeswarm plot:

* Each point represents one customer.
* Features are ordered by their overall importance.
* Positive SHAP values push the prediction toward churn.
* Negative SHAP values push the prediction away from churn.
* Red represents higher feature values.
* Blue represents lower feature values.

The most influential feature was:

```text
recency_days
```

This was consistent with the earlier EDA.

Recent customer activity therefore appeared important from both:

```text
EDA
+
Machine Learning Interpretation
```

---

## SHAP vs XGBoost Feature Importance

I also found that the feature rankings from XGBoost's built-in feature importance and SHAP were not identical.

This is expected because they measure feature influence differently.

XGBoost feature importance reflects how features contribute to the tree-building process.

SHAP focuses on how features contribute to individual predictions.

Therefore, I used SHAP to provide a more detailed interpretation of the final model instead of treating the built-in feature importance ranking as the complete explanation.

[Back to Top](#table-of-contents)

---

# Key Findings

## 1. Recent activity is strongly associated with churn

Recency was one of the strongest signals in both EDA and SHAP analysis.

Customers who had gone longer without ordering were more likely to churn.

---

## 2. Recent ordering activity matters

Customers with fewer orders in the last 30 days generally showed higher churn.

This suggests that a decline in recent engagement can act as an early warning signal.

---

## 3. Historical spending alone is not enough

Total spending and average order value showed considerable overlap between retained and churned customers.

A customer can have high historical spending but still become inactive.

This highlights why churn analysis should focus on **behavior over time**, rather than only cumulative customer value.

---

## 4. Restaurant exploration may relate to retention

Retained customers generally showed slightly greater restaurant diversity.

This suggests that customers who interact with a wider variety of restaurants may have somewhat stronger engagement with the platform.

---

## 5. Identity segment has limited influence

Identity segments showed relatively similar churn rates.

This is consistent with the way the synthetic churn mechanism was designed, where engagement behavior was more influential than demographic identity.

---

## 6. Behavioral features dominate profile features

The strongest model signals came primarily from:

* Recency
* Recent order frequency
* Order gaps
* Spending activity
* Restaurant diversity

This reinforces an important analytical idea:

> For churn prediction, what a customer is doing can be more informative than simply who the customer is.

[Back to Top](#table-of-contents)

---

# Challenges and Analytical Decisions

This project was also an exercise in identifying assumptions, testing them, and changing the approach when the output did not make sense.

## 1. Why not use a ready-made churn dataset?

A ready-made churn dataset would have allowed me to move directly into modeling.

However, I wanted to understand the process behind the customer-level data.

Building the customers and their order history myself allowed me to understand how the final features were created and how assumptions made earlier in the project could affect the final model.

---

## 2. Normalised Rating Was Not Enough

My initial restaurant quality signal used normalised rating.

The problem was that normalisation does not consider how many ratings support that average.

A restaurant with a perfect rating from only a few customers could therefore receive an unrealistically strong score.

This led me to Bayesian rating.

The Bayesian approach introduced the idea of rating reliability by considering both:

```text
Observed Rating
+
Number of Ratings
```

This was more appropriate for the restaurant-selection mechanism I wanted to build.

---

## 3. The First Churn Simulation Approach Had a Problem

Initially, churn was checked once after every order.

I realised this unintentionally gave frequent customers more opportunities to churn.

I changed the approach so that churn probability accumulates over the time represented by each order gap.

This made the simulated churn mechanism more consistent with the intended behavior.

---

## 4. Temporal Leakage Had to Be Avoided

The model uses days 0-90 to predict churn during days 91-180.

Features such as:

```text
recency_days
orders_last_30d
spend_last_30d
```

are calculated using only the observation period.

Future orders are used only to determine the churn outcome.

This keeps the prediction setup temporally separated.

---

## 5. Missing Values Were Not Always Errors

Some statistical features naturally contain missing values.

For example:

```text
One order
    ↓
No previous order
    ↓
Order gap = NaN
```

Similarly, standard deviation cannot be meaningfully calculated when there is only one observation.

Therefore, these missing values were retained rather than automatically replacing them with zero.

The distinction was important:

```text
NaN
=
Not enough observations


0
=
Actual measured value is zero
```

XGBoost can also handle missing numerical values natively.

[Back to Top](#table-of-contents)

---

# Limitations

This project is intentionally synthetic, so the results should not be interpreted as real food-delivery industry benchmarks.

## Synthetic Customer Behavior

Customer behavior was generated using assumptions defined for the project.

Therefore, the model may learn relationships that are partly created by the simulation itself.

## Simulated Churn

The churn label is based on the simulated ordering process rather than observed real-world customer churn.

## Restaurant Data

The restaurant dataset provides the environment for the simulation, but it does not contain actual FoodFirst customer behavior.

## Model Generalisation

The reported model performance should not be interpreted as evidence that the same model would achieve a 93.20% ROC-AUC on a real food-delivery platform.

The primary purpose of the project is to demonstrate the analytical workflow:

```text
Data
→
Assumptions
→
Feature Engineering
→
EDA
→
Modeling
→
Interpretation
```

[Back to Top](#table-of-contents)

---

# Project Structure

```text
FoodFirst-Customer-Churn-Prediction/
│
├── 01_cleaned.ipynb
├── 02_restaurants_eda.ipynb
├── 03_restaurants_feature_engineering.ipynb
├── 04_Synthetic_Customer_Generation.ipynb
├── 05_order_simulation.ipynb
├── 06_Preprocessing_and_Model_training.ipynb
├── 07_Churn_EDA.ipynb
│
├── EDA_restaurants.csv
├── restaurants.csv
├── customers_dummy.csv
├── order_simulation(1).csv
├── customer_churn_features_clean.csv
│
├── requirements.txt
└── .gitignore
```

Large raw and intermediate datasets that are not required for reproducing the final analysis are intentionally excluded from version control.

[Back to Top](#table-of-contents)

---

# Technologies Used

### Programming and Analysis

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn

### Database and SQL

* PostgreSQL
* SQL aggregations
* Window functions
* Percentiles
* Customer-level feature engineering

### Machine Learning

* XGBoost
* Scikit-learn
* RandomizedSearchCV
* GridSearchCV

### Model Interpretation

* SHAP

### Version Control

* Git
* GitHub

[Back to Top](#table-of-contents)

---

# How to Explore the Project

If you want to follow the project in the order in which it was built, I recommend going through the notebooks in this sequence.

## 1. Data Cleaning

[Open `01_cleaned.ipynb` →](./01_cleaned.ipynb)

Initial cleaning and preparation of the restaurant dataset.

---

## 2. Restaurant EDA

[Open `02_restaurants_eda.ipynb` →](./02_restaurants_eda.ipynb)

Exploration of the restaurant dataset and identification of useful patterns.

---

## 3. Restaurant Feature Engineering

[Open `03_restaurants_feature_engineering.ipynb` →](./03_restaurants_feature_engineering.ipynb)

Restaurant scoring, Bayesian rating, popularity, and price-tier development.

---

## 4. Synthetic Customer Generation

[Open `04_Synthetic_Customer_Generation.ipynb` →](./04_Synthetic_Customer_Generation.ipynb)

Creation of customer personas, customer attributes, engagement behavior, and churn tendencies.

---

## 5. Order Simulation

[Open `05_order_simulation.ipynb` →](./05_order_simulation.ipynb)

Simulation of customer ordering behavior and restaurant selection.

---

## 6. Churn Modeling

[Open `06_Preprocessing_and_Model_training.ipynb` →](./06_Preprocessing_and_Model_training.ipynb)

Preprocessing, XGBoost modeling, hyperparameter tuning, evaluation, and SHAP analysis.

---

## 7. Churn EDA

[Open `07_Churn_EDA.ipynb` →](./07_Churn_EDA.ipynb)

Customer-level exploratory analysis focused on retained versus churned customers.

[Back to Top](#table-of-contents)

---

# Final Takeaway

The main thing I wanted to understand through this project was not simply:

> "Can I train a churn prediction model?"

It was:

> **"Can I build a complete analytical pipeline where the data, assumptions, features, model, and conclusions all connect to each other?"**

Starting from restaurant data, I built a simulated customer environment, introduced different customer behaviors, generated an order history, defined churn using a future time window, engineered customer-level features using SQL, and finally trained and interpreted an XGBoost model.

The final model achieved:

```text
Accuracy  = 87.19%
Precision = 84.27%
Recall    = 84.86%
F1 Score  = 84.56%
ROC-AUC   = 93.20%
```

More importantly, the analysis showed a consistent story.

**Recent customer activity, particularly recency and recent order frequency, was much more informative for churn than simply looking at historical spending or customer identity.**

The project also helped me understand that analytical work is not always a straight line.

Some of the most useful parts of the project came from questioning my initial assumptions:

```text
Normalised Rating
       ↓
Realised rating volume matters
       ↓
Bayesian Rating


Order-Based Churn Check
       ↓
Realised frequent customers get more churn checks
       ↓
Time-Based Churn Risk


Initial Model
       ↓
Large train-test gap
       ↓
Hyperparameter Tuning


Model Performance
       ↓
Need to understand why predictions happen
       ↓
SHAP Analysis
```

Because the dataset is synthetic, these findings should be viewed as the result of the assumptions built into the simulation rather than real-world food-delivery behavior.

The main value of the project is the end-to-end analytical process and the reasoning behind each stage.

---

[Back to Top](#table-of-contents)

```
```
