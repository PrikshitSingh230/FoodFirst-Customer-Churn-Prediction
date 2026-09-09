--customer churn (feature engineering) 
-- Observation window : Day 0 - 90
-- Outcome window      : Day 90 - 180

DROP TABLE IF EXISTS customer_churn_features;

CREATE TABLE customer_churn_features AS

WITH observation_orders AS (
    SELECT *
    FROM order_simulation
    WHERE order_day <= 90
),

outcome_orders AS (
    SELECT DISTINCT customer_id
    FROM order_simulation
    WHERE order_day > 90
      AND order_day <= 180
),

--basic customer order feature

basic_features AS (
    SELECT
        customer_id,
        SUM(order_value) AS total_spend,
        AVG(order_value) AS avg_order_value,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY order_value) AS median_order_value,
        STDDEV_SAMP(order_value) AS std_order_value,
        MAX(order_value) AS max_order_value,
        MAX(order_day) AS last_order_day,
        90 - MAX(order_day) AS recency_days,
        COUNT(DISTINCT restaurant_id) AS unique_restaurants,
        COUNT(DISTINCT cuisine) AS unique_cuisines,
        COUNT(DISTINCT rest_type) AS unique_rest_types,
        AVG(group_size) AS avg_actual_group_size,
        STDDEV_SAMP(group_size) AS std_group_size
    FROM observation_orders
    GROUP BY customer_id
),

--building gap feature

orders_with_previous AS (
    SELECT
        customer_id,
        order_day,
        LAG(order_day) OVER (
            PARTITION BY customer_id
            ORDER BY order_day
        ) AS previous_order_day
    FROM observation_orders
),

gap_features AS (
    SELECT
        customer_id,
        AVG(order_day - previous_order_day) AS mean_order_gap,
        STDDEV_SAMP(order_day - previous_order_day) AS std_order_gap,
        MIN(order_day - previous_order_day) AS min_order_gap,
        MAX(order_day - previous_order_day) AS max_order_gap
    FROM orders_with_previous
    WHERE previous_order_day IS NOT NULL
    GROUP BY customer_id
),

-- ORDER ACTIVITY BY 30-DAY PERIOD

period_features AS (
    SELECT
        customer_id,
        COUNT(*) FILTER (WHERE order_day <= 30) AS orders_first_30d,
        COUNT(*) FILTER (WHERE order_day > 30 AND order_day <= 60) AS orders_middle_30d,
        COUNT(*) FILTER (WHERE order_day > 60 AND order_day <= 90) AS orders_last_30d,

        SUM(order_value) FILTER (WHERE order_day <= 30) AS spend_first_30d,
        SUM(order_value) FILTER (WHERE order_day > 30 AND order_day <= 60) AS spend_middle_30d,
        SUM(order_value) FILTER (WHERE order_day > 60 AND order_day <= 90) AS spend_last_30d,

        COUNT(*) FILTER (WHERE order_day > 83) AS orders_last_7d,
        COUNT(*) FILTER (WHERE order_day > 76) AS orders_last_14d
    FROM observation_orders
    GROUP BY customer_id
),

-- RESTAURANT LOYALTY

restaurant_counts AS (
    SELECT customer_id, restaurant_id, COUNT(*) AS restaurant_orders
    FROM observation_orders
    GROUP BY customer_id, restaurant_id
),

restaurant_loyalty AS (
    SELECT
        customer_id,
        MAX(restaurant_orders)::DECIMAL / SUM(restaurant_orders) AS restaurant_loyalty
    FROM restaurant_counts
    GROUP BY customer_id
),

-- CUISINE LOYALTY

cuisine_counts AS (
    SELECT customer_id, cuisine, COUNT(*) AS cuisine_orders
    FROM observation_orders
    GROUP BY customer_id, cuisine
),

cuisine_loyalty AS (
    SELECT
        customer_id,
        MAX(cuisine_orders)::DECIMAL / SUM(cuisine_orders) AS cuisine_loyalty
    FROM cuisine_counts
    GROUP BY customer_id
),

-- CUSTOMER PREFERENCE MATCHING

preference_features AS (
    SELECT
        o.customer_id,
        AVG(ABS(o.order_value - c.cost_preference)) AS avg_price_deviation,
        AVG(CASE WHEN o.rest_type = c.preferred_rest_type THEN 1.0 ELSE 0.0 END) AS rest_type_match_rate
    FROM observation_orders o
    JOIN customers c ON o.customer_id = c.customer_id
    GROUP BY o.customer_id
),

-- CHURN LABEL

churn_labels AS (
    SELECT
        bf.customer_id,
        CASE WHEN oo.customer_id IS NULL THEN 1 ELSE 0 END AS churned
    FROM basic_features bf
    LEFT JOIN outcome_orders oo ON bf.customer_id = oo.customer_id
)

-- FINAL CUSTOMER-LEVEL DATASET

SELECT
    c.customer_id,
    c.identity_segment,
    c.age,
    c.income,
    c.group_size,
    c.location,
    c.preferred_price_tier,
    c.preferred_rest_type,

    b.total_spend,
    b.avg_order_value,
    b.median_order_value,
    b.std_order_value,
    b.max_order_value,
    b.recency_days,

    g.mean_order_gap,
    g.std_order_gap,
    g.min_order_gap,
    g.max_order_gap,

    p.orders_first_30d,
    p.orders_middle_30d,
    p.orders_last_30d,
    p.orders_last_7d,
    p.orders_last_14d,

    p.spend_first_30d,
    p.spend_middle_30d,
    p.spend_last_30d,

    b.unique_restaurants,
    b.unique_cuisines,
    b.unique_rest_types,

    r.restaurant_loyalty,
    cl.cuisine_loyalty,

    b.avg_actual_group_size,
    b.std_group_size,

    pf.avg_price_deviation,
    pf.rest_type_match_rate,

    ch.churned

FROM customers c
JOIN basic_features b ON c.customer_id = b.customer_id
LEFT JOIN gap_features g ON c.customer_id = g.customer_id
LEFT JOIN period_features p ON c.customer_id = p.customer_id
LEFT JOIN restaurant_loyalty r ON c.customer_id = r.customer_id
LEFT JOIN cuisine_loyalty cl ON c.customer_id = cl.customer_id
LEFT JOIN preference_features pf ON c.customer_id = pf.customer_id
JOIN churn_labels ch ON c.customer_id = ch.customer_id;
