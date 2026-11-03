# Phase 3: Customer Intelligence

**Status:** Done.
**Code:** [customers.py](../src/rec/customers.py)
**Outputs:** [customers.txt](../reports/customers.txt) (full tables), `artifacts/customer_intelligence.json`, `data/processed/customers_full.parquet`
**Run:** `PYTHONPATH=src python -m rec.customers`

## 1. What this phase can and cannot do

Each customer has **one order** (Phase 1, F1), so the textbook tools are out of reach:

| Requested | Standard method | Possible here? | What we did instead |
|---|---|---|---|
| Segmentation | RFM (recency, frequency, monetary) | **No.** Frequency is always 1 and recency is fixed by the order date. | Value tiers on realized net profit, plus K-Means on profile and order features. |
| High-value customers | Top customers by lifetime spend | **Partly.** There is one order of value per customer. | Top 10% by net profit, with a Pareto analysis and a test of whether they can be identified in advance. |
| Behavioral analysis | Journey, cohort and repeat analysis | **Partly.** There is no sequence over time. | Conversion and value by device, campaign, traffic, coupon, cart abandonment and more, with significance tests. |
| CLV | BG/NBD + Gamma-Gamma | **No.** It needs repeat purchases. | A transparent scenario proxy with explicit assumptions. |

**Definition used everywhere:** *net profit* = profit on **Completed** orders only. Returned, cancelled and pending orders count as 0, consistent with Phase 2.

## 2. Value tiers and high-value customers

| Tier | Customers | Avg net profit | Share of customers | Share of net profit |
|---|---|---|---|---|
| **Top 10%** (≥ $304.29) | 100,008 | $478.4 | 10% | **42.4%** |
| High value ($32 to $304) | 400,043 | $146.4 | 40% | 51.9% |
| Standard (≤ $32) | 200,181 | $32.0 | 20% | 5.7% |
| No realized value (returned, cancelled or pending) | 299,891 | $0 | 30% | 0% |

**Pareto:** the top 1% of customers produce 7.9% of net profit, the top 10% produce 42.4%, the top 20% produce 63.9%, and the top 50% produce 94.3%. 30% of customers produce nothing because their order did not complete.

**What a "high-value" customer is here:** it is **order mechanics, not the customer.** Value is `price × quantity × margin`. High-value orders are concentrated in Electronics (24.8% of Electronics customers are in the top 10%, against 1.0% for Clothing, 4.0% for Health, 8.1% for Sports and 12.1% for Home).

**Can we spot them in advance?** A gradient-boosted model using only customer attributes (age, loyalty, tenure, session length, pages, fraud score) predicts "top 10%" with an **AUC of 0.504** (0.5 is a coin flip). No, they can't be predicted from who the customer is. This is the same no-signal result as Phase 2. Value comes from *what is bought*, so the best lever is to steer customers toward higher-value products. That is the profit-aware re-ranker and the upsell logic from Phase 2.

## 3. Segmentation

### 3.1 The supplied label means nothing
`customer_segment` (Regular 60%, Premium 30%, VIP 10%) shows no relationship to anything:

| Label | Avg net profit | Loyalty score | Share who are high value |
|---|---|---|---|
| Regular | $112.7 | 49.99 | 9.97% |
| Premium | $112.9 | 49.97 | 10.04% |
| VIP | $113.1 | 49.84 | 10.07% |

A "VIP" customer is worth the same as a "Regular" one. We keep the label as an input feature but **do not use it as a value signal**.

### 3.2 K-Means clustering
K-Means (k = 4) on nine standardized features: age, loyalty, tenure, session length, pages visited, fraud score, quantity, discount %, and log unit price.

- **Silhouette is about 0.07** for every k from 3 to 8 (1 is perfect, near 0 means no cluster structure). The data is uniformly random, so the algorithm just cuts it into slices.
- The four clusters differ in average net profit ($145, $135, $111, $61), but that comes from basket size and price, which are order mechanics. They are **not distinct customer types**.
- Cluster names are therefore neutral ("Cluster A (highest value)" through "Cluster D (lowest value)"). The first version of the script used descriptive names such as "Big-basket premium buyers", and I removed them because the numbers did not support them.

**Verdict:** value tiers are the usable segmentation. The clusters are kept for completeness and shown with their weak silhouette score.

## 4. Behavioral analysis

For each dimension, we tested whether conversion (Completed rate) or net profit differs across groups (chi-square and ANOVA).

| Dimension | Conversion p-value | Profit p-value | Profit spread across groups |
|---|---|---|---|
| Device | 0.29 | 0.61 | 0.3% |
| Campaign source | 0.41 | 0.58 | 0.7% |
| Traffic source | 0.030 | 0.12 | 1.1% |
| Coupon used | 0.97 | 0.64 | 0.1% |
| Abandoned cart before | 0.37 | 0.30 | 0.3% |
| Weekend | 0.09 | 0.71 | 0.1% |
| Gender | 0.66 | 0.97 | 0.0% |
| Country | 0.70 | 0.63 | 1.3% |
| Given segment | 0.40 | 0.58 | 0.4% |

- **Nothing is meaningful.** One p-value (traffic source, conversion, 0.030) is under 0.05, but with nine dimensions tested a result like that is expected by chance, and the profit spread is 1.1%.
- A customer who abandoned a cart converts at 70.1% and one who did not at 70.0%. Coupons do not change conversion either.
- **Practical meaning:** device, channel, coupon and cart abandonment can't be used to target customers in this dataset.

## 5. Customer lifetime value (CLV)

With one order per customer, a probabilistic CLV model (BG/NBD + Gamma-Gamma) can't be fitted. We use a scenario formula:

```
CLV_3y = realized net profit + (mean net profit per order × orders per year × Σ r^t for t = 1..3)
```

- Assumptions: **r = 0.8 yearly retention** (factor 1.952), mean net profit per order **$112.78**, base case **1 order per year**.
- **Results:**

  | Scenario | Orders per year | Mean 3-year CLV |
  |---|---|---|
  | Conservative | 0.5 | $222.9 |
  | **Base** | 1.0 | **$332.9** (median $277.7, p90 $524.4) |
  | Optimistic | 2.0 | $553.1 |

- **Problem found and fixed:** the first version used `total_orders_by_customer` (average 25.5 orders) to set each customer's order rate, which gave an unrealistic mean CLV of $2,144. That column contradicts the one-order data, and the derived order rate has **0.002 correlation** with actual profit. It is noise, so I removed it and used the flat assumption instead.
- **Limitation to state in the showcase:** retention and order rate are **assumptions**, not estimates from the data. Customer-level differences in CLV come only from the first order's value.

## 6. Problems encountered and fixes

| Problem | Fix |
|---|---|
| RFM impossible (one order per customer) | Value tiers on net profit, with the limitation documented. |
| K-Means produced descriptive-sounding clusters that the data did not support | Silhouette is about 0.07. Renamed the clusters neutrally and documented the clustering as a weak partition. |
| First CLV version was driven by a noisy column (mean $2,144) | Dropped `total_orders_by_customer`. Used explicit scenarios (mean $333 in the base case). |
| The given VIP/Premium/Regular label is meaningless | Tested and documented. It is not used as a value signal. |
| Windows console encoding garbled the report box characters | Wrote reports to UTF-8 files and read results from the JSON. |

## 7. Takeaways for the showcase

1. About 42% of profit comes from the top 10% of customers, but they **can't be identified in advance** from customer attributes (AUC 0.504). What differs is the product and basket.
2. Segment labels, device, campaign, coupon and cart behavior have no effect on value. That is a finding, not a gap in the analysis.
3. CLV is a scenario with stated assumptions. The $333 base case is a planning number, not a forecast.

## 8. Next

Phase 4: business insights (product performance, marketing, behavior, revenue opportunities, returns).
