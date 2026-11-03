# Phase 4: Business Insights

**Status:** Done.
**Code:** [insights.py](../src/rec/insights.py)
**Outputs:** `artifacts/insights.json` (served by the API and shown in the dashboard), [insights.txt](../reports/insights.txt)
**Run:** `PYTHONPATH=src python -m rec.insights`

**Conventions:** *net profit* = profit on Completed orders only. Returns, cancellations and pending orders count as 0. P-values come from chi-square or ANOVA tests. Because there are 1M rows, tiny differences can be "significant", so each finding also reports the **effect size**.

## 1. Headline KPIs

| KPI | Value |
|---|---|
| Orders | 1,000,123 |
| Gross order value (all statuses) | $403.3M |
| Net revenue / net profit (Completed only) | $282.6M / $112.8M |
| Realized revenue / profit (Completed or Returned, and Paid) | $322.7M / $128.8M |
| Completed / returned / cancelled / payment failed | 70.0% / 10.0% / 5.0% / 5.0% |
| Average margin | 39.5% |
| Orders with contradictory status fields | 23.2% |

## 2. Product performance

| Category | Net profit | Share of profit | Share of orders | Net profit per order |
|---|---|---|---|---|
| **Electronics** | $39.2M | **34.8%** | 20.0% | **$196** |
| Home | $24.5M | 21.7% | 20.0% | $122 |
| Sports | $20.4M | 18.1% | 20.0% | $102 |
| Health | $16.1M | 14.3% | 20.0% | $81 |
| Clothing | $12.6M | 11.2% | 20.0% | $63 |

- Each category has the same share of orders, so **the profit ranking is a pure price effect.** Electronics' average price is about $255, against about $83 for Clothing.
- **Within a category the items are indistinguishable.** The ANOVA p-values for net profit across items are 0.92 (Clothing), 0.91 (Electronics), 0.54 (Health), 0.69 (Home) and 0.61 (Sports). The best-vs-worst item gap is only 1% to 3%. **Category, not individual product, is the unit of performance.**
- Brand and sub-category tables are in the JSON and show no meaningful spread beyond what the category mix explains.

## 3. Marketing effectiveness

| Dimension | Conversion p-value | Finding |
|---|---|---|
| Campaign source (6) | 0.41 | Net profit per order is within **0.8%** across channels (Affiliate best, Google Ads worst). No channel is better. |
| Traffic source | 0.030 | Borderline significance with a 1.1% profit spread. Treat as noise. |
| Device | 0.29 | Flat. |
| Coupon used | 0.97 | Coupons do not change conversion. |
| **Discount depth** | **0.49** | **Discounts do not increase conversion but cost margin.** |

**Discounts are the one real marketing lever:** average margin is 45.0% at 0% off, then 42.1%, 38.9%, 35.3%, 31.3%, and **26.7% at 25% off**, with conversion unchanged. In this data, discounting only gives away profit.

## 4. Customer behavior

- **No seasonality or trend:** monthly order volume has a coefficient of variation of 2.5% across full months, and hourly volume is uniform (p = 0.52). Weekday volume differs by 1.1% (weekends are slightly higher).
- Age band, country and given segment all produce flat conversion and profit (see Phase 3).
- Behavioral targeting (device, channel, coupon, cart abandonment) has no measurable effect on value.

## 5. Return patterns

- **10.0% of orders are returned, costing $16.0M in profit.**
- The five reasons are evenly split (19.8% to 20.1% each): Wrong Item, Defective Product, Better Price Elsewhere, Changed Mind and Not As Described.
- Return rate does **not** differ by category (p = 0.21), shipping method (p = 0.92), segment (p = 0.62) or quantity (p = 0.23).
- The **only detectable driver is payment method** (p < 0.001), and the effect is small: PayPal 9.7% vs. Bank Transfer 10.2% (0.5 points). This is statistically significant only because of the sample size.
- Fixable reasons (Wrong Item, Not As Described) are about 40% of returns, and these come from catalogue and fulfilment quality.

## 6. Fraud and payment risk

- `fraud_risk_score` looks uniform from 0 to 100, so 20% of orders score 80 or above.
- Return rate varies by at most 0.2 points across fraud bands (p = 0.05), and payment failure is a flat 5.0%. The score does not predict outcomes in this data, so it works as a **policy filter** for the recommender, not as a validated risk signal.

## 7. Revenue opportunities (sized, with assumptions)

| Opportunity | Estimated profit | Key assumption |
|---|---|---|
| Clear Pending/Processing orders that are already paid | **$13.4M** | These are real orders. Verify first, since 23% of rows have conflicting statuses. |
| Cap discounts at 10% | **$8.8M** | Conversion stays unchanged, as the data suggests. **Validate with an A/B test.** |
| Reduce returns by 20% | $3.2M | Needs about half of the fixable return reasons to be removed. |
| Shift 10% of Clothing demand toward Electronics through recommendations | $1.9M | Customers accept the suggestion. Phase 2 measured +22% profit captured with no hit-rate loss at λ = 0.25. |
| Recover 25% of failed payments | $1.3M | A retry or alternative payment flow converts 25%. |

These are scenario estimates, not forecasts, and each one states its assumption.

## 8. Operations

All four shipping methods (Express, Next Day, Standard, Economy) average about **7.5 delivery days** at different costs. Faster methods do not deliver faster, which is a data inconsistency. We report it and do not draw conclusions from delivery speed.

## 9. Problems encountered and fixes

| Problem | Fix |
|---|---|
| Polars API errors (`isin` is now `is_in`, and `cut` is deprecated) | Changed to `is_in` and suppressed the deprecation warning. |
| I first wrote that returns are "flat across payment method," but the test gave p < 0.001 | Checked the data. The effect is only 0.5 points. Wording corrected to "detectable but small," with a note on effect size at this sample size. |
| The first fraud finding claimed "no relation," but the test gave p = 0.05 | Reworded to the measured gap (0.2 points). It is borderline, and the score is not a validated signal. |
| The mix-shift opportunity used all completed orders, not Clothing orders ($9.3M, overstated) | Used the Clothing completed orders only. The figure is now $1.9M. |
| The returns-cut text contained a clumsy expression | Replaced with a plain statement of the assumption. |

## 10. Takeaways for the showcase

1. **Only three things matter in this dataset:** category mix (Electronics earns 3× Clothing per order), discount depth (it destroys margin without lifting conversion), and order status leakage (returns, cancellations and stuck orders).
2. Product, channel, device, segment, country and time-of-day show no real differences. We can say so with evidence.
3. The business-aware recommender from Phase 2 acts on the one lever that exists: it steers toward higher-value categories and avoids deep discounts.

## 11. Next

Phase 5: solution architecture.
