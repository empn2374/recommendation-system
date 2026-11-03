# Phase 2: Feature Engineering and Recommendation Design

**Status:** Done. Strategy chosen: **honest, business-aware recommender** (see Phase 1, section 6).
**Code:** [features.py](../src/rec/features.py), [models.py](../src/rec/models.py), [train.py](../src/rec/train.py), [evaluate.py](../src/rec/evaluate.py), [crosssell.py](../src/rec/crosssell.py)
**Outputs:** `data/processed/model_table.parquet`, `artifacts/items.parquet`, `artifacts/cross_sell.parquet`, `artifacts/price_tiers.parquet`, `artifacts/metrics.json`, **`artifacts/models/lite_models.pkl` (the served, numpy-only models) and `data/processed/models_sklearn/*.joblib` (the scikit-learn originals, local only)**, [evaluation.txt](../reports/evaluation.txt)
**Run:** `PYTHONPATH=src` then `python -m rec.features`, `python -m rec.train` (fit and save models), `python -m rec.evaluate`, `python -m rec.crosssell`

## 1. Problem framing (shaped by Phase 1)

Every customer has one order, so there is no history to learn taste from. Each customer is a cold-start user described by a **profile plus context** (segment, country, age, loyalty, tenure, device, campaign, traffic, coupon, abandoned cart, weekend, hour, month, session length and pages visited). The task is:

> Given a customer profile/context, rank the **48 catalogue items** by relevance **and** business value.

- **Item** = `product_name` (48 items in 5 categories). `product_id` is unusable (Phase 1, F2).
- **Positive signal** = a Completed order. Returned, Cancelled and Pending orders are not counted as purchases.
- **Temporal split (no leakage):** train = orders before 2025-11-01 (871,769 rows). Test = the last three months, completed purchases only (**89,889**). All item statistics are computed on train only.

## 2. Feature engineering (`features.py`)

**Customer/context features (16, all bucketed):** segment, country, gender, age band (decade), loyalty band (quartiles of the 0 to 100 score), tenure band (years since account creation), device, campaign source, traffic source, coupon used, abandoned cart, weekend, hour band (6 h blocks), order month, session-duration band, pages-visited band.

**Item table (`artifacts/items.parquet`, 48 rows):** order count and popularity share, average price and order value, margin %, average discount, return rate, cancel rate, average and high-risk fraud score, rating, positive-sentiment share, **net profit per order** (profit only on Completed orders, so returns and cancellations count as zero), net revenue per order, 90-day recent demand, stock, and price rank within the category.

Net profit per order is the key business metric. It includes the cost of returns and cancellations, unlike a gross profit figure.

## 3. Recommendation approaches (`models.py`)

| # | Approach | Type | Notes |
|---|---|---|---|
| 1 | `random` | Reference | The floor. Hit@5 should be about 5/48 = 10.4%. |
| 2 | `popularity` | Popularity | Share of completed purchases in train. |
| 3 | `trending_90d` | Popularity | Last 90 days of train only. |
| 4 | `segment_popularity` | Popularity + rules | Popularity within (segment × country), shrunk to the global share. |
| 5 | `profile_naive_bayes` | Profile model | P(item given the 16 features) with CategoricalNB. |
| 6 | `profile_logreg` | Profile model | Multinomial logistic regression on one-hot features (250k-row sample). |
| 7 | `hybrid` | Hybrid | 50/50 blend of the profile model and popularity, which is the cold-start safety net. |
| 8 | **Business re-ranker** | Business rules | `final = (1−λ)·relevance + λ·net_profit_per_order`, both scaled to 0..1 with an optional risk penalty. λ is the "profit weight" slider for the UI. |
| 9 | Cross-sell | Rule-based | See section 5. |
| 10 | Upsell | Rule-based | See section 5. |

**Not built, with reasons:** classic collaborative filtering (ALS or item-item) and co-purchase association rules need repeat customers or baskets, which do not exist (F1). Content-based similarity on 48 items with identical statistics adds nothing measurable, so only category/price-tier structure is used for the rules.

## 4. Results (`reports/evaluation.txt`)

### 4.1 Relevance (Hit@5; random = 0.104; CI is ±0.002)

| Model | Hit@5 | NDCG@5 | Catalogue coverage | Lift vs. popularity |
|---|---|---|---|---|
| random | 0.1052 | 0.0633 | 10% | n/a |
| popularity | 0.1112 | 0.0662 | 10% | baseline |
| trending 90d | 0.1105 | 0.0649 | 10% | −0.6% |
| segment popularity | 0.1121 | 0.0665 | 75% | +0.8% |
| profile naive Bayes | 0.1116 | 0.0661 | 77% | +0.3% |
| profile logistic regression | 0.1118 | 0.0662 | 94% | +0.5% |
| hybrid | 0.1122 | 0.0665 | 77% | +0.9% |

- **Personalization gives no measurable lift.** Paired test of logistic regression vs. popularity on hit@5: difference +0.0006, 95% CI [−0.0022, +0.0034]. This includes zero. At @10 the profile models are slightly *below* popularity.
- **Popularity beats random by only about 0.6 points.** The cause is an artifact, not taste: each category has about 20% of orders but a different number of items (Clothing 9, Electronics 12), so Clothing items each get a bigger share.
- This confirms Phase 1 F4 end to end. **The honest conclusion is that this dataset contains no learnable individual preference.**

### 4.2 Business-aware re-ranking, the part with real signal (hybrid relevance, K=5)

| λ (profit weight) | Hit@5 | Profit captured per customer | Avg item value ($) | Top-5 list |
|---|---|---|---|---|
| 0.00 (relevance only) | 0.1122 | $14.96 | 92.6 | Sports 2, Health 2, Home 1 |
| **0.25** | **0.1116** | **$18.29 (+22%)** | 114.4 | Home 3, Sports 1, Health 1 |
| 0.50 | 0.0846 | $23.62 (+58%) | 196.3 | Electronics 5 |
| 0.75 | 0.0844 | $23.56 | 196.6 | Electronics 5 |
| 1.00 (profit only) | 0.0837 | $23.08 | 197.5 | Electronics 5 |

- "Profit captured" is the average actual profit of test purchases whose item appears in the recommended list.
- **λ ≈ 0.25 is the sweet spot:** +22% profit captured for a 0.5% relative drop in hit rate, and full catalogue coverage.
- Past λ = 0.5, the list collapses to five Electronics items. Hit rate falls by about 25% and profit gain flattens. This is the relevance-vs-profit trade-off for the demo slider.
- Item value differs by category, from Electronics $196 to Clothing $63 per order. **Real differences exist only between categories. Differences between items within a category are noise** (for example, the cross-sell scores below differ by about 1%).

## 5. Cross-sell and upsell (rule-based, `crosssell.py`)

- **Cross-sell:** with no basket data, we use a documented business assumption of complementary categories (Electronics ↔ Home/Sports, Home ↔ Electronics/Health, Sports ↔ Health/Clothing, and so on), ranked by net profit per order × (1 − return rate). Top 5 per item are stored. Within-category score differences are small, so the category rule is what drives the list.
- **Upsell:** the same product at the next **price tier** (quartiles of unit price within the category). The evidence supports this: margin (about 39.5%), return rate (about 10%) and rating (about 3.0) are flat across tiers, while net profit per order rises steadily (Electronics: $55 → $149 → $243 → $338). Moving a customer up one tier is therefore pure profit gain with no measured downside.

## 6. Problems encountered and fixes

| Problem | Resolution |
|---|---|
| No collaborative signal (one order per customer) | Reframed as cold-start, profile-based ranking and measured honestly against baselines. |
| Popularity "beats" random only by an artifact (items per category) | Documented in section 4.1. We do not claim it as a learned preference. |
| Gross profit would overstate value (returns/cancellations) | Defined `net_profit_per_order` (profit only on Completed orders) and used it for ranking and evaluation. |
| Possible leakage from item statistics | All item stats and models use only pre-2025-11-01 data. Evaluation uses only later orders. |
| Co-purchase lift is impossible | Replaced with an explicit category-complement rule, stated as an assumption. |
| Trained models were not saved (evaluation fitted them in memory and discarded them) | Moved fitting into `fit_models()` in `models.py` and added `train.py`, which saves the models. Evaluation results are unchanged. The saved format was later changed to numpy-only inference classes (see [phase-6-api.md](phase-6-api.md), section 4). |
| Slow logistic regression on 700k rows | Trained on a 250k-row sample (not tested with the full set). Naive Bayes, which uses all rows, reaches the same conclusion. |
| Fraud and rating signals are flat across items | Kept as optional penalty terms in the re-ranker. Their effect is near zero, and we say so. |

## 7. What this means for the showcase

1. **Be upfront:** the data is synthetic and carries no individual taste, so personalization lift is statistically zero. We prove it with a paired test and report that as a finding.
2. **The value is in the business layer:** the profit-aware re-ranker (+22% profit for about the same hit rate), upsell by price tier, and category-level cross-sell.
3. The UI's profit slider (λ) will show this trade-off live, using the numbers above.

## 8. Next steps

Phase 3: customer intelligence (RFM or value segmentation, high-value customers, behavior, CLV). Note that with one order per customer, true RFM and lifetime value are not estimable, so this phase will use the profile attributes and loyalty score and state the limitation.
