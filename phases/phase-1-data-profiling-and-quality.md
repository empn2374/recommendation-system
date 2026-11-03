# Phase 1: Data Profiling, Quality and Preparation

**Status:** Profiling and cleaning done. **A strategy decision is needed before Phase 2** (see section 6).
**Code:** [ingest.py](../src/rec/ingest.py), [profile.py](../src/rec/profile.py), [clean.py](../src/rec/clean.py)
**Outputs:** `data/interim/raw.parquet`, `data/interim/clean.parquet`, [profile.txt](../reports/profile.txt), [quality.txt](../reports/quality.txt)

## 1. What was done

| Step | Command | Result |
|---|---|---|
| Environment | `.venv` with polars, pyarrow, numpy, scipy, scikit-learn | Python 3.13 |
| Ingest | `python -m rec.ingest` | 391 MB CSV → 81 MB Parquet (zstd). **1,000,123 rows × 62 columns**, matching the brief. |
| Profile | `python -m rec.profile` | Schema, nulls, cardinality, grain and key checks, distributions, consistency checks, ID stability |
| Signal test | Cramér's V (chi-square) of product vs. customer attributes | See section 3 |
| Clean | `python -m rec.clean` | 72 columns (62 original + 10 derived), all rows kept, quality flags added |

Run from the repo root with `PYTHONPATH=src`, for example `.venv\Scripts\python -m rec.ingest`.

## 2. Dataset at a glance

- **Period:** 2024-02-03 to 2026-02-02. Accounts created 2021-02 to 2026-01. No order precedes its account creation date.
- **Rows are order lines:** one product per row. 991,930 distinct `order_id`s, 0 fully duplicated rows.
- **Catalogue:** 5 categories, 22 sub-categories, **48 distinct product names**, 20 brands, 10 countries, 3 customer segments (Regular 60%, Premium 30%, VIP 10%).
- **Order outcomes:** Completed 70%, Returned 10%, Pending 10%, Cancelled 5%, Processing 5%.
- **Money (realized sales, i.e. Completed or Returned and Paid):** about $322.7M revenue and $128.8M profit, a 39.9% margin.
- **Nulls:** only `return_reason` (90%, null exactly when not returned), `coupon_code` (50%, null when no coupon) and `customer_feedback` (20%). Nothing else is missing.
- **Arithmetic is clean:** `total = unit_price × qty − discount`, `profit = total − cost`, and `margin% = profit / total` hold on every row. There are no negative or zero prices, quantities or profits, and ratings are all within 1 to 5.

## 3. Key findings (these change the project)

The data is **synthetic and almost entirely random**. The business columns carry little real structure, which limits what a recommender can learn.

| # | Finding | Evidence | Impact |
|---|---|---|---|
| F1 | **Practically no repeat customers.** | 991,945 distinct `customer_id` over 1,000,123 rows. After fixing ID collisions (F3), 1,000,121 of 1,000,122 customers have exactly 1 order. | **No purchase history per customer.** Classic collaborative filtering (a customer × product matrix) has nothing to learn from. |
| F2 | **`product_id` is not a product key.** | 753,516 distinct ids. 26% of ids map to more than one name, category and price. Only **48** distinct `product_name`s exist, and each name belongs to exactly one category. | The real catalogue is the 48 product names, not ~750k ids. Item key = `product_name`. |
| F3 | **`customer_id` collisions.** | About 8,100 ids are reused by different people (different name, age, country). | Customer key = `customer_id` + `customer_name`. |
| F4 | **Product choice is independent of customer attributes.** | Cramér's V of `product_name` vs. segment, country, gender, age, loyalty, device, campaign, month, hour is **0.006 to 0.008** (zero means independent). | Personalization from demographics has no real signal to find. |
| F5 | **All 48 products look statistically identical.** | Margin 39.4 to 39.6%, return rate 9.6 to 10.5%, rating 2.99 to 3.02, and fraud score about 50 for every product. Prices are uniform random within a **category** range: Clothing $15 to 150, Electronics $10 to 500, Health $10 to 200, Home $20 to 300, Sports $15 to 250. | Real differences exist only at **category** level (price range, average order value). Per-product "best seller" differences are noise. |
| F6 | **Customer, marketing and geography dimensions do not move outcomes.** | AOV, margin, return rate and rating are flat across segment, campaign, device and country. For example, VIP and Regular both have AOV of about $403 and a return rate of about 10%. | Segmentation and marketing insights will be mostly flat. We report that honestly. |
| F7 | **The only real mechanical relationships** are discount → margin (0%: 45.0%, 5%: 42.1%, 10%: 38.9%, 15%: 35.3%, 20%: 31.3%, 25%: 26.7%) and rating → sentiment (rating 1 or 2 = Negative, 3 = Neutral, 4 or 5 = Positive, with no exceptions). | Discount-vs-profit is a valid business insight. Sentiment adds no information beyond rating. |
| F8 | **Reference columns are inconsistent with the rows.** | `total_orders_by_customer` (1 to 50) contradicts the one-order reality. `product_rating_avg` has about zero correlation (−0.0004) with the row `rating`. | Treat these as noise or as profile attributes, not as ground truth. |
| F9 | **Status fields contradict each other.** | 23.3% of rows conflict, for example Completed with delivery Failed or Pending, Cancelled with payment Paid, or Pending with delivery Delivered. | Flagged as `flag_status_conflict`. Revenue is defined carefully (see section 5). |
| F10 | **Duplicated `order_id`s.** | 8,140 ids are reused (16,333 rows) with different contents. | Flagged as `flag_order_id_dup`. Rows are kept because they are distinct orders. |

## 4. Problems encountered and fixes

| Problem | Fix |
|---|---|
| `.crdownload` partial file earlier | You re-downloaded it. The full CSV (391 MB) was verified at 1,000,123 rows and 62 columns. |
| Polars `DuplicateError` in `clean.py` (a `strip_chars` on all string columns collided with the date conversion) | Split the transformation into two `with_columns` steps. |
| Windows console `UnicodeEncodeError` printing tables | `PYTHONIOENCODING=utf-8`. Reports are written to files as UTF-8. |
| `pip install` printed a pip-upgrade notice that PowerShell showed as an error | It is harmless (the packages installed). |
| Wide tables unreadable in the profile output | Wrote the output to `reports/profile.txt` and inspected it in sections. |

## 5. Cleaning rules and derived columns (`clean.py`)

All 1,000,123 rows are kept. Nothing is dropped, because dropping would silently remove 24.5% of the data. Issues are exposed as flags so each later phase can choose how to handle them.

- **Types:** `order_date` is a datetime and `account_creation_date` a date. The five Yes/No columns became booleans. All strings were trimmed.
- **Keys:** `customer_key` = `customer_id|customer_name`. `item_key` = `product_name`.
- **Nulls:** `return_reason` → "Not returned", `customer_feedback` → "No feedback", `coupon_code` → "No coupon". Each null had a clear meaning, so none needed imputation.
- **Outcome flags:** `is_returned`, `is_cancelled`, `is_payment_failed`.
- **Quality flags:** `flag_order_id_dup` (1.63%), `flag_customer_id_reused` (1.63%), `flag_status_conflict` (23.25%).
- **Business definitions:**
  - `is_realized_sale` = order status Completed or Returned **and** payment Paid (**800,111 rows, 80%**). Used for revenue and profit.
  - `is_positive_purchase` = order status Completed (**700,232 rows, 70%**). Used as the positive signal for recommendations. Returned counts as a negative signal.

## 6. Implications and decision needed

Because of F1, F4 and F5, a model cannot honestly be sold as "learns each customer's taste". Offline accuracy will be close to a popularity baseline, and any gain is noise. A recommender should still be built, but it should be framed around what the data supports.

**Recommended strategy (needs your approval):**

1. **Be transparent.** Document the synthetic nature and report that personalization lift is not detectable. We show the evidence (Cramér's V, a baseline comparison). That kind of analytical honesty is what the assignment grades ("analytical thinking and problem-solving").
2. **Item = 48 products** (optionally product × brand = 960 variants). Recommend by **business-aware ranking**: relevance (category affinity, a customer's current basket or session where available, popularity) combined with **profit, price tier, return risk and fraud risk**.
3. **Cold-start / profile-based recommender:** since every customer is effectively new, recommendations are driven by segment, country, age and loyalty features. We measure whether they beat plain popularity. We expect they will not, and we report it.
4. **Cross-sell and upsell** are built on category structure and price tiers instead of co-purchase. There is no basket data (one product per order and one order per customer), so co-purchase lift cannot be computed. We state that.
5. **Where we can show real value:** profit-aware re-ranking (discount and margin trade-off), return-aware and fraud-aware filtering, revenue-opportunity sizing, and the architecture. These do not depend on hidden customer taste.

**Alternative:** simulate the missing behavior (for example, generate a realistic purchase history). I would **not** recommend this because it would be fabricated and could mislead the showcase. If you want it for demonstration only, it must be clearly labeled as simulated.

## 7. Next steps

Phase 2 (feature engineering and recommendation design) once the strategy in section 6 is confirmed.
