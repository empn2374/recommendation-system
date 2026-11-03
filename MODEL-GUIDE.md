# Beginner's Guide to Every Model in This Project

This guide explains what each method does, why we used it, what it learns, where to find it, and what the results mean.

Small numerical examples are imaginary unless marked as project results.

## 1. Start with the purpose, not the algorithm name

The main recommendation question is:

> Given a customer's profile and browsing/order context, which five of our 48 products should we recommend?

There is also a separate customer-analysis question:

> Can we find meaningful customer groups or predict which customers will generate high value?

These are different tasks. The customer-analysis models do not generate the main product recommendations.

### Important data limitations

- The dataset has 1,000,123 order rows.
- After creating customer keys, practically every customer has one order.
- The usable catalogue contains 48 product names.
- Customer attributes show very little relationship with the purchased product.
- The data is synthetic; having many rows does not guarantee useful patterns.

We therefore compare simple approaches before claiming that personalization works.

## 2. Basic terms with one example

Imagine a historical row:

| Age | Membership | Country | Device | Purchased product |
|---|---|---|---|---|
| 43 | Regular | United States | Mobile | Wireless mouse |

- **Feature:** an input, such as age group, country, or device.
- **Target/label:** the answer to predict, here the purchased product.
- **Training:** learning from examples where both inputs and answers are available.
- **Prediction/inference:** using the learned information for a profile whose answer is not supplied.
- **Score:** a number used to rank a product.
- **Probability:** an estimated likelihood. It is not a guarantee.
- **Baseline:** a simple reference used to judge more complicated methods.
- **Classification:** predicting a category, such as one of 48 products.
- **Clustering:** grouping similar records without predefined answer labels.

The recommendation models estimate product scores, then the system selects the highest-ranked products.

### What feature preparation means

[features.py](src/rec/features.py) prepares 16 customer/context features:

Membership segment, country, gender, age band, loyalty band, tenure band, device, campaign, traffic source, coupon use, prior cart abandonment, weekend, hour band, month, session band, and pages band.

For example, age 43 becomes age-band label `"40"`, meaning ages 40-49. The person's age is not changed to 40 or 140.

Other numerical inputs are grouped into bands too. This reduces detail and lets similar values share evidence, although it can also hide differences within a band.

The file also creates a product summary:

- Average unit price: average price recorded for one unit across the training orders.
- Return rate: fraction of that product's orders marked Returned.
- Net profit per order: average recorded profit, with non-completed orders contributing zero.
- Popularity and recent-order shares.
- Ratings, sentiment shares, and other business statistics.

Example: if three product orders have completed profit $8, completed profit $12, and a returned order, the ranking value is `(8 + 12 + 0) / 3 = $6.67`.

This is a simplified scoring convention, not complete refund/return-cost accounting.

The profile models use the customer features. The business ranking layer uses product summaries.

## 3. Where training happens

Start with [train.py](src/rec/train.py), function `main`, and then [models.py](src/rec/models.py), function `fit_models`.

1. Load the prepared model table.
2. Select orders before 2025-11-01.
3. Create the ordered list of 48 products.
4. Fit the recommendation methods.
5. Save original models locally.
6. Convert them into small serving versions and save those too.

The training period contains 871,769 total rows. Methods that learn purchase preferences filter this period to Completed orders. Logistic regression further limits its fitting sample to 250,000 completed purchases.

[evaluate.py](src/rec/evaluate.py) fits the models again and tests them on 89,889 later completed purchases.

Why split by date? We learn from earlier purchases and check against later ones, rather than using future purchases to construct the recommendation model. Product summaries used for recommendation also come from the training period.

## 4. RandomModel: the equal-score reference

**Code:** [models.py](src/rec/models.py), class `RandomModel`, starting at line 22.

### Worked example: input to output

Imagine there are only three products: Mouse, Keyboard, and Lamp.

1. Receive a customer profile.
2. Do not inspect the customer's features or past purchases.
3. Give each product score `1 / 3`.
4. Return `[0.3333, 0.3333, 0.3333]`.

All three products tie. If we ask for two recommendations, the ranking code chooses two tied products; the model itself does not randomly draw them.

**What is stored after training?** No learned preferences; only the product list inherited from the base class.

### Why we included it

To establish a reference representing no useful knowledge.

If a uniformly random five-product list is chosen from 48 products, the chance of including a particular purchased product is `5 / 48`, or about 10.4%.

### What the actual code does

```python
def fit(self, train):
    return self

def score(self, df):
    return np.full((df.height, len(self.items)), 1 / len(self.items))
```

- `fit` does not learn anything.
- Every product receives score `1 / 48`.
- The returned matrix has one row per profile and one column per product.

**Important limitation:** this implementation does not draw a fresh random recommendation list. Equal scores are resolved by the ranking code's tie behavior. Its recorded result should be described as an equal-score reference, not a rigorous randomized experiment.

### What to say

> We included a no-information reference so we could judge whether other methods provided useful improvement.

## 5. Popularity: recommend overall best sellers

**Code:** [models.py](src/rec/models.py), class `Popularity`, starting at line 32.

### Worked example: from purchases to recommendations

Training purchases: `Mouse, Mouse, Lamp, Keyboard, Mouse`.

1. Count: Mouse = 3, Lamp = 1, Keyboard = 1.
2. Total purchases = 5.
3. Calculate shares: Mouse = `3/5`, Lamp = `1/5`, Keyboard = `1/5`.
4. Save those shares.
5. For any new customer, return Mouse = 0.6, Lamp = 0.2, Keyboard = 0.2.
6. Rank Mouse first. Lamp and Keyboard tie.

**What is stored after training?** One probability per product, in `self.p`. Prediction is a lookup of this stored array, not recounting the dataset.

### Why we used it

- No customer history is required.
- It works for new customers.
- It is inexpensive and easy to explain.
- Personalized methods should justify their complexity by improving on it.

### What fitting does

1. Keep Completed orders from the training period.
2. Count purchases for each product.
3. Divide each count by the total completed-purchase count.
4. Store the resulting product shares.

Example:

| Product | Completed purchases | Stored share |
|---|---:|---:|
| Mouse | 500 | 50% |
| Headphones | 300 | 30% |
| Keyboard | 200 | 20% |

The ranking is Mouse, Headphones, Keyboard.

### What prediction does

It returns the same stored shares for every profile. Age, country, and device do not change the ranking.

### What it cannot do

It cannot learn personal taste. Popularity can also favor products because of how the dataset was generated or which products were exposed, rather than genuine preference.

### What to say

> Popularity is our main benchmark. It recommends the products most frequently purchased successfully.

## 6. Trending: recommend recent best sellers

**Code:** [models.py](src/rec/models.py), class `Trending`, starting at line 46.

### Worked example: why the time window changes the answer

Suppose training data contains:

| Product | Older completed purchases | Completed purchases in the latest 90-day window |
|---|---:|---:|
| Mouse | 90 | 10 |
| Lamp | 10 | 30 |

1. Ignore the older column for this method.
2. Count recent purchases: Mouse = 10, Lamp = 30.
3. Divide by the recent total of 40.
4. Save Mouse = 25%, Lamp = 75%.
5. Recommend Lamp first for every customer.

Overall popularity would favor Mouse, with 100 purchases versus Lamp's 40. Trending favors Lamp because it only uses the recent window.

**What is stored after training?** One recent-purchase share per product, in `self.p`.

### Why we used it

Overall popularity may be outdated. We tested whether recent demand produces better recommendations.

### How it works

1. Keep Completed training purchases.
2. Find the latest date among those purchases.
3. Keep purchases within 90 days of that date.
4. Count products and calculate shares.

Example: a mouse led sales over the full year, but headphones led in the latest 90 days. Popularity might recommend the mouse first; Trending might recommend headphones first.

### Important distinctions

- It counts recent demand; it does not forecast future trends.
- Recent purchases are not given continuously decaying weights. Orders outside the window are excluded.
- Everyone receives the same ranking.
- It uses the training period, not the current calendar date.

### What to say

> Trending tests whether recent completed purchases are more informative than the whole training history.

## 7. SegmentPopularity: recommend group best sellers

**Code:** [models.py](src/rec/models.py), class `SegmentPopularity`, starting at line 61.

### Worked example: how much should we trust a group?

Imagine only Mouse and Lamp exist:

- Overall shares: Mouse 60%, Lamp 40%.
- A group has 100 purchases: Mouse 20, Lamp 80.

The configured smoothing strength is 500:

```text
Mouse = (20 + 500 x 0.60) / (100 + 500) = 53.33%
Lamp  = (80 + 500 x 0.40) / (100 + 500) = 46.67%
```

Although Lamp leads inside this small group, overall evidence still puts Mouse first.

Now imagine the group has 10,000 purchases with the same 20%/80% split:

```text
Mouse = (2,000 + 300) / 10,500 = 21.90%
Lamp  = (8,000 + 200) / 10,500 = 78.10%
```

The larger group now has enough evidence to dominate the global fallback.

**What is stored after training?** A dictionary of group probability arrays, plus a global probability array for unseen groups.

### Why we used it

We do not have meaningful individual histories, but customers sharing a membership level and country might have different preferences.

### How it works

The group key is `(customer_segment, country)`.

For each group, the code counts completed purchases and blends them with global popularity.

For one product:

```text
group score =
    (group purchase count + 500 x global product share)
    / (total group purchases + 500)
```

The value 500 is a smoothing strength. It acts like adding 500 purchases distributed according to overall popularity.

Example:

- A group has 10 purchases; 4 are mice.
- Globally, mice represent 20% of completed purchases.
- Mouse score: `(4 + 500 x 0.20) / (10 + 500) = 104 / 510`, about 20.4%.

Without smoothing, two or three unusual purchases could strongly distort a small group's ranking.

### Prediction and unknown groups

The method looks up the customer's group. If the group was never seen, it uses overall popularity.

### Limitation

This is group-level personalization, not individual preference learning.

### What to say

> Segment popularity checks for group preferences while protecting small groups from unreliable purchase counts.

## 8. ProfileNB: Categorical Naive Bayes

**Code:** [models.py](src/rec/models.py), class `ProfileNB`, starting at line 86. `CategoricalNB` is constructed at line 95.

### Worked example: combining two clues

Use only two products and two features to keep the arithmetic small. Suppose the learned probability tables contain:

| Learned quantity | Mouse | Lamp |
|---|---:|---:|
| Overall product probability | 0.60 | 0.40 |
| Probability of Mobile among this product's buyers | 0.80 | 0.30 |
| Probability of Regular membership among this product's buyers | 0.50 | 0.75 |

For a Mobile, Regular customer:

```text
Mouse evidence = 0.60 x 0.80 x 0.50 = 0.24
Lamp evidence  = 0.40 x 0.30 x 0.75 = 0.09
```

These are not yet normalized probabilities. Divide by their total:

```text
Total = 0.24 + 0.09 = 0.33
Mouse estimate = 0.24 / 0.33 = 72.73%
Lamp estimate  = 0.09 / 0.33 = 27.27%
```

Mouse ranks first. The real model combines all 16 features across 48 products.

For numerical stability, the implementation adds logarithms of probabilities instead of directly multiplying many small numbers. Normalization then produces the final probabilities.

**What is stored after training?** Category mappings, product prior probabilities, and feature-category probability tables for each product.

### Why we used it

We wanted a fast statistical test of whether the 16 categorical profile/context features help predict products.

### How to think about it

Suppose we know a customer is in age band 40, uses Mobile, and has Regular membership.

For each candidate product, Naive Bayes considers:

- How common the product is overall.
- How often buyers of that product have age band 40.
- How often buyers of that product use Mobile.
- How often buyers of that product have Regular membership.

It combines those clues and normalizes the results into probabilities across products.

### What "naive" means

It assumes feature clues are independent once the product is known. Real features can be related, so this is a simplifying assumption, not a guaranteed fact.

### What fitting does

1. Keep completed training purchases.
2. Convert feature categories into integer codes using `OrdinalEncoder`.
3. Learn product frequencies and feature-category frequencies within each product.
4. Store the learned probability tables.

Those integer codes identify categories. This categorical model does not interpret code 2 as twice as much as code 1.

`alpha=1.0` adds smoothing so unseen feature/product combinations do not automatically receive zero probability.

### What prediction does

It encodes a profile, combines learned probabilities, and returns a probability for each product.

### Unknown input behavior

The current code maps unknown feature categories to category index 0. That is a fallback to an existing category, not a separately learned "unknown" category. It is a limitation to keep in mind.

### Strengths and limitations

- Fast, compact frequency-based training.
- Suitable for categorical inputs.
- Cannot fully represent dependencies between features.
- Cannot learn useful preferences if the input features contain no preference signal.

### What to say

> Naive Bayes is a lightweight way to combine profile clues and test whether those clues predict the purchased product.

## 9. ProfileLR: multinomial logistic regression

**Code:** [models.py](src/rec/models.py), class `ProfileLR`, starting at line 105. `LogisticRegression` is constructed at line 119.

### Worked example: turning learned weights into probabilities

Suppose Mobile and Regular are the active indicators for a simplified profile:

| Learned value | Mouse | Lamp |
|---|---:|---:|
| Starting offset | 0.20 | 0.10 |
| Mobile weight | 0.50 | -0.20 |
| Regular weight | -0.10 | 0.30 |

1. Add the offset and active feature weights:

```text
Mouse score = 0.20 + 0.50 - 0.10 = 0.60
Lamp score  = 0.10 - 0.20 + 0.30 = 0.20
```

2. Turn scores into positive values with the exponential function, written `exp`:

```text
exp(0.60) is approximately 1.822
exp(0.20) is approximately 1.221
```

3. Divide by their sum. This is softmax:

```text
Mouse probability = 1.822 / (1.822 + 1.221) = 59.87%
Lamp probability  = 1.221 / (1.822 + 1.221) = 40.13%
```

4. Rank Mouse first.

Where did the weights come from? During training, the optimizer changes them to reduce prediction error plus the regularization penalty. If the true product is Lamp, the objective favors increasing Lamp's probability for that example; all examples jointly determine the final weights.

The real model uses every active profile indicator and produces 48 probabilities.

**What is stored after training?** One-hot category mappings, a weight matrix, and one starting offset per product. It does not keep a copy of every customer to search at prediction time.

### Why we used it

It provides a second profile-based approach that learns feature weights rather than relying on Naive Bayes's frequency-combination assumptions.

It is a standard, relatively lightweight classification method.

### What "multinomial" means

There are multiple possible answers: one of 48 products.

Despite the name "regression," this model performs classification, not prediction of a price or other continuous number.

### One-hot encoding

Membership labels become indicator columns:

| Membership | Regular indicator | Premium indicator | VIP indicator |
|---|---:|---:|---:|
| Regular | 1 | 0 | 0 |
| VIP | 0 | 0 | 1 |

The code uses a sparse representation, which stores nonzero entries efficiently instead of filling memory with zeros.

### How training works

1. Keep completed purchases.
2. Sample at most 250,000 rows using a fixed seed.
3. Encode the 16 features into indicator columns.
4. Start with feature weights.
5. Compare predictions against actual purchased products.
6. Adjust weights to reduce prediction error.
7. Stop when the optimizer converges or reaches its iteration limit.

### How prediction works

Each product gets a score:

```text
product score = starting offset + sum of weights for active profile features
```

A softmax calculation turns the 48 scores into probabilities that sum to one.

Example: if Mobile and age-band-40 indicators are active, their learned weights contribute to every product's score. Different products have different weights.

### Settings in this project

- `sample=250_000`: limits the fitting dataset.
- `seed=42`: makes sample selection reproducible.
- `C=0.5`: controls regularization. Regularization discourages overly large weights; smaller C means stronger regularization.
- `max_iter=200`: maximum optimizer iterations, not 200 seconds or necessarily 200 full dataset passes.
- `handle_unknown="ignore"`: unseen category indicators contribute no active column.

### Limitations

This setup learns additive feature contributions. It does not explicitly create complex feature interactions. Sampling also means it does not fit every completed training purchase.

### What to say

> Logistic regression learns how strongly each profile feature relates to each product, using a manageable sample and regularization.

## 10. Hybrid: blend two fitted opinions

**Code:** [models.py](src/rec/models.py), class `Hybrid`, starting at line 127.

### Worked example: how blending can change the winner

| Product | Logistic regression | Popularity | 50/50 Hybrid |
|---|---:|---:|---:|
| Mouse | 30% | 60% | 45% |
| Lamp | 50% | 20% | 35% |
| Keyboard | 20% | 20% | 20% |

1. Ask the fitted logistic model for its probability array.
2. Ask the fitted popularity model for its probability array.
3. Multiply each array by 0.5 and add them.
4. Rank the combined values.

Logistic regression alone prefers Lamp, but the Hybrid prefers Mouse after considering overall popularity.

**What is stored?** References to the fitted components and their mixture weight. There is no additional learned weight matrix.

### Why we used it

Profile predictions may be weak or unreliable. Popularity provides a simple anchor.

### How it works

```text
hybrid probability =
    0.5 x logistic-regression probability
    + 0.5 x popularity probability
```

Example for a mouse:

- Logistic regression: 8%.
- Popularity: 4%.
- Hybrid: 6%.

This calculation is repeated for every product.

### Does it train again?

No. Its `fit` method does nothing. The component methods have already been fitted. The 50/50 weight is configured in the code, not learned by another optimization step.

### Important distinction

The Hybrid combines profile relevance and popularity. The profit slider is a separate re-ranking step.

### What to say

> The Hybrid balances profile-based estimates with overall popularity rather than trusting personalization alone.

## 11. Customer analysis: MiniBatch K-Means

**Code:** [customers.py](src/rec/customers.py), candidate fits at line 101 and final fit at line 106.

### Worked example: moving the group centers

To see the mechanics, temporarily use only one feature: quantity purchased.

```text
Records: 1, 2, 8, 9
Requested groups: K = 2
Starting centers: 1 and 8
```

1. Assign quantities 1 and 2 to center 1 because it is closer.
2. Assign quantities 8 and 9 to center 8.
3. With ordinary K-Means, update centers to the group averages:
   - First center: `(1 + 2) / 2 = 1.5`.
   - Second center: `(8 + 9) / 2 = 8.5`.
4. Assign records again using the new centers.
5. Stop when the assignments/centers stabilize according to stopping criteria.

MiniBatch K-Means approximates these updates using successive batches and accumulated counts, rather than recomputing full-group averages every time.

Your real setup compares distance across nine standardized features, not quantity alone. Distance combines differences across those feature values.

**What is learned?** Group centers and record assignments. No purchased-product label or high-value label is used to fit these centers.

### Why we used it

To investigate whether similar customer/order records form meaningful groups without predefined labels.

This is exploration, not product prediction.

### Inputs

Nine numerical features: age, loyalty, tenure, session duration, pages visited, fraud score, quantity, discount, and log unit price.

Log price compresses the numerical scale of price. Standardization then puts the different feature units on comparable scales.

Without standardization, a feature measured in thousands of days could dominate one measured in tens of years simply because its numbers are larger.

### How K-Means works

1. Choose K starting centers.
2. Assign records to their nearest center.
3. Update centers toward their assigned records.
4. Repeat until stopping.

K is the requested number of groups.

### What MiniBatch changes

Center updates use smaller batches, making fitting more practical for large datasets. The configured batch size is 20,000.

The code tries K values 3, 4, 5, 6, and 8, then explicitly chooses K=4 for the final fit. It does not automatically pick the highest silhouette score.

The final fit uses `n_init=5`, meaning multiple initialization attempts.

### How the groups are checked

Silhouette compares similarity within a group with separation from other groups:

- Near 1: strongly separated groups.
- Near 0: weak separation.
- Negative: some records may fit another group better.

The code computes this score on a 20,000-record sample for each candidate K. This calculation can be expensive independently of fitting.

### Project finding

Silhouette values were around 0.065-0.070. The groups were weakly separated, not compelling natural customer types.

Cluster names are assigned after examining average profit. Those names are descriptive summaries, not predicted customer classes.

### What to say

> MiniBatch K-Means helped us test for natural groups. The weak separation meant we avoided claiming that the data contained distinct customer personas.

## 12. Customer analysis: Histogram Gradient Boosting

**Code:** [customers.py](src/rec/customers.py), fitted at line 89.

### Worked example: improving a high-value prediction

Suppose 10% of training records are high-value. An initial constant prediction can start around that overall rate.

1. Compare current probabilities against the known high-value/not-high-value labels.
2. Calculate how the model's internal scores should change to reduce classification loss.
3. Fit a small decision tree to those correction signals.
4. Each tree leaf contributes a positive or negative adjustment to the internal score.
5. Add a scaled adjustment, then fit another tree using the updated errors.
6. Convert the final combined score into a probability.

An illustrative sequence for one record might be:

```text
Initial probability:           10%
After one tree's adjustment:  approximately 12%
After later adjustments:     approximately 15%
```

Those percentages are not added directly. Trees adjust an internal score, and the score is converted into probability.

A tree might split on tenure, then loyalty. Another might split on session duration. These are illustrative questions, not rules verified from your fitted model.

For the histogram optimization, numerical values are first mapped into bins. Candidate splits can then be checked using aggregated bin statistics instead of every distinct numerical value.

**What is learned?** A sequence of trees, their split thresholds and leaf contributions, and the information needed to map inputs into bins. These work together to predict high-value probability.

### Why we used it

To test whether profile/context attributes can identify high-value customers without using the order's price or quantity as prediction inputs.

Unlike the additive logistic model, decision-tree boosting can capture nonlinear relationships and combinations of features.

### Target and inputs

- Target: whether net profit reaches the top-decile threshold across all records.
- Inputs: age, loyalty, tenure, session duration, pages visited, and fraud score.

These inputs are not necessarily all available before a browsing session begins. "Profile-only" here means excluding purchase-value mechanics, not predicting before any interaction occurs.

### Decision trees and boosting

A decision tree splits records with learned questions such as whether a numerical value exceeds a threshold.

Boosting builds multiple small trees in sequence. New trees improve the combined model where the current predictions need correction.

The histogram implementation groups numerical values into bins to make finding splits more efficient.

### Training setup

- Sample 300,000 records.
- Randomly split 70% for training and 30% for testing.
- Preserve the high-value proportion with stratification.
- Fit with a maximum of 150 boosting iterations.

This random split differs from the recommendation models' chronological evaluation.

### Project finding: AUC 0.504

AUC measures how well the model ranks positive examples above negative ones:

- About 0.5: chance-level discrimination.
- 1.0: perfect discrimination.

0.504 is not 50.4% classification accuracy.

The result suggests little useful high-value signal in these inputs, even with a more flexible model. It does not prove that every possible future model must fail.

### What to say

> Gradient boosting tested whether more complex profile relationships predicted high value. Its near-chance result supported the finding that value mostly comes from purchase mechanics.

## 13. Business re-ranking: not a trained model

**Code:** [models.py](src/rec/models.py), `business_rerank`, starting at line 156; live adjustments are in [serving.py](src/rec/serving.py), method `_final`.

### Why we use it

Relevance is not the only business goal. We may also care about profit, returns, and risk.

### How it works

Relevance and profit are each divided by their maximum before blending:

```text
final score =
    (1 - profit weight) x scaled relevance
    + profit weight x scaled profit
    - optional penalties
```

Live serving additionally supports a positive-sentiment boost.

Example:

| Product | Scaled relevance | Scaled profit |
|---|---:|---:|
| A | 1.0 | 0.4 |
| B | 0.9 | 1.0 |

At profit weight 0, A wins. At profit weight 0.25:

- A: `0.75 x 1.0 + 0.25 x 0.4 = 0.85`.
- B: `0.75 x 0.9 + 0.25 x 1.0 = 0.925`.

B moves above A because of business value.

These final scores are ranking scores, not purchase probabilities.

No weights are learned here. The profit/risk controls are policy settings.

## 14. Other rules and formulas, not trained algorithms

### Value tiers

[customers.py](src/rec/customers.py) groups records by completed-order profit, including a top-10% tier. These are threshold-based labels, not K-Means predictions.

### Lifetime-value scenarios

[customers.py](src/rec/customers.py), formula at line 166:

```text
scenario value =
    recorded completed-order profit
    + mean profit per order x assumed annual orders
      x (retention + retention^2 + retention^3)
```

The base assumptions are 80% annual retention, one order per year, and three future years.

Example: recorded profit $50 and assumed mean profit $100:

```text
$50 + $100 x (0.8 + 0.64 + 0.512) = $245.20
```

This is an assumption-based scenario, not a learned promise. The project lacks the repeat-purchase history needed to estimate reliable individual purchase frequency and retention.

### Cross-sell and upsell

[crosssell.py](src/rec/crosssell.py) uses explicit complementary-category rules and price-tier statistics.

Cross-sell does not mean the data proved these products are bought together. Upsell is a price-tier suggestion, not a verified premium variant catalogue or a measured customer acceptance probability.

## 15. What the evaluation numbers mean

**Evidence:** [evaluation.txt](reports/evaluation.txt).

### Hit@5

If the customer bought a keyboard and our list is Mouse, Keyboard, Lamp, Bottle, Headphones, that is a hit.

If the actual purchase is Shoes, that is a miss.

Hit@5 is the proportion of test purchases whose product appears anywhere in the first five recommendations.

If 1,112 out of 10,000 test purchases are hits, Hit@5 is 11.12%.

It is not website conversion, revenue uplift, or the probability of each listed product.

### Recorded project results

| Method | Hit@5 |
|---|---:|
| Equal-score reference | 10.52% |
| Popularity | 11.12% |
| Trending | 11.05% |
| Segment popularity | 11.21% |
| Naive Bayes | 11.16% |
| Logistic regression | 11.18% |
| Hybrid | 11.22% |

Logistic regression minus popularity has a paired 95% confidence interval that includes zero. The evaluation does not establish a clear personalization improvement. The Hybrid's small numerical advantage should not be sold as a proven improvement either.

### Other metrics

- **NDCG:** rewards putting the purchased product nearer the top, rather than merely including it.
- **Coverage:** the fraction of the 48 products appearing across all recommendation lists.
- **Captured profit per customer:** average actual test-purchase profit counted only when the purchased product is in the list; misses contribute zero.
- **Average list item value:** average training-derived profit value of recommended products. This is different from captured test profit.
- **Confidence interval:** a range expressing sampling uncertainty around an estimate.

At profit weight 0.25, Hybrid captured profit rose from $14.96 to $18.29, approximately 22%, while Hit@5 changed from 11.22% to 11.16%.

This is an offline trade-off. It does not show that customers purchased more, changed products, or generated 22% more real-world profit because of the recommendations. That needs an online experiment.

## 16. Are these methods lightweight? How long do they take?

Yes, relative to large neural models, these methods are CPU-friendly. But "lightweight" does not mean every step is instant.

**The project does not record model-training durations. Do not quote an exact number of seconds or minutes as a measured result.**

| Operation | Work involved |
|---|---|
| Equal-score reference | No fitting |
| Popularity and Trending | Filtering and counting |
| Segment popularity | Grouped counting and smoothing |
| Naive Bayes | Encoding and learning frequency tables |
| Logistic regression | Iterative fitting on at most 250,000 purchases |
| Hybrid | Combining fitted components; no separate optimization |
| MiniBatch K-Means | Repeated center updates across large data |
| Silhouette checks | Potentially expensive distance comparisons on sampled records |
| Gradient boosting | Multiple tree-building iterations |
| CLV and business re-ranking | Formulas; no model fitting |

CPU, memory, disk speed, iteration behavior, and other machine activity determine actual duration.

### Measuring on your machine

Run these from the project root after features have been prepared.

To measure recommendation loading, fitting, conversion, and saving:

```powershell
$env:PYTHONPATH = "src"
Measure-Command {
    .\.venv\Scripts\python.exe -m rec.train
    if ($LASTEXITCODE -ne 0) { throw "Recommendation training failed." }
}
```

To measure the complete customer-analysis script:

```powershell
$env:PYTHONPATH = "src"
Measure-Command {
    .\.venv\Scripts\python.exe -m rec.customers
    if ($LASTEXITCODE -ne 0) { throw "Customer analysis failed." }
}
```

These commands rebuild saved outputs. They do not report isolated time for every individual model. The second command includes clustering checks, statistical analysis, and output generation.

Evaluation fits the recommendation models again, so running training followed by evaluation repeats fitting work.

### Training versus serving

Training happens offline. A website request uses saved models and does not train again.

Fast API response measurements must not be presented as training durations.

## 17. Lite models: the same learned information, smaller runtime

**Code:** [lite.py](src/rec/lite.py), function `to_lite`, starting at line 71.

The serving versions use NumPy calculations instead of requiring scikit-learn and SciPy at runtime.

| Training method | Serving version |
|---|---|
| Random, Popularity, Trending | `LiteConst` |
| Segment popularity | `LiteSegmentPopularity` |
| Naive Bayes | `LiteNaiveBayes` |
| Logistic regression | `LiteLogReg` |
| Hybrid | `LiteHybrid` |

The code copies learned distributions, category mappings, probability tables, coefficients, and offsets. It does not train a new "Lite algorithm."

The API serves six recommendation methods; the equal-score reference is kept for offline comparison.

Customer-analysis models are used to generate summaries, not served as live recommendation models.

## 18. Why not use more advanced algorithms?

- Collaborative filtering normally benefits from repeated customer-item interactions. Here, almost every customer has one observation, giving it very little history to use.
- Co-purchase association rules require useful basket/grouping evidence that this dataset does not provide.
- Neural embeddings and deep learning add cost and complexity without evidence that the available features contain enough preference signal.

This does not mean advanced algorithms are always unsuitable. It means this dataset does not justify claiming that they would solve the missing-signal problem.

## 19. A presentation-ready summary

> We first inspected the data and found almost no repeat purchases or strong customer-preference patterns. We used popularity as a benchmark, then tested recent popularity, group popularity, Naive Bayes, and logistic regression. The Hybrid combined profile estimates with popularity. Personalization did not show a clear improvement, so we reported that limitation honestly. We also tested customer clustering and high-value prediction, but those signals were weak. The practical business layer combines recommendation relevance with profit and explicit policy rules. Training is offline; the API uses small saved models for fast scoring.

## 20. Suggested reading order in the code

1. [features.py](src/rec/features.py): inputs and product summaries.
2. [models.py](src/rec/models.py), `fit_models` at line 142: all recommendation methods together.
3. [models.py](src/rec/models.py), each class's `fit` and `score`: learning versus prediction.
4. [train.py](src/rec/train.py): training and saving.
5. [evaluate.py](src/rec/evaluate.py): historical testing.
6. [customers.py](src/rec/customers.py): separate customer-analysis experiments.
7. [lite.py](src/rec/lite.py) and [serving.py](src/rec/serving.py): lightweight live scoring.

### Quick memory aid

| Method | Question |
|---|---|
| Popularity | What sells most overall? |
| Trending | What sells most recently? |
| Segment popularity | What sells in this customer's group? |
| Naive Bayes | What do separate profile clues suggest? |
| Logistic regression | What learned feature weights predict the product? |
| Hybrid | What if we average profile predictions and popularity? |
| K-Means | Which records are similar enough to group? |
| Gradient boosting | Can combinations of profile features predict high value? |
| CLV scenario | What future value follows from stated assumptions? |
| Business re-ranking | How should profit and policy change the product order? |
