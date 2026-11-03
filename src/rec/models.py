"""Recommenders. Every model returns a relevance matrix of shape (n_rows, n_items), columns ordered like `items`."""
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from rec.features import CUSTOMER_FEATURES


class Base:
    name = "base"

    def __init__(self, items: list[str]):
        self.items = items
        self.idx = {k: i for i, k in enumerate(items)}

    def y(self, df: pl.DataFrame) -> np.ndarray:
        return df["item_key"].replace_strict(self.idx, return_dtype=pl.Int32).to_numpy()


class RandomModel(Base):
    name = "random"

    def fit(self, train):
        return self

    def score(self, df):
        return np.full((df.height, len(self.items)), 1 / len(self.items))


class Popularity(Base):
    """Global share of completed purchases in the training period."""
    name = "popularity"

    def fit(self, train):
        pos = train.filter(pl.col("is_positive_purchase"))
        counts = np.bincount(self.y(pos), minlength=len(self.items)).astype(float)
        self.p = counts / counts.sum()
        return self

    def score(self, df):
        return np.tile(self.p, (df.height, 1))


class Trending(Base):
    """Completed purchases in the last 90 days of training, so recent demand counts more."""
    name = "trending_90d"

    def fit(self, train):
        pos = train.filter(pl.col("is_positive_purchase"))
        recent = pos.filter(pl.col("order_date") >= pos["order_date"].max() - pl.duration(days=90))
        counts = np.bincount(self.y(recent), minlength=len(self.items)).astype(float)
        self.p = counts / counts.sum()
        return self

    def score(self, df):
        return np.tile(self.p, (df.height, 1))


class SegmentPopularity(Base):
    """Popularity within (customer_segment, country), shrunk towards the global share."""
    name = "segment_popularity"

    def __init__(self, items, keys=("customer_segment", "country"), shrink=500.0):
        super().__init__(items)
        self.keys, self.shrink = list(keys), shrink

    def fit(self, train):
        pos = train.filter(pl.col("is_positive_purchase"))
        k = len(self.items)
        g = np.bincount(self.y(pos), minlength=k).astype(float)
        self.global_p = g / g.sum()
        self.table = {}
        pos = pos.with_columns(pl.Series("_y", self.y(pos)))
        for key, grp in pos.group_by(self.keys):
            c = np.bincount(grp["_y"].to_numpy(), minlength=k).astype(float)
            self.table[key] = (c + self.shrink * self.global_p) / (c.sum() + self.shrink)
        return self

    def score(self, df):
        keys = df.select(self.keys).rows()
        return np.vstack([self.table.get(tuple(r), self.global_p) for r in keys])


class ProfileNB(Base):
    """Naive Bayes over customer/context features: P(item | profile)."""
    name = "profile_naive_bayes"

    def fit(self, train):
        pos = train.filter(pl.col("is_positive_purchase"))
        self.enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
        X = self.enc.fit_transform(pos.select(CUSTOMER_FEATURES).to_numpy())
        self.min_cat = [len(c) for c in self.enc.categories_]
        self.nb = CategoricalNB(alpha=1.0, min_categories=self.min_cat)
        self.nb.fit(X, self.y(pos))
        return self

    def score(self, df):
        X = self.enc.transform(df.select(CUSTOMER_FEATURES).to_numpy())
        X = np.where(X < 0, 0, X)
        return self.nb.predict_proba(X)


class ProfileLR(Base):
    """Multinomial logistic regression over one-hot customer/context features."""
    name = "profile_logreg"

    def __init__(self, items, sample=250_000, seed=42):
        super().__init__(items)
        self.sample, self.seed = sample, seed

    def fit(self, train):
        pos = train.filter(pl.col("is_positive_purchase"))
        if pos.height > self.sample:
            pos = pos.sample(self.sample, seed=self.seed)
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        X = self.enc.fit_transform(pos.select(CUSTOMER_FEATURES).to_numpy())
        self.lr = LogisticRegression(C=0.5, max_iter=200)
        self.lr.fit(X, self.y(pos))
        return self

    def score(self, df):
        return self.lr.predict_proba(self.enc.transform(df.select(CUSTOMER_FEATURES).to_numpy()))


class Hybrid(Base):
    """Convex blend of a profile model and popularity (the cold-start fallback)."""
    name = "hybrid"

    def __init__(self, items, profile: Base, pop: Base, alpha=0.5):
        super().__init__(items)
        self.profile, self.pop, self.alpha = profile, pop, alpha

    def fit(self, train):
        return self

    def score(self, df):
        return self.alpha * self.profile.score(df) + (1 - self.alpha) * self.pop.score(df)


def fit_models(train: pl.DataFrame, items: list[str]) -> dict[str, Base]:
    pop = Popularity(items).fit(train)
    models: dict[str, Base] = {
        "random": RandomModel(items).fit(train),
        "popularity": pop,
        "trending_90d": Trending(items).fit(train),
        "segment_popularity": SegmentPopularity(items).fit(train),
        "profile_naive_bayes": ProfileNB(items).fit(train),
        "profile_logreg": ProfileLR(items).fit(train),
    }
    models["hybrid(logreg+popularity)"] = Hybrid(items, models["profile_logreg"], pop, alpha=0.5)
    return models


def business_rerank(rel: np.ndarray, value: np.ndarray, lam: float, penalty: np.ndarray | None = None) -> np.ndarray:
    """final = (1-lam) * relevance/max + lam * value/max - penalty.

    `value` is the expected net profit per order of each item, `penalty` an optional per-item risk deduction
    (e.g. fraud or return risk) on the same 0..1 scale.
    """
    r = rel / rel.max(axis=1, keepdims=True)
    v = value / value.max()
    out = (1 - lam) * r + lam * v[None, :]
    if penalty is not None:
        out = out - penalty[None, :]
    return out
