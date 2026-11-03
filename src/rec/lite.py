"""Dependency-light inference classes (numpy only).

Training uses scikit-learn (models.py). For serving, each fitted model is converted into one of these classes so the
API needs neither scikit-learn nor scipy. `score(feats)` takes one profile dict (see serving.profile_to_features)
and returns a probability vector over the items.
"""
import numpy as np

from rec.features import CUSTOMER_FEATURES


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class LiteConst:
    """Popularity, trending and random: the same distribution for everyone."""

    def __init__(self, p: np.ndarray):
        self.p = np.asarray(p, dtype=float)

    def score(self, feats: dict) -> np.ndarray:
        return self.p


class LiteSegmentPopularity:
    def __init__(self, keys: list[str], table: dict, global_p: np.ndarray):
        self.keys, self.table, self.global_p = keys, table, np.asarray(global_p, dtype=float)

    def score(self, feats: dict) -> np.ndarray:
        return self.table.get(tuple(feats[k] for k in self.keys), self.global_p)


class LiteNaiveBayes:
    def __init__(self, categories: list[list[str]], feature_log_prob: list[np.ndarray], class_log_prior: np.ndarray):
        self.index = [{c: i for i, c in enumerate(cats)} for cats in categories]
        self.flp = [np.asarray(a, dtype=float) for a in feature_log_prob]
        self.prior = np.asarray(class_log_prior, dtype=float)

    def score(self, feats: dict) -> np.ndarray:
        jll = self.prior.copy()
        for j, name in enumerate(CUSTOMER_FEATURES):
            jll += self.flp[j][:, self.index[j].get(feats[name], 0)]  # unseen category falls back to index 0, as in training code
        return _softmax(jll)


class LiteLogReg:
    def __init__(self, categories: list[list[str]], coef: np.ndarray, intercept: np.ndarray):
        self.index, offset = [], 0
        for cats in categories:
            self.index.append({c: offset + i for i, c in enumerate(cats)})
            offset += len(cats)
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = np.asarray(intercept, dtype=float)

    def score(self, feats: dict) -> np.ndarray:
        cols = [self.index[j][feats[name]] for j, name in enumerate(CUSTOMER_FEATURES) if feats[name] in self.index[j]]
        return _softmax(self.intercept + self.coef[:, cols].sum(axis=1))


class LiteHybrid:
    def __init__(self, profile, pop, alpha: float):
        self.profile, self.pop, self.alpha = profile, pop, alpha

    def score(self, feats: dict) -> np.ndarray:
        return self.alpha * self.profile.score(feats) + (1 - self.alpha) * self.pop.score(feats)


def to_lite(models: dict) -> dict:
    """Convert the fitted scikit-learn based models (models.fit_models) into lite equivalents."""
    pop = LiteConst(models["popularity"].p)
    nb_m, lr_m = models["profile_naive_bayes"], models["profile_logreg"]
    nb = LiteNaiveBayes([list(c) for c in nb_m.enc.categories_], nb_m.nb.feature_log_prob_, nb_m.nb.class_log_prior_)
    lr = LiteLogReg([list(c) for c in lr_m.enc.categories_], lr_m.lr.coef_, lr_m.lr.intercept_)
    seg = models["segment_popularity"]
    return {
        "random": LiteConst(models["random"].score(_Dummy())[0]),
        "popularity": pop,
        "trending_90d": LiteConst(models["trending_90d"].p),
        "segment_popularity": LiteSegmentPopularity(seg.keys, seg.table, seg.global_p),
        "profile_naive_bayes": nb,
        "profile_logreg": lr,
        "hybrid": LiteHybrid(lr, pop, models["hybrid(logreg+popularity)"].alpha),
    }


class _Dummy:
    height = 1
