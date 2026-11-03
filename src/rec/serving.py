"""Serving layer: loads artifacts once and turns a customer profile into a ranked, explained recommendation list.

Dependencies are kept minimal on purpose (numpy + polars for reading artifacts): no scikit-learn, no scipy.
Everything that can be computed once (upsell tiers, option lists) is precomputed at startup.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import polars as pl

from rec.features import CUSTOMER_FEATURES

DEFAULT_ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", Path(__file__).resolve().parents[2] / "artifacts"))
SERVED_MODELS = ["popularity", "trending_90d", "segment_popularity", "profile_naive_bayes", "profile_logreg", "hybrid"]

DEFAULT_PROFILE = {
    "customer_segment": "Regular", "country": "United States", "gender": "Female", "age": 40,
    "customer_loyalty_score": 50.0, "tenure_days": 700, "device_type": "Mobile", "campaign_source": "Organic",
    "traffic_source": "Search", "coupon_used": False, "abandoned_cart_before": False, "is_weekend": False,
    "order_hour": 12, "order_month": 6, "session_duration_minutes": 30.0, "pages_visited": 10, "fraud_risk_score": 30.0,
}


def _clip(v: float, lo: int, hi: int) -> int:
    return int(min(max(v, lo), hi))


def profile_to_features(p: dict) -> dict:
    """Same bucketing as features.add_customer_features, for a single raw profile."""
    return {
        "customer_segment": p["customer_segment"], "country": p["country"], "gender": p["gender"],
        "age_band": str(int(p["age"]) // 10 * 10),
        "loyalty_band": str(_clip(p["customer_loyalty_score"] // 25, 0, 3)),
        "tenure_band": str(_clip(p["tenure_days"] // 365, 0, 5)),
        "device_type": p["device_type"], "campaign_source": p["campaign_source"], "traffic_source": p["traffic_source"],
        "coupon_used": str(bool(p["coupon_used"])).lower(),
        "abandoned_cart_before": str(bool(p["abandoned_cart_before"])).lower(),
        "is_weekend": str(bool(p["is_weekend"])).lower(),
        "hour_band": str(int(p["order_hour"]) // 6),
        "order_month": str(int(p["order_month"])),
        "session_band": str(_clip(p["session_duration_minutes"] // 15, 0, 3)),
        "pages_band": str(_clip(p["pages_visited"] // 5, 0, 3)),
    }


class Recommender:
    def __init__(self, artifacts: Path = DEFAULT_ARTIFACTS):
        items = pl.read_parquet(artifacts / "items.parquet").sort("item_key")
        self.names: list[str] = items["item_key"].to_list()
        self.items = items.to_dicts()
        with open(artifacts / "models" / "lite_models.pkl", "rb") as f:
            lite = pickle.load(f)
        self.models = {k: lite[k] for k in SERVED_MODELS}
        self.meta = json.loads((artifacts / "models" / "meta.json").read_text(encoding="utf-8"))
        self.metrics = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))
        self.insights = json.loads((artifacts / "insights.json").read_text(encoding="utf-8"))
        self.customer_intel = json.loads((artifacts / "customer_intelligence.json").read_text(encoding="utf-8"))
        samples = artifacts / "sample_customers.parquet"
        self.samples: list[dict] = pl.read_parquet(samples).to_dicts() if samples.exists() else []
        tiers = pl.read_parquet(artifacts / "price_tiers.parquet").sort("category", "price_tier").to_dicts()
        cross = pl.read_parquet(artifacts / "cross_sell.parquet").sort("item_key", "rank").to_dicts()

        self.value = items["net_profit_per_order"].to_numpy()
        self.category = items["category"].to_list()
        self.price = items["avg_unit_price"].to_numpy()
        self.ret = items["return_rate"].to_numpy()
        self.fraud = items["avg_fraud_score"].to_numpy() / 100
        self.pos = items["positive_share"].to_numpy()
        self.rating = items["avg_rating"].to_numpy()
        self.margin = items["avg_margin_pct"].to_numpy()
        self.mean_value = float(self.value.mean())
        self.pos_in = {n: i for i, n in enumerate(self.names)}

        self.tiers: dict[str, list[dict]] = {}
        for t in tiers:
            self.tiers.setdefault(t["category"], []).append(t)
        self.upsell = [self._upsell(i) for i in range(len(self.names))]
        self.cross: dict[str, list[dict]] = {}
        for c in cross:
            self.cross.setdefault(c["item_key"], []).append(c)
        self._options = self._build_options()

    def _upsell(self, i: int) -> dict | None:
        for t in self.tiers[self.category[i]]:
            if t["avg_unit_price"] > self.price[i]:
                return {"tier": int(t["price_tier"]), "avg_unit_price": float(t["avg_unit_price"]),
                        "net_profit_per_order": float(t["net_profit_per_order"]),
                        "extra_profit": float(t["net_profit_per_order"] - self.value[i])}
        return None

    def _final(self, rel: np.ndarray, lam: float, w_return: float, w_fraud: float, w_sent: float) -> np.ndarray:
        r = rel / rel.max()
        v = self.value / self.value.max()
        return (1 - lam) * r + lam * v - w_return * self.ret - w_fraud * self.fraud + w_sent * self.pos

    def recommend(self, profile: dict | None = None, model: str = "hybrid", k: int = 5, lam: float = 0.25,
                  w_return: float = 0.0, w_fraud: float = 0.0, w_sentiment: float = 0.0,
                  exclude_high_risk: bool = False, risk_threshold: float = 80.0, category: str | None = None) -> dict:
        p = {**DEFAULT_PROFILE, **(profile or {})}
        if model not in self.models:
            raise ValueError(f"unknown model {model}")
        high_risk = float(p["fraud_risk_score"]) >= risk_threshold
        out = {"model": model, "k": k, "lambda": lam, "profile": p, "high_risk": high_risk, "blocked": False, "items": [], "baseline": []}
        if exclude_high_risk and high_risk:
            out["blocked"] = True
            out["message"] = (f"Fraud risk score {p['fraud_risk_score']:.0f} is at or above the threshold {risk_threshold:.0f}: "
                              "recommendations are withheld and the order is routed for review.")
            return out

        rel = self.models[model].score(profile_to_features(p))
        final = self._final(rel, lam, w_return, w_fraud, w_sentiment)
        base = self._final(rel, 0.0, 0.0, 0.0, 0.0)
        mask = np.array([c == category for c in self.category]) if category else np.ones(len(self.names), dtype=bool)
        order = [int(i) for i in np.argsort(-final) if mask[i]][:k]
        base_order = [int(i) for i in np.argsort(-base) if mask[i]][:k]
        rel_rank = {int(i): r for r, i in enumerate(np.argsort(-rel), 1)}

        out["items"] = [self._item(i, rel, final, rank, rel_rank, lam, w_return, w_fraud, w_sentiment) for rank, i in enumerate(order, 1)]
        out["baseline"] = [{"item": self.names[i], "category": self.category[i], "net_profit_per_order": float(self.value[i])} for i in base_order]
        base_val = float(self.value[base_order].mean())
        new_val = float(self.value[order].mean())
        base_rel = float(rel[base_order].mean())
        new_rel = float(rel[order].mean())
        out["comparison"] = {
            "avg_profit_baseline": base_val, "avg_profit_reranked": new_val,
            "profit_uplift_pct": (new_val / base_val - 1) if base_val else 0.0,
            "avg_relevance_baseline": base_rel, "avg_relevance_reranked": new_rel,
            "relevance_change_pct": (new_rel / base_rel - 1) if base_rel else 0.0,
            "overlap": len(set(order) & set(base_order)),
        }
        return out

    def _item(self, i, rel, final, rank, rel_rank, lam, w_return, w_fraud, w_sent) -> dict:
        cat = self.category[i]
        reasons = []
        if rel_rank[i] <= 10:
            reasons.append(f"Ranked #{rel_rank[i]} of {len(self.names)} on relevance for this profile")
        else:
            reasons.append(f"Relevance rank #{rel_rank[i]}: lifted by business value, not by predicted preference")
        ratio = self.value[i] / self.mean_value
        if lam > 0 and ratio >= 1.2:
            reasons.append(f"High value: ${self.value[i]:.0f} net profit per order ({ratio:.1f}x the catalogue average)")
        elif ratio < 0.8:
            reasons.append(f"Lower value item: ${self.value[i]:.0f} net profit per order ({ratio:.1f}x the average)")
        reasons.append(f"{cat} item with {self.margin[i]:.1f}% margin and {self.ret[i]:.1%} return rate")
        if w_return or w_fraud or w_sent:
            reasons.append("Risk and sentiment adjustments applied")
        return {
            "rank": rank, "item": self.names[i], "category": cat, "relevance": float(rel[i]),
            "final_score": float(final[i]), "net_profit_per_order": float(self.value[i]),
            "avg_unit_price": float(self.price[i]), "margin_pct": float(self.margin[i]), "return_rate": float(self.ret[i]),
            "avg_rating": float(self.rating[i]), "positive_share": float(self.pos[i]), "reasons": reasons, "upsell": self.upsell[i],
        }

    def related(self, item: str) -> dict:
        if item not in self.pos_in:
            raise KeyError(item)
        i = self.pos_in[item]
        cat = self.category[i]
        cross = []
        for r in self.cross.get(item, []):
            j = self.pos_in[r["cross_sell_item"]]
            cross.append({"item": r["cross_sell_item"], "category": self.category[j], "score": float(r["score"]),
                          "net_profit_per_order": float(self.value[j])})
        return {"item": item, "category": cat, "net_profit_per_order": float(self.value[i]), "cross_sell": cross,
                "price_tiers": self.tiers[cat],
                "note": "Cross-sell uses an assumed complementary-category rule (no basket data exists). Upsell moves to the next price tier."}

    def _build_options(self) -> dict:
        out = {"items": self.names, "categories": sorted(set(self.category)), "models": list(self.models),
               "defaults": DEFAULT_PROFILE, "features": CUSTOMER_FEATURES}
        for col in ["customer_segment", "country", "gender", "device_type", "campaign_source", "traffic_source"]:
            out[col] = sorted({s[col] for s in self.samples})
        return out

    def options(self) -> dict:
        return self._options
