"""Phase 3: customer intelligence. Value tiers, clustering, high-value analysis, behaviour tests, CLV proxy."""
import io
import json
import math
from contextlib import redirect_stdout

import numpy as np
import polars as pl
from scipy import stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rec.config import ARTIFACTS_DIR, PROCESSED_DIR, REPORTS_DIR, SEED

CLUSTER_FEATURES = ["age", "customer_loyalty_score", "tenure_days", "session_duration_minutes", "pages_visited",
                    "fraud_risk_score", "quantity", "discount_percent", "log_price"]
PROFILE_ONLY = ["age", "customer_loyalty_score", "tenure_days", "session_duration_minutes", "pages_visited",
                "fraud_risk_score"]
ASSUMED_RETENTION = 0.8
ORDERS_PER_YEAR = 1.0
HORIZON_YEARS = 3


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    grand = values.mean()
    ss_b = sum(len(values[labels == g]) * (values[labels == g].mean() - grand) ** 2 for g in np.unique(labels))
    return float(ss_b / ((values - grand) ** 2).sum())


def main() -> None:
    df = pl.read_parquet(PROCESSED_DIR / "model_table.parquet").with_columns(
        (pl.col("order_date").dt.date() - pl.col("account_creation_date")).dt.total_days().alias("tenure_days"),
        pl.col("unit_price_usd").log().alias("log_price"),
        pl.when(pl.col("is_positive_purchase")).then(pl.col("profit_usd")).otherwise(0.0).alias("net_profit"),
        pl.when(pl.col("is_positive_purchase")).then(pl.col("total_price_usd")).otherwise(0.0).alias("net_revenue"),
    )
    n = df.height
    buf = io.StringIO()
    out: dict = {}
    with redirect_stdout(buf):
        print(f"customers (one row each): {n:,}")

        # 1. value tiers (RFM is not computable with one order, so value is the only usable dimension)
        q = df["net_profit"].quantile(0.5), df["net_profit"].quantile(0.9)
        df = df.with_columns(
            pl.when(pl.col("net_profit") == 0).then(pl.lit("No realised value"))
            .when(pl.col("net_profit") <= q[0]).then(pl.lit("Standard"))
            .when(pl.col("net_profit") <= q[1]).then(pl.lit("High value"))
            .otherwise(pl.lit("Top 10%")).alias("value_tier"))
        # top decile definition used everywhere: top 10% of ALL customers by net profit
        thr = df["net_profit"].quantile(0.9)
        df = df.with_columns((pl.col("net_profit") >= thr).alias("is_high_value"))
        tiers = (df.group_by("value_tier").agg(pl.len().alias("customers"), pl.col("net_profit").sum().alias("profit"),
                                               pl.col("net_profit").mean().alias("avg_profit"))
                 .with_columns((pl.col("profit") / pl.col("profit").sum()).alias("profit_share"),
                               (pl.col("customers") / n).alias("customer_share")).sort("avg_profit", descending=True))
        print("\n== VALUE TIERS ==\n", tiers)
        hv = df.filter(pl.col("is_high_value"))
        print(f"high-value threshold (top 10% of all customers) net profit >= ${thr:,.2f}; "
              f"they generate {hv['net_profit'].sum() / df['net_profit'].sum():.1%} of net profit")
        prof_sorted = np.sort(df["net_profit"].to_numpy())[::-1]
        cum = prof_sorted.cumsum() / prof_sorted.sum()
        pareto = {p: float(cum[int(n * p / 100) - 1]) for p in (1, 5, 10, 20, 50)}
        print("Pareto (top x% of customers -> share of net profit):", {k: f"{v:.1%}" for k, v in pareto.items()})
        out["value_tiers"] = tiers.to_dicts()
        out["pareto"] = pareto
        out["high_value_threshold"] = float(thr)

        # 2. what makes a high-value customer? (order mechanics vs customer attributes)
        print("\n== HIGH-VALUE vs OTHERS: mean of order mechanics ==")
        mech = df.group_by("is_high_value").agg(pl.col("quantity").mean(), pl.col("unit_price_usd").mean(),
                                                pl.col("discount_percent").mean(), pl.col("age").mean(),
                                                pl.col("customer_loyalty_score").mean(), pl.col("tenure_days").mean(),
                                                pl.col("fraud_risk_score").mean())
        print(mech)
        print("share of high-value by category:", df.group_by("category").agg(pl.col("is_high_value").mean().round(4)).sort("category").to_dicts())
        out["high_value_by_category"] = {r["category"]: r["is_high_value"] for r in df.group_by("category").agg(pl.col("is_high_value").mean()).to_dicts()}
        print("share of high-value by given customer_segment:", df.group_by("customer_segment").agg(pl.col("is_high_value").mean().round(4)).sort("customer_segment").to_dicts())

        # can high value be predicted from customer attributes only (before the order)?
        feats = PROFILE_ONLY
        s = df.sample(300_000, seed=SEED)
        X = s.select(feats).to_numpy().astype(float)
        yv = s["is_high_value"].to_numpy()
        Xtr, Xte, ytr, yte = train_test_split(X, yv, test_size=0.3, random_state=SEED, stratify=yv)
        gb = HistGradientBoostingClassifier(max_iter=150, random_state=SEED).fit(Xtr, ytr)
        auc = float(roc_auc_score(yte, gb.predict_proba(Xte)[:, 1]))
        print(f"\nAUC predicting 'top 10% value' from customer attributes only (age, loyalty, tenure, session, pages, fraud): {auc:.3f} (0.5 = no signal)")
        out["high_value_auc_profile_only"] = auc

        # 3. clustering
        print("\n== CLUSTERING (MiniBatchKMeans on standardised order + profile features) ==")
        Z = StandardScaler().fit_transform(df.select(CLUSTER_FEATURES).to_numpy().astype(float))
        rng = np.random.default_rng(SEED)
        sil_idx = rng.choice(n, 20_000, replace=False)
        sil = {}
        for k in (3, 4, 5, 6, 8):
            km = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=3, batch_size=20_000).fit(Z)
            sil[k] = float(silhouette_score(Z[sil_idx], km.labels_[sil_idx]))
        print("silhouette by k (1 = perfect, ~0 = no structure):", {k: round(v, 3) for k, v in sil.items()})
        out["silhouette"] = sil
        k_best = 4
        km = MiniBatchKMeans(n_clusters=k_best, random_state=SEED, n_init=5, batch_size=20_000).fit(Z)
        df = df.with_columns(pl.Series("cluster_raw", km.labels_))
        prof = (df.group_by("cluster_raw").agg(pl.len().alias("customers"), pl.col("quantity").mean().alias("qty"),
                                               pl.col("unit_price_usd").mean().alias("price"),
                                               pl.col("discount_percent").mean().alias("disc"),
                                               pl.col("age").mean().alias("age"),
                                               pl.col("customer_loyalty_score").mean().alias("loyalty"),
                                               pl.col("tenure_days").mean().alias("tenure_d"),
                                               pl.col("fraud_risk_score").mean().alias("fraud"),
                                               pl.col("session_duration_minutes").mean().alias("session"),
                                               pl.col("pages_visited").mean().alias("pages"),
                                               pl.col("net_profit").mean().alias("avg_net_profit"),
                                               pl.col("is_returned").mean().alias("return_rate"))
                .sort("avg_net_profit", descending=True))
        names = ["Cluster A (highest value)", "Cluster B", "Cluster C", "Cluster D (lowest value)"]
        mapping = {int(r): names[i] for i, r in enumerate(prof["cluster_raw"].to_list())}
        df = df.with_columns(pl.col("cluster_raw").replace_strict(mapping, return_dtype=pl.String).alias("cluster"))
        prof = prof.with_columns(pl.col("cluster_raw").replace_strict(mapping, return_dtype=pl.String).alias("cluster")).drop("cluster_raw")
        with pl.Config(tbl_cols=20, tbl_width_chars=250):
            print(prof)
        out["clusters"] = prof.to_dicts()
        # which features separate clusters? eta-squared (share of variance explained by cluster)
        labs = df["cluster"].to_numpy()
        eta = {f: eta_squared(df[f].to_numpy().astype(float), labs) for f in CLUSTER_FEATURES}
        print("variance explained by cluster (eta^2):", {k: round(v, 3) for k, v in sorted(eta.items(), key=lambda x: -x[1])})
        out["cluster_eta2"] = eta
        out["cluster_k"] = k_best

        # 4. behaviour analysis with significance tests
        print("\n== BEHAVIOUR: does it change conversion or value? (chi-square / ANOVA p-values) ==")
        beh = {}
        for col in ["device_type", "campaign_source", "traffic_source", "coupon_used", "abandoned_cart_before", "is_weekend", "gender", "country", "customer_segment"]:
            g = df.group_by(col).agg(pl.len().alias("n"), pl.col("is_positive_purchase").mean().alias("conversion"),
                                     pl.col("net_profit").mean().alias("avg_net_profit"),
                                     pl.col("total_price_usd").mean().alias("aov"), pl.col("is_returned").mean().alias("return_rate")).sort(col)
            ct = df.group_by(col, "is_positive_purchase").len().pivot(on="is_positive_purchase", index=col, values="len").drop(col).to_numpy()
            p_conv = float(stats.chi2_contingency(ct)[1])
            groups = [df.filter(pl.col(col) == v)["net_profit"].to_numpy() for v in g[col].to_list()]
            p_val = float(stats.f_oneway(*groups)[1])
            spread = float((g["avg_net_profit"].max() - g["avg_net_profit"].min()) / g["avg_net_profit"].mean())
            print(f"{col:22s} conversion p={p_conv:.3f} | net-profit ANOVA p={p_val:.3f} | profit spread max-min={spread:.1%}")
            beh[col] = {"p_conversion": p_conv, "p_profit": p_val, "profit_spread": spread, "rows": g.rename({col: "value"}).with_columns(pl.col("value").cast(pl.String)).to_dicts()}
        out["behaviour"] = beh
        for band_col, edges in {"age": [18, 30, 40, 50, 60, 76], "pages_visited": [0, 5, 10, 15, 21], "session_duration_minutes": [0, 15, 30, 45, 61], "customer_loyalty_score": [0, 25, 50, 75, 101]}.items():
            lab = df.with_columns(pl.col(band_col).cut(edges[1:-1], left_closed=True).alias("b")).group_by("b").agg(
                pl.len().alias("n"), pl.col("is_positive_purchase").mean().alias("conv"), pl.col("net_profit").mean().alias("avg_net_profit")).sort("b")
            print(f"\n{band_col} bands:\n", lab)

        # 5. CLV proxy
        print("\n== CLV PROXY ==")
        print("Each customer has exactly one order, so BG/NBD or Gamma-Gamma cannot be fitted. A transparent scenario proxy is used:")
        print(f"  CLV_{HORIZON_YEARS}y = realised net profit + mean net profit per order x ORDERS_PER_YEAR x sum(r^t, t=1..{HORIZON_YEARS}), r={ASSUMED_RETENTION}")
        tenure_years = (pl.col("tenure_days") / 365.25).clip(0.25, None)
        df = df.with_columns((pl.col("total_orders_by_customer") / tenure_years).clip(0, 12).alias("claimed_orders_per_year"))
        corr_claim = df.select(pl.corr("claimed_orders_per_year", "net_profit")).item()
        print(f"  total_orders_by_customer was NOT used: it averages {df['total_orders_by_customer'].mean():.1f} orders per customer, contradicting the "
              f"1-order reality, and its derived order rate has correlation {corr_claim:.4f} with actual profit (noise).")
        mean_profit = float(df["net_profit"].mean())
        factor = sum(ASSUMED_RETENTION ** t for t in range(1, HORIZON_YEARS + 1))
        scenarios = {"conservative": 0.5, "base": 1.0, "optimistic": 2.0}
        df = df.with_columns((pl.col("net_profit") + mean_profit * ORDERS_PER_YEAR * factor).alias("clv_proxy"))
        print(f"  mean net profit per order ${mean_profit:.2f}; retention factor {factor:.3f}; base scenario orders/year={ORDERS_PER_YEAR}")
        for nm, opm in scenarios.items():
            print(f"  scenario {nm:12s} orders/year={opm}: mean CLV ${df['net_profit'].mean() + mean_profit * opm * factor:,.2f}")
        print(f"  base CLV proxy: mean ${df['clv_proxy'].mean():,.2f}; median ${df['clv_proxy'].median():,.2f}; p90 ${df['clv_proxy'].quantile(0.9):,.2f}")
        clv_tier = df.group_by("customer_segment").agg(pl.col("clv_proxy").mean().alias("mean_clv"), pl.len().alias("n")).sort("customer_segment")
        print("  CLV proxy by given customer_segment:", clv_tier.to_dicts())
        out["clv"] = {"mean_profit_per_order": mean_profit, "retention": ASSUMED_RETENTION, "horizon_years": HORIZON_YEARS,
                      "retention_factor": factor, "orders_per_year_base": ORDERS_PER_YEAR, "mean": float(df["clv_proxy"].mean()),
                      "median": float(df["clv_proxy"].median()), "p90": float(df["clv_proxy"].quantile(0.9)),
                      "claimed_rate_corr_with_profit": corr_claim,
                      "scenarios": {nm: float(df["net_profit"].mean() + mean_profit * opm * factor) for nm, opm in scenarios.items()}}

        # 6. does the given label (VIP/Premium/Regular) mean anything?
        lab = df.group_by("customer_segment").agg(pl.len().alias("n"), pl.col("net_profit").mean().alias("avg_net_profit"),
                                                  pl.col("customer_loyalty_score").mean().alias("loyalty"),
                                                  pl.col("total_orders_by_customer").mean().alias("claimed_orders")).sort("customer_segment")
        print("\n== IS THE GIVEN customer_segment LABEL MEANINGFUL? ==\n", lab)
        out["given_segment"] = lab.to_dicts()

    text = buf.getvalue()
    print(text)
    (REPORTS_DIR / "customers.txt").write_text(text, encoding="utf-8")
    (ARTIFACTS_DIR / "customer_intelligence.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    keep = ["customer_key", "customer_segment", "country", "gender", "age", "customer_loyalty_score", "tenure_days", "device_type",
            "campaign_source", "traffic_source", "coupon_used", "abandoned_cart_before", "is_weekend", "order_hour", "order_month",
            "session_duration_minutes", "pages_visited", "category", "item_key", "net_profit", "value_tier", "is_high_value", "cluster",
            "clv_proxy", "claimed_orders_per_year"]
    df.select(keep).write_parquet(PROCESSED_DIR / "customers_full.parquet", compression="zstd")


if __name__ == "__main__":
    main()
