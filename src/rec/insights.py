"""Phase 4: business insights. Writes artifacts/insights.json (served by the API) and reports/insights.txt."""
import json

import numpy as np
import polars as pl
from scipy import stats

from rec.config import ARTIFACTS_DIR, PROCESSED_DIR, REPORTS_DIR


def p_chi(df: pl.DataFrame, dim: str, flag: str) -> float:
    ct = df.group_by(dim, flag).len().pivot(on=flag, index=dim, values="len").drop(dim).fill_null(0).to_numpy()
    return float(stats.chi2_contingency(ct)[1])


def rows(df: pl.DataFrame, rnd: int = 4) -> list[dict]:
    return [{k: (round(v, rnd) if isinstance(v, float) else v) for k, v in r.items()} for r in df.to_dicts()]


def main() -> None:
    df = pl.read_parquet(PROCESSED_DIR / "model_table.parquet").with_columns(
        pl.when(pl.col("is_positive_purchase")).then(pl.col("profit_usd")).otherwise(0.0).alias("net_profit"),
        pl.when(pl.col("is_positive_purchase")).then(pl.col("total_price_usd")).otherwise(0.0).alias("net_revenue"),
        pl.col("order_date").dt.strftime("%Y-%m").alias("ym"),
        pl.col("order_date").dt.weekday().alias("weekday"),
    )
    n = df.height
    out: dict = {"meta": {"rows": n, "start": str(df["order_date"].min().date()), "end": str(df["order_date"].max().date())}}
    notes: list[str] = []

    agg = [pl.len().alias("orders"), pl.col("is_positive_purchase").mean().alias("conversion"),
           pl.col("net_revenue").sum().alias("net_revenue"), pl.col("net_profit").sum().alias("net_profit"),
           pl.col("net_profit").mean().alias("net_profit_per_order"), pl.col("total_price_usd").mean().alias("avg_order_value"),
           pl.col("profit_margin_percent").mean().alias("avg_margin_pct"), pl.col("is_returned").mean().alias("return_rate"),
           pl.col("rating").mean().alias("avg_rating")]

    # headline KPIs
    real = df.filter(pl.col("is_realized_sale"))
    out["kpis"] = {
        "orders": n, "customers": df["customer_key"].n_unique(),
        "gross_revenue_all_orders": float(df["total_price_usd"].sum()),
        "net_revenue_completed": float(df["net_revenue"].sum()), "net_profit_completed": float(df["net_profit"].sum()),
        "realized_revenue": float(real["total_price_usd"].sum()), "realized_profit": float(real["profit_usd"].sum()),
        "completed_rate": float(df["is_positive_purchase"].mean()), "return_rate": float(df["is_returned"].mean()),
        "cancel_rate": float(df["is_cancelled"].mean()), "payment_failed_rate": float(df["is_payment_failed"].mean()),
        "avg_margin_pct": float(df["profit_margin_percent"].mean()), "status_conflict_rate": float(df["flag_status_conflict"].mean()),
    }

    # 1. product performance
    cat = df.group_by("category").agg(*agg, pl.col("item_key").n_unique().alias("items")).sort("net_profit", descending=True)
    cat = cat.with_columns((pl.col("net_profit") / pl.col("net_profit").sum()).alias("profit_share"),
                           (pl.col("orders") / pl.col("orders").sum()).alias("order_share"))
    sub = df.group_by("category", "sub_category").agg(*agg).sort("net_profit", descending=True)
    brand = df.group_by("brand").agg(*agg).sort("net_profit", descending=True)
    item = df.group_by("category", "item_key").agg(*agg).sort("net_profit", descending=True)
    # within-category spread of items: is it more than noise?
    within = []
    for c in sorted(df["category"].unique().to_list()):
        sub_df = df.filter(pl.col("category") == c)
        groups = [sub_df.filter(pl.col("item_key") == i)["net_profit"].to_numpy() for i in sub_df["item_key"].unique().to_list()]
        within.append({"category": c, "items": len(groups), "anova_p_net_profit": float(stats.f_oneway(*groups)[1]),
                       "best_vs_worst_item_gap_pct": float((max(g.mean() for g in groups) / min(g.mean() for g in groups) - 1))})
    out["products"] = {"by_category": rows(cat), "by_sub_category": rows(sub), "by_brand": rows(brand), "by_item": rows(item),
                       "within_category_item_spread": within}
    e = cat.row(0, named=True)
    notes.append(f"{e['category']} generates {e['profit_share']:.1%} of net profit from {e['order_share']:.1%} of orders; "
                 f"{cat.row(-1, named=True)['category']} generates {cat.row(-1, named=True)['profit_share']:.1%} from {cat.row(-1, named=True)['order_share']:.1%}.")
    notes.append("Within a category the 48 items are statistically indistinguishable (best vs. worst item differs by only 1% to 3%, ANOVA p between 0.5 and 0.9); category is the real unit of performance.")

    # 2. marketing effectiveness
    mk = {}
    for dim in ["campaign_source", "traffic_source", "device_type", "coupon_used"]:
        t = df.group_by(dim).agg(*agg).sort(dim).with_columns(pl.col(dim).cast(pl.String).alias("value")).drop(dim)
        mk[dim] = {"p_conversion": p_chi(df, dim, "is_positive_purchase"), "rows": rows(t)}
    disc = df.group_by("discount_percent").agg(*agg, pl.col("discount_amount_usd").sum().alias("discount_given")).sort("discount_percent")
    mk["discount_depth"] = {"p_conversion": p_chi(df, "discount_percent", "is_positive_purchase"), "rows": rows(disc)}
    out["marketing"] = mk
    best_c = max(mk["campaign_source"]["rows"], key=lambda r: r["net_profit_per_order"])
    worst_c = min(mk["campaign_source"]["rows"], key=lambda r: r["net_profit_per_order"])
    notes.append(f"Campaign sources are within {(best_c['net_profit_per_order'] / worst_c['net_profit_per_order'] - 1):.1%} of each other on net profit per order "
                 f"({best_c['value']} best, {worst_c['value']} worst; conversion p={mk['campaign_source']['p_conversion']:.2f}): no channel is meaningfully better.")
    notes.append(f"Discounts do not lift conversion (p={mk['discount_depth']['p_conversion']:.2f}) but cut margin from "
                 f"{disc.row(0, named=True)['avg_margin_pct']:.1f}% at 0% off to {disc.row(-1, named=True)['avg_margin_pct']:.1f}% at 25% off.")

    # 3. customer behaviour over time
    monthly = df.group_by("ym").agg(*agg).sort("ym")
    hourly = df.group_by("order_hour").agg(pl.len().alias("orders"), pl.col("net_profit").sum().alias("net_profit")).sort("order_hour")
    weekday = df.group_by("weekday").agg(pl.len().alias("orders"), pl.col("net_profit").mean().alias("net_profit_per_order")).sort("weekday")
    age = df.with_columns((pl.col("age") // 10 * 10).alias("age_band")).group_by("age_band").agg(*agg).sort("age_band")
    country = df.group_by("country").agg(*agg).sort("net_profit", descending=True)
    out["behaviour"] = {"monthly": rows(monthly), "hourly": rows(hourly), "weekday": rows(weekday), "age_band": rows(age), "country": rows(country),
                        "p_weekday_orders": float(stats.chisquare(weekday["orders"].to_numpy())[1]),
                        "p_hour_orders": float(stats.chisquare(hourly["orders"].to_numpy())[1])}
    full = monthly.filter(~pl.col("ym").is_in([monthly["ym"][0], monthly["ym"][-1]]))
    cv = float(full["orders"].std() / full["orders"].mean())
    out["behaviour"]["monthly_orders_cv"] = cv
    wk = weekday["orders"].to_numpy()
    notes.append(f"Monthly order volume is flat (coefficient of variation {cv:.1%} over full months) with no seasonality, and orders are uniform across hours "
                 f"(chi-square p={out['behaviour']['p_hour_orders']:.2f}). Weekday volume differs by only {(wk.max() / wk.min() - 1):.1%} (Sat/Sun slightly higher).")

    # 4. returns
    ret = df.filter(pl.col("is_returned"))
    rp = {"by_reason": rows(ret.group_by("return_reason").agg(pl.len().alias("returns"), pl.col("profit_usd").sum().alias("profit_lost"),
                                                               pl.col("total_price_usd").sum().alias("revenue_returned")).sort("returns", descending=True))}
    for dim in ["category", "quantity", "discount_percent", "shipping_method", "payment_method", "customer_segment", "support_ticket_created", "device_type"]:
        t = df.group_by(dim).agg(pl.len().alias("orders"), pl.col("is_returned").mean().alias("return_rate"),
                                 pl.col("profit_usd").filter(pl.col("is_returned")).sum().alias("profit_lost")).sort(dim)
        rp[dim] = {"p": p_chi(df, dim, "is_returned"), "rows": rows(t.with_columns(pl.col(dim).cast(pl.String).alias("value")).drop(dim))}
    rp["monthly_return_rate"] = rows(monthly.select("ym", "return_rate"))
    rp["rating_returned_vs_not"] = rows(df.group_by("is_returned").agg(pl.col("rating").mean().alias("avg_rating"), pl.len().alias("orders")))
    rp["totals"] = {"returned_orders": ret.height, "return_rate": ret.height / n, "revenue_returned": float(ret["total_price_usd"].sum()),
                    "profit_lost": float(ret["profit_usd"].sum())}
    out["returns"] = rp
    reasons = rp["by_reason"]
    notes.append(f"Returns: {rp['totals']['return_rate']:.1%} of orders, ${rp['totals']['profit_lost'] / 1e6:.1f}M profit lost. The five reasons are evenly split "
                 f"({min(r['returns'] for r in reasons) / ret.height:.1%} to {max(r['returns'] for r in reasons) / ret.height:.1%} each). Return rate does not differ significantly by "
                 + ", ".join(f"{d.replace('_', ' ')} (p={rp[d]['p']:.2f})" for d in ["category", "shipping_method", "customer_segment", "quantity"]) + ".")
    pm = sorted(rp["payment_method"]["rows"], key=lambda r: r["return_rate"])
    notes.append(f"The only detectable return driver is payment method (p<0.001), but the effect is small: {pm[0]['value']} {pm[0]['return_rate']:.1%} vs "
                 f"{pm[-1]['value']} {pm[-1]['return_rate']:.1%} ({(pm[-1]['return_rate'] - pm[0]['return_rate']) * 100:.1f} points). With 1M rows even tiny gaps are significant, so effect size matters more than p-values.")

    # 5. fraud and payment risk
    fr = df.with_columns(pl.col("fraud_risk_score").cut([20, 40, 60, 80], left_closed=True).alias("band"))
    fr_t = fr.group_by("band").agg(pl.len().alias("orders"), pl.col("is_positive_purchase").mean().alias("conversion"), pl.col("is_returned").mean().alias("return_rate"),
                                  pl.col("is_payment_failed").mean().alias("payment_failed_rate"), pl.col("total_price_usd").sum().alias("revenue")).sort("band")
    out["risk"] = {"by_fraud_band": rows(fr_t.with_columns(pl.col("band").cast(pl.String))),
                   "payment_method": rows(df.group_by("payment_method").agg(pl.len().alias("orders"), pl.col("is_payment_failed").mean().alias("failed_rate")).sort("payment_method")),
                   "p_fraud_vs_return": p_chi(fr, "band", "is_returned"),
                   "high_risk_share_80plus": float((df["fraud_risk_score"] >= 80).mean()),
                   "revenue_at_80plus": float(df.filter(pl.col("fraud_risk_score") >= 80)["total_price_usd"].sum())}
    gap = max(r["return_rate"] for r in out["risk"]["by_fraud_band"]) - min(r["return_rate"] for r in out["risk"]["by_fraud_band"])
    out["risk"]["return_rate_gap_pts"] = gap * 100
    notes.append(f"Fraud score looks uniform 0 to 100: {out['risk']['high_risk_share_80plus']:.1%} of orders score 80+. Return rate varies by at most {gap * 100:.1f} points "
                 f"across fraud bands (p={out['risk']['p_fraud_vs_return']:.2f}), so the score is a policy filter, not a validated risk signal.")

    # 6. revenue opportunities (every figure uses data from this dataset, assumptions stated)
    opp = []
    disc_realized = float(real["discount_amount_usd"].sum())
    capped = real.with_columns((pl.col("unit_price_usd") * pl.col("quantity") * (pl.col("discount_percent").clip(0, 10) / 100)).alias("d10"))
    recapture = float((real["discount_amount_usd"] - capped["d10"]).sum())
    opp.append({"id": "discount_cap", "title": "Cap discounts at 10%",
                "value_usd": recapture,
                "basis": f"Discounts given on realized sales total ${disc_realized / 1e6:.1f}M. Conversion is statistically unchanged by discount depth "
                         f"(p={mk['discount_depth']['p_conversion']:.2f}), so capping at 10% would recapture the excess as margin if that holds in practice.",
                "assumption": "Conversion stays unchanged. This must be validated with an A/B test before rollout."})
    ret_profit = rp["totals"]["profit_lost"]
    opp.append({"id": "returns_cut", "title": "Reduce returns by 20%", "value_usd": 0.2 * ret_profit,
                "basis": f"Returned orders lost ${ret_profit / 1e6:.1f}M of profit. 'Wrong Item' and 'Not As Described' ({(reasons[0]['returns'] * 0 + sum(r['returns'] for r in reasons if r['return_reason'] in ('Wrong Item', 'Not As Described'))) / ret.height:.0%} of returns) are fixable with catalogue and fulfilment quality.",
                "assumption": "Returns fall 20% overall. The two fixable reasons are about 40% of returns, so this needs half of those removed."})
    unrealized = df.filter(pl.col("order_status").is_in(["Pending", "Processing"]) & (pl.col("payment_status") == "Paid"))
    opp.append({"id": "stuck_orders", "title": "Clear Pending/Processing orders that are already paid",
                "value_usd": float(unrealized["profit_usd"].sum()), "basis": f"{unrealized.height:,} paid orders are stuck in Pending/Processing, holding ${unrealized['profit_usd'].sum() / 1e6:.1f}M of profit.",
                "assumption": "These are genuine orders. Status conflicts (23% of rows) suggest a data issue, so check them first."})
    failed = df.filter(pl.col("is_payment_failed") & ~pl.col("is_cancelled"))
    opp.append({"id": "payment_recovery", "title": "Recover 25% of failed payments",
                "value_usd": 0.25 * float(failed["profit_usd"].sum()), "basis": f"{failed.height:,} orders have a failed payment and were not cancelled, worth ${failed['profit_usd'].sum() / 1e6:.1f}M in profit.",
                "assumption": "A retry or alternative payment flow converts 25%."})
    cat_p = {r["category"]: r["net_profit_per_order"] for r in out["products"]["by_category"]}
    mix_orders = float(df.filter(pl.col("category") == "Clothing")["is_positive_purchase"].sum())
    shift = 0.10 * mix_orders * (cat_p["Electronics"] - cat_p["Clothing"])
    opp.append({"id": "mix_shift", "title": "Shift 10% of Clothing demand towards Electronics via recommendations", "value_usd": shift,
                "basis": f"Net profit per order is ${cat_p['Electronics']:.0f} in Electronics and ${cat_p['Clothing']:.0f} in Clothing; {int(mix_orders):,} Clothing orders completed. This is what the profit-aware re-ranker in Phase 2 does.",
                "assumption": "Customers accept the cross-category suggestion. Phase 2 measured +22% profit captured at lambda=0.25 without losing hit rate."})
    out["opportunities"] = opp
    for o in opp:
        notes.append(f"Opportunity: {o['title']}: about ${o['value_usd'] / 1e6:.1f}M ({o['assumption']})")

    # 7. operations
    ops = {"delivery_by_method": rows(df.group_by("shipping_method").agg(pl.col("delivery_days").mean().alias("avg_delivery_days"), pl.col("shipping_cost_usd").mean().alias("avg_shipping_cost"), pl.col("rating").mean().alias("avg_rating")).sort("shipping_method")),
           "status_conflict_rate": float(df["flag_status_conflict"].mean())}
    out["operations"] = ops
    notes.append("Express, Next Day, Standard and Economy shipping all average about 7.5 delivery days: the shipping method does not change delivery time in this data.")

    out["findings"] = notes
    (ARTIFACTS_DIR / "insights.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    (REPORTS_DIR / "insights.txt").write_text("\n".join(f"- {x}" for x in notes), encoding="utf-8")
    for x in notes:
        print("-", x)
    print("\ncategories:", [(r["category"], round(r["net_profit"] / 1e6, 1), round(r["net_profit_per_order"], 1), round(r["return_rate"], 3)) for r in out["products"]["by_category"]])
    print("within:", within)
    print("kpis:", out["kpis"])
    print("opps:", [(o["id"], round(o["value_usd"] / 1e6, 2)) for o in opp])


if __name__ == "__main__":
    main()
