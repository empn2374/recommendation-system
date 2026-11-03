"""Rule-based cross-sell and upsell tables.

There is no basket data (one product per order, one order per customer), so co-purchase lift cannot be computed.
Cross-sell therefore uses an explicit, documented category-complement rule ranked by business value, and
upsell moves the same product to the next price tier (same margin %, higher absolute profit).
"""
import polars as pl

from rec.config import ARTIFACTS_DIR
from rec.features import SPLIT_DATE, build_item_table

COMPLEMENTS = {
    "Electronics": ["Home", "Sports"],
    "Home": ["Electronics", "Health"],
    "Sports": ["Health", "Clothing"],
    "Health": ["Sports", "Home"],
    "Clothing": ["Sports", "Health"],
}
TOP_N = 5


def main() -> None:
    df = pl.read_parquet("data/processed/model_table.parquet")
    train = df.filter(pl.col("order_date") < SPLIT_DATE)
    items = build_item_table(train)

    rows = []
    for item, cat in items.select("item_key", "category").iter_rows():
        cand = (items.filter(pl.col("category").is_in(COMPLEMENTS[cat]))
                .with_columns((pl.col("net_profit_per_order") * (1 - pl.col("return_rate"))).alias("score"))
                .sort("score", descending=True).head(TOP_N))
        for rank, (k, s) in enumerate(cand.select("item_key", "score").iter_rows(), 1):
            rows.append({"item_key": item, "rank": rank, "cross_sell_item": k, "score": s})
    pl.DataFrame(rows).write_parquet(ARTIFACTS_DIR / "cross_sell.parquet")

    # price tiers: quartiles of unit price within each category (training period)
    q = (train.group_by("category").agg(
        pl.col("unit_price_usd").quantile(0.25).alias("q1"),
        pl.col("unit_price_usd").quantile(0.5).alias("q2"),
        pl.col("unit_price_usd").quantile(0.75).alias("q3")))
    t = train.join(q, on="category").with_columns(
        pl.when(pl.col("unit_price_usd") <= pl.col("q1")).then(1)
        .when(pl.col("unit_price_usd") <= pl.col("q2")).then(2)
        .when(pl.col("unit_price_usd") <= pl.col("q3")).then(3).otherwise(4).alias("price_tier"))
    tiers = t.group_by("category", "price_tier").agg(
        pl.len().alias("n"),
        pl.col("unit_price_usd").mean().alias("avg_unit_price"),
        pl.col("profit_margin_percent").mean().alias("avg_margin_pct"),
        pl.col("is_returned").mean().alias("return_rate"),
        pl.when(pl.col("is_positive_purchase")).then(pl.col("profit_usd")).otherwise(0.0).mean().alias("net_profit_per_order"),
        pl.col("rating").mean().alias("avg_rating"),
    ).sort("category", "price_tier")
    tiers.write_parquet(ARTIFACTS_DIR / "price_tiers.parquet")
    with pl.Config(tbl_rows=30, tbl_width_chars=200):
        print(tiers)
        print(pl.read_parquet(ARTIFACTS_DIR / "cross_sell.parquet").filter(pl.col("item_key") == "Wireless Mouse 2.4GHz"))


if __name__ == "__main__":
    main()
