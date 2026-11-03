"""Feature engineering: item (catalogue) table and customer/context features for the 48-item recommender."""
import polars as pl

from rec.config import ARTIFACTS_DIR, CLEAN_PARQUET, PROCESSED_DIR

SPLIT_DATE_STR = "2025-11-01"
SPLIT_DATE = pl.datetime(2025, 11, 1)  # train < split <= test (about the last 3 months)

CUSTOMER_FEATURES = [
    "customer_segment", "country", "gender", "age_band", "loyalty_band", "tenure_band",
    "device_type", "campaign_source", "traffic_source", "coupon_used", "abandoned_cart_before",
    "is_weekend", "hour_band", "order_month", "session_band", "pages_band",
]


def load_clean() -> pl.DataFrame:
    return pl.read_parquet(CLEAN_PARQUET)


def add_customer_features(df: pl.DataFrame) -> pl.DataFrame:
    tenure = (pl.col("order_date").dt.date() - pl.col("account_creation_date")).dt.total_days()
    return df.with_columns(
        (pl.col("age") // 10 * 10).cast(pl.String).alias("age_band"),
        (pl.col("customer_loyalty_score") // 25).cast(pl.Int32).clip(0, 3).cast(pl.String).alias("loyalty_band"),
        (tenure // 365).clip(0, 5).cast(pl.String).alias("tenure_band"),
        (pl.col("order_hour") // 6).cast(pl.String).alias("hour_band"),
        pl.col("order_month").cast(pl.String),
        (pl.col("session_duration_minutes") // 15).cast(pl.Int32).clip(0, 3).cast(pl.String).alias("session_band"),
        (pl.col("pages_visited") // 5).cast(pl.Int32).clip(0, 3).cast(pl.String).alias("pages_band"),
        pl.col("is_weekend").cast(pl.String),
        pl.col("coupon_used").cast(pl.String),
        pl.col("abandoned_cart_before").cast(pl.String),
    )


def build_item_table(train: pl.DataFrame) -> pl.DataFrame:
    """Per-item business statistics, computed on the training period only (no leakage)."""
    net_profit = pl.when(pl.col("is_positive_purchase")).then(pl.col("profit_usd")).otherwise(0.0)
    net_revenue = pl.when(pl.col("is_positive_purchase")).then(pl.col("total_price_usd")).otherwise(0.0)
    recent_cut = train["order_date"].max() - pl.duration(days=90)
    items = train.group_by("item_key").agg(
        pl.col("category").first(),
        pl.len().alias("n_orders"),
        pl.col("is_positive_purchase").sum().alias("n_completed"),
        pl.col("unit_price_usd").mean().alias("avg_unit_price"),
        pl.col("total_price_usd").mean().alias("avg_order_value"),
        pl.col("profit_margin_percent").mean().alias("avg_margin_pct"),
        (pl.col("discount_percent").mean()).alias("avg_discount_pct"),
        pl.col("is_returned").mean().alias("return_rate"),
        pl.col("is_cancelled").mean().alias("cancel_rate"),
        pl.col("fraud_risk_score").mean().alias("avg_fraud_score"),
        (pl.col("fraud_risk_score") >= 80).mean().alias("high_fraud_share"),
        pl.col("rating").mean().alias("avg_rating"),
        (pl.col("review_sentiment") == "Positive").mean().alias("positive_share"),
        net_profit.mean().alias("net_profit_per_order"),
        net_revenue.mean().alias("net_revenue_per_order"),
        (pl.col("order_date") >= recent_cut).sum().alias("n_recent_90d"),
        pl.col("stock_quantity").mean().alias("avg_stock"),
    )
    total = items["n_orders"].sum()
    return items.with_columns(
        (pl.col("n_orders") / total).alias("popularity_share"),
        (pl.col("n_recent_90d") / pl.col("n_recent_90d").sum()).alias("recent_share"),
        pl.col("avg_unit_price").rank(method="dense").over("category").alias("price_rank_in_category"),
    ).sort("item_key")


def main() -> None:
    df = add_customer_features(load_clean())
    train = df.filter(pl.col("order_date") < SPLIT_DATE)
    test = df.filter(pl.col("order_date") >= SPLIT_DATE)
    print(f"train={train.height:,} test={test.height:,}")
    items = build_item_table(train)
    items.write_parquet(ARTIFACTS_DIR / "items.parquet")
    df.write_parquet(PROCESSED_DIR / "model_table.parquet", compression="zstd")
    with pl.Config(tbl_rows=60, tbl_cols=30, tbl_width_chars=250):
        print(items.select("item_key", "category", "n_orders", "popularity_share", "avg_unit_price",
                           "net_profit_per_order", "return_rate", "avg_fraud_score", "avg_rating").sort("net_profit_per_order", descending=True))


if __name__ == "__main__":
    main()
