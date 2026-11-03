"""Export a small set of real customer profiles (from the test period) for the demo UI."""
import polars as pl

from rec.config import ARTIFACTS_DIR, PROCESSED_DIR, SEED
from rec.features import SPLIT_DATE

N = 300


def main() -> None:
    df = pl.read_parquet(PROCESSED_DIR / "model_table.parquet")
    cust = pl.read_parquet(PROCESSED_DIR / "customers_full.parquet").select("customer_key", "value_tier", "cluster", "clv_proxy", "net_profit")
    test = df.filter(pl.col("order_date") >= SPLIT_DATE).join(cust, on="customer_key", how="left", suffix="_c")
    s = test.sample(N, seed=SEED).select(
        pl.col("customer_id"), "customer_segment", "country", "gender", "age", "customer_loyalty_score",
        (pl.col("order_date").dt.date() - pl.col("account_creation_date")).dt.total_days().alias("tenure_days"),
        "device_type", "campaign_source", "traffic_source",
        (pl.col("coupon_used") == "true").alias("coupon_used"),
        (pl.col("abandoned_cart_before") == "true").alias("abandoned_cart_before"),
        (pl.col("is_weekend") == "true").alias("is_weekend"),
        "order_hour", "order_month", "session_duration_minutes", "pages_visited", "fraud_risk_score",
        pl.col("item_key").alias("actual_item"), pl.col("category").alias("actual_category"), "order_status",
        "total_price_usd", "profit_usd", "value_tier", "cluster", "clv_proxy",
    )
    s.write_parquet(ARTIFACTS_DIR / "sample_customers.parquet", compression="zstd")
    print(s.height, "sample customers written;", (ARTIFACTS_DIR / "sample_customers.parquet").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
