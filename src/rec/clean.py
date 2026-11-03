"""Cleaning + standardisation. Keeps all rows, adds explicit quality flags and stable business keys.

Findings from profiling that drive the rules (see phases/phase-1-data-profiling-and-quality.md):
- product_id is not a stable catalogue key (same id -> different product_name/price/brand), so the item key is product_name.
- customer_id collides across different people (different name/age/country), so customer_key = customer_id + customer_name.
- order_id is duplicated for ~8k different orders; rows are kept and flagged.
- order_status / payment_status / delivery_status often contradict each other; rows are kept and flagged.
"""
import io
from contextlib import redirect_stdout

import polars as pl

from rec.config import CLEAN_PARQUET, RAW_PARQUET, REPORTS_DIR


def clean(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.col(pl.String).str.strip_chars())
    df = df.with_columns(
        pl.col("order_date").str.to_datetime(strict=True),
        pl.col("account_creation_date").str.to_date(strict=True),
        *[(pl.col(c) == "Yes").alias(c) for c in
          ["is_weekend", "installment_plan", "coupon_used", "abandoned_cart_before", "support_ticket_created"]],
    )
    df = df.with_columns(
        (pl.col("customer_id") + "|" + pl.col("customer_name")).alias("customer_key"),
        pl.col("product_name").alias("item_key"),
        (pl.col("order_status") == "Returned").alias("is_returned"),
        (pl.col("order_status") == "Cancelled").alias("is_cancelled"),
        (pl.col("payment_status") == "Failed").alias("is_payment_failed"),
        pl.col("customer_feedback").fill_null("No feedback"),
        pl.col("return_reason").fill_null("Not returned"),
        pl.col("coupon_code").fill_null("No coupon"),
    )
    df = df.with_columns(
        (pl.col("order_id").is_duplicated()).alias("flag_order_id_dup"),
        pl.col("customer_id").is_duplicated().alias("flag_customer_id_reused"),
        # status combinations that cannot all be true at once
        (
            ((pl.col("order_status") == "Completed") & pl.col("delivery_status").is_in(["Failed", "Pending"]))
            | ((pl.col("order_status").is_in(["Pending", "Cancelled", "Processing"])) & (pl.col("delivery_status") == "Delivered"))
            | ((pl.col("order_status") == "Cancelled") & (pl.col("payment_status") == "Paid"))
            | ((pl.col("order_status") == "Pending") & (pl.col("payment_status") == "Paid"))
        ).alias("flag_status_conflict"),
    )
    # a sale that actually counts towards revenue/positive feedback
    df = df.with_columns(
        (pl.col("order_status").is_in(["Completed", "Returned"]) & (pl.col("payment_status") == "Paid")).alias("is_realized_sale"),
        (pl.col("order_status") == "Completed").alias("is_positive_purchase"),
    )
    return df.sort("order_date")


def main() -> None:
    raw = pl.read_parquet(RAW_PARQUET)
    df = clean(raw)
    df.write_parquet(CLEAN_PARQUET, compression="zstd")

    buf = io.StringIO()
    with redirect_stdout(buf):
        n = df.height
        print(f"rows in={raw.height:,} out={n:,} cols={df.width}")
        print("customer_key distinct:", df["customer_key"].n_unique(), "(customer_id distinct:", df["customer_id"].n_unique(), ")")
        per = df.group_by("customer_key").len()
        print("customers with >1 order:", per.filter(pl.col("len") > 1).height, "| max orders:", per["len"].max())
        for f in ["flag_order_id_dup", "flag_customer_id_reused", "flag_status_conflict", "is_realized_sale", "is_positive_purchase", "is_returned", "is_cancelled", "is_payment_failed"]:
            k = int(df[f].sum())
            print(f"{f}: {k:,} ({100 * k / n:.2f}%)")
        print("any flag_status_conflict or dup rows:", df.filter(pl.col("flag_status_conflict") | pl.col("flag_order_id_dup")).height)
        print("items:", df["item_key"].n_unique())
        print("realized revenue USD:", round(df.filter(pl.col("is_realized_sale"))["total_price_usd"].sum()))
        print("realized profit USD:", round(df.filter(pl.col("is_realized_sale"))["profit_usd"].sum()))
        print("nulls remaining:", {c: df[c].null_count() for c in df.columns if df[c].null_count()})
    (REPORTS_DIR / "quality.txt").write_text(buf.getvalue(), encoding="utf-8")
    print(buf.getvalue())


if __name__ == "__main__":
    main()
