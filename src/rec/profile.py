"""Data profiling: writes reports/profile.txt with schema, nulls, cardinality, distributions and grain checks."""
import io
from contextlib import redirect_stdout

import polars as pl

from rec.config import RAW_PARQUET, REPORTS_DIR


def main() -> None:
    df = pl.read_parquet(RAW_PARQUET)
    buf = io.StringIO()
    with redirect_stdout(buf), pl.Config(tbl_rows=100, tbl_cols=20, fmt_str_lengths=60):
        n = df.height
        print(f"ROWS {n:,}  COLS {df.width}\n")

        print("== NULLS / UNIQUE / DTYPE ==")
        rows = []
        for c in df.columns:
            rows.append((c, str(df[c].dtype), df[c].null_count(), round(100 * df[c].null_count() / n, 2), df[c].n_unique()))
        print(pl.DataFrame(rows, schema=["col", "dtype", "nulls", "null_pct", "n_unique"], orient="row"))

        print("\n== GRAIN / KEYS ==")
        print("distinct order_id:", df["order_id"].n_unique(), "| full-row duplicates:", n - df.unique().height)
        print("distinct customers:", df["customer_id"].n_unique(), "| products:", df["product_id"].n_unique())
        per_cust = df.group_by("customer_id").len()["len"]
        print("orders/customer describe:\n", per_cust.describe())
        per_prod = df.group_by("product_id").len()["len"]
        print("rows/product describe:\n", per_prod.describe())
        print("order_id rows with count>1:", df.group_by("order_id").len().filter(pl.col("len") > 1).height)

        print("\n== LOW-CARDINALITY VALUE COUNTS ==")
        for c in df.columns:
            if df[c].dtype == pl.String and df[c].n_unique() <= 30:
                print(f"\n-- {c}")
                print(df[c].value_counts(sort=True))
        for c in ["order_year", "order_month", "discount_percent", "rating", "order_hour"]:
            print(f"\n-- {c}")
            print(df[c].value_counts(sort=True).sort(c))

        print("\n== NUMERIC DESCRIBE ==")
        num = [c for c in df.columns if df[c].dtype in (pl.Int64, pl.Float64)]
        print(df.select(num).describe())

        print("\n== DATES ==")
        d = df.with_columns(pl.col("order_date").str.to_datetime(strict=False).alias("od"),
                            pl.col("account_creation_date").str.to_date(strict=False).alias("ac"))
        print("order_date range:", d["od"].min(), d["od"].max(), "unparsed:", d["od"].null_count())
        print("account_creation range:", d["ac"].min(), d["ac"].max(), "unparsed:", d["ac"].null_count())
        print("orders before account creation:", d.filter(pl.col("od").dt.date() < pl.col("ac")).height)

        print("\n== CONSISTENCY ==")
        calc = (pl.col("unit_price_usd") * pl.col("quantity") - pl.col("discount_amount_usd"))
        print("total_price != unit*qty-discount (>0.05):", df.filter((calc - pl.col("total_price_usd")).abs() > 0.05).height)
        print("profit != total-cost (>0.05):", df.filter((pl.col("total_price_usd") - pl.col("cost_usd") - pl.col("profit_usd")).abs() > 0.05).height)
        print("negative profit rows:", df.filter(pl.col("profit_usd") < 0).height)
        print("margin% != profit/total*100 (>0.1):", df.filter((100 * pl.col("profit_usd") / pl.col("total_price_usd") - pl.col("profit_margin_percent")).abs() > 0.1).height)
        print("non-positive qty/price:", df.filter((pl.col("quantity") <= 0) | (pl.col("unit_price_usd") <= 0)).height)
        print("rating outside 1..5:", df.filter((pl.col("rating") < 1) | (pl.col("rating") > 5)).height)

        print("\n== PRODUCT / CUSTOMER ATTRIBUTE STABILITY (should be 1 per id) ==")
        for key, cols in {
            "product_id": ["product_name", "category", "sub_category", "brand", "unit_price_usd", "product_rating_avg", "stock_quantity", "cost_usd"],
            "customer_id": ["customer_name", "gender", "age", "customer_segment", "country", "city", "account_creation_date", "customer_loyalty_score", "total_orders_by_customer"],
        }.items():
            for c in cols:
                m = df.group_by(key).agg(pl.col(c).n_unique().alias("k"))
                print(f"{key}->{c}: ids with >1 value = {m.filter(pl.col('k') > 1).height:,} / {m.height:,}")

        print("\n== STATUS vs RETURN / PAYMENT ==")
        print(df.group_by("order_status").agg(pl.len(), pl.col("return_reason").is_not_null().sum().alias("has_reason")).sort("len", descending=True))
        print(df.group_by("order_status", "payment_status").len().sort("len", descending=True))
        print(df.group_by("order_status", "delivery_status").len().sort("len", descending=True))

        print("\n== RATING vs SENTIMENT ==")
        print(df.group_by("rating", "review_sentiment").len().sort("rating", "review_sentiment"))
        print("nulls rating/sentiment/feedback:", df["rating"].null_count(), df["review_sentiment"].null_count(), df["customer_feedback"].null_count())

        print("\n== PRODUCT NAME vs CATEGORY (sample) ==")
        print(df.group_by("product_name").agg(pl.col("category").n_unique().alias("n_cat"), pl.col("sub_category").n_unique().alias("n_sub"), pl.col("brand").n_unique().alias("n_brand"), pl.col("product_id").n_unique().alias("n_pid"), pl.len()).sort("len", descending=True).head(40))
        print("\ncategory x sub_category:")
        print(df.group_by("category", "sub_category").len().sort("category", "sub_category"))

    out = REPORTS_DIR / "profile.txt"
    out.write_text(buf.getvalue(), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
