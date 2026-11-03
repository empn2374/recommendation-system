"""CSV -> Parquet conversion (first step of the pipeline)."""
import polars as pl

from rec.config import RAW_CSV, RAW_PARQUET


def main() -> None:
    df = pl.read_csv(RAW_CSV, infer_schema_length=200_000, try_parse_dates=False)
    print(f"rows={df.height:,} cols={df.width}")
    print(df.schema)
    df.write_parquet(RAW_PARQUET, compression="zstd")
    print(f"written {RAW_PARQUET} ({RAW_PARQUET.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
