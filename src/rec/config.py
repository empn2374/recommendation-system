from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "ecommerce_dataset_+1m.csv"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

RAW_PARQUET = INTERIM_DIR / "raw.parquet"
CLEAN_PARQUET = INTERIM_DIR / "clean.parquet"

SEED = 42

for d in (INTERIM_DIR, PROCESSED_DIR, ARTIFACTS_DIR, REPORTS_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:  # read-only filesystem on a deployment host: the API only reads artifacts
        pass
