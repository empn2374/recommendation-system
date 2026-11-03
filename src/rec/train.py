"""Train all recommenders on the training period.

- scikit-learn models are saved to data/processed/models_sklearn/ (local only, used for tests)
- numpy-only equivalents (rec.lite) are saved to artifacts/models/lite_models.pkl (this is what the API ships and loads)
"""
import json
import pickle

import joblib
import polars as pl

from rec.config import ARTIFACTS_DIR, PROCESSED_DIR
from rec.features import CUSTOMER_FEATURES, SPLIT_DATE, SPLIT_DATE_STR, build_item_table
from rec.lite import to_lite
from rec.models import fit_models

MODEL_DIR = ARTIFACTS_DIR / "models"
SKLEARN_DIR = PROCESSED_DIR / "models_sklearn"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SKLEARN_DIR.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(PROCESSED_DIR / "model_table.parquet")
    train = df.filter(pl.col("order_date") < SPLIT_DATE)
    items = build_item_table(train)["item_key"].to_list()
    models = fit_models(train, items)

    for old in MODEL_DIR.glob("*.joblib"):
        old.unlink()
    for name, model in models.items():
        joblib.dump(model, SKLEARN_DIR / (name.replace("(", "_").replace(")", "").replace("+", "_") + ".joblib"), compress=3)

    lite = to_lite(models)
    with open(MODEL_DIR / "lite_models.pkl", "wb") as f:
        pickle.dump(lite, f, protocol=pickle.HIGHEST_PROTOCOL)
    size = (MODEL_DIR / "lite_models.pkl").stat().st_size

    meta = {"split_date": SPLIT_DATE_STR, "n_train": train.height, "items": items, "features": CUSTOMER_FEATURES,
            "models": list(lite), "lite_models_kb": round(size / 1024, 1)}
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {len(lite)} lite models ({size / 1024:.1f} KB) and {len(models)} sklearn models")


if __name__ == "__main__":
    main()
