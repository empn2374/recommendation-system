import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rec.features import CUSTOMER_FEATURES  # noqa: E402

SKLEARN_DIR = ROOT / "data" / "processed" / "models_sklearn"
LITE = ROOT / "artifacts" / "models" / "lite_models.pkl"
TABLE = ROOT / "data" / "processed" / "model_table.parquet"

pytestmark = pytest.mark.skipif(not (SKLEARN_DIR.exists() and TABLE.exists()), reason="needs local training outputs")

PAIRS = {
    "popularity": "popularity.joblib",
    "trending_90d": "trending_90d.joblib",
    "segment_popularity": "segment_popularity.joblib",
    "profile_naive_bayes": "profile_naive_bayes.joblib",
    "profile_logreg": "profile_logreg.joblib",
    "hybrid": "hybrid_logreg_popularity.joblib",
}


@pytest.mark.parametrize("name", list(PAIRS))
def test_lite_model_matches_sklearn_model(name):
    lite = pickle.load(open(LITE, "rb"))[name]
    ref = joblib.load(SKLEARN_DIR / PAIRS[name])
    rows = pl.read_parquet(TABLE).sample(1500, seed=7)
    expected = ref.score(rows)
    for r, row in enumerate(rows.select(CUSTOMER_FEATURES).iter_rows(named=True)):
        got = lite.score(row)
        np.testing.assert_allclose(got, expected[r], atol=1e-9, err_msg=f"{name} row {r}")


def test_lite_models_do_not_import_sklearn():
    import subprocess
    code = ("import sys; sys.path.insert(0, r'%s'); import pickle; import rec.serving; "
            "from rec.serving import Recommender; Recommender(); "
            "assert not any(m.startswith(('sklearn', 'scipy')) for m in sys.modules), 'sklearn/scipy imported'" % (ROOT / "src"))
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
