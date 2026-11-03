import sys
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from api.main import app  # noqa: E402
from rec.config import PROCESSED_DIR  # noqa: E402
from rec.features import CUSTOMER_FEATURES  # noqa: E402
from rec.serving import DEFAULT_PROFILE, Recommender, profile_to_features  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_profile_bucketing_matches_training_features():
    df = pl.read_parquet(PROCESSED_DIR / "model_table.parquet").sample(500, seed=1)
    for row in df.iter_rows(named=True):
        p = {
            "customer_segment": row["customer_segment"], "country": row["country"], "gender": row["gender"], "age": row["age"],
            "customer_loyalty_score": row["customer_loyalty_score"],
            "tenure_days": (row["order_date"].date() - row["account_creation_date"]).days,
            "device_type": row["device_type"], "campaign_source": row["campaign_source"], "traffic_source": row["traffic_source"],
            "coupon_used": row["coupon_used"] == "true", "abandoned_cart_before": row["abandoned_cart_before"] == "true",
            "is_weekend": row["is_weekend"] == "true", "order_hour": row["order_hour"], "order_month": int(row["order_month"]),
            "session_duration_minutes": row["session_duration_minutes"], "pages_visited": row["pages_visited"],
        }
        feats = profile_to_features({**DEFAULT_PROFILE, **p})
        assert feats == {k: row[k] for k in CUSTOMER_FEATURES}


def test_health_and_options(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["items"] == 48
    o = client.get("/options").json()
    assert len(o["items"]) == 48 and "hybrid" in o["models"] and len(o["country"]) == 10


@pytest.mark.parametrize("origin", ["http://localhost:5173", "https://example.com"])
def test_cors_allows_any_origin(client, origin):
    preflight = client.options("/recommendations", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "*"
    assert "POST" in preflight.headers["access-control-allow-methods"]
    assert "content-type" in preflight.headers["access-control-allow-headers"].lower()

    response = client.post("/recommendations", json={"k": 5}, headers={"Origin": origin})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in response.headers
    assert len(response.json()["items"]) == 5


def test_recommend_shape_and_probabilities(client):
    r = client.post("/recommendations", json={"k": 5}).json()
    assert len(r["items"]) == 5 and [i["rank"] for i in r["items"]] == [1, 2, 3, 4, 5]
    assert all(0 <= i["relevance"] <= 1 for i in r["items"])
    scores = [i["final_score"] for i in r["items"]]
    assert scores == sorted(scores, reverse=True)
    assert r["items"][0]["reasons"]


def test_profit_weight_raises_profit_and_changes_list(client):
    low = client.post("/recommendations", json={"profit_weight": 0.0}).json()
    high = client.post("/recommendations", json={"profit_weight": 1.0}).json()
    assert high["comparison"]["avg_profit_reranked"] > low["comparison"]["avg_profit_reranked"]
    assert {i["category"] for i in high["items"]} == {"Electronics"}
    assert low["comparison"]["profit_uplift_pct"] == pytest.approx(0.0)


def test_every_model_works(client):
    for m in client.get("/options").json()["models"]:
        r = client.post("/recommendations", json={"model": m}).json()
        assert len(r["items"]) == 5, m


def test_fraud_block(client):
    p = {"fraud_risk_score": 95}
    blocked = client.post("/recommendations", json={"profile": p, "exclude_high_risk": True}).json()
    assert blocked["blocked"] and blocked["items"] == []
    allowed = client.post("/recommendations", json={"profile": p, "exclude_high_risk": False}).json()
    assert not allowed["blocked"] and len(allowed["items"]) == 5


def test_category_filter(client):
    r = client.post("/recommendations", json={"category": "Home", "k": 5}).json()
    assert {i["category"] for i in r["items"]} == {"Home"}


def test_validation_and_errors(client):
    assert client.post("/recommendations", json={"model": "nope"}).status_code == 400
    assert client.post("/recommendations", json={"k": 0}).status_code == 422
    assert client.post("/recommendations", json={"profile": {"age": 5}}).status_code == 422
    assert client.get("/items/does-not-exist/related").status_code == 404
    assert client.get("/insights/does-not-exist").status_code == 404


def test_related_and_insights(client):
    rel = client.get("/items/Wireless Mouse 2.4GHz/related").json()
    assert len(rel["cross_sell"]) == 5 and len(rel["price_tiers"]) == 4
    assert {c["category"] for c in rel["cross_sell"]} <= {"Home", "Sports"}
    ins = client.get("/insights").json()
    assert {"kpis", "products", "marketing", "returns", "opportunities", "findings"} <= set(ins)
    assert client.get("/models/metrics").json()["n_test"] > 80000
    assert client.get("/customers/summary").json()["pareto"]
    assert len(client.get("/customers?limit=10").json()) == 10


def test_upsell_points_to_higher_value():
    r = Recommender().recommend({}, k=48, lam=0.0)
    for it in r["items"]:
        if it["upsell"]:
            assert it["upsell"]["extra_profit"] > 0
