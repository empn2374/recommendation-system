"""FastAPI service. Loads precomputed artifacts once and serves recommendations, insights and evaluation."""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from rec.serving import Recommender  # noqa: E402

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["rec"] = Recommender()
    yield


app = FastAPI(title="Intelligent Product Recommendation Platform", version="1.0.0", lifespan=lifespan,
              description="Profile-based, business-aware recommendations over a 48-item catalogue.")

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])


def rec() -> Recommender:
    return state["rec"]


class Profile(BaseModel):
    customer_segment: str = "Regular"
    country: str = "United States"
    gender: str = "Female"
    age: int = Field(40, ge=18, le=100)
    customer_loyalty_score: float = Field(50.0, ge=0, le=100)
    tenure_days: int = Field(700, ge=0)
    device_type: str = "Mobile"
    campaign_source: str = "Organic"
    traffic_source: str = "Search"
    coupon_used: bool = False
    abandoned_cart_before: bool = False
    is_weekend: bool = False
    order_hour: int = Field(12, ge=0, le=23)
    order_month: int = Field(6, ge=1, le=12)
    session_duration_minutes: float = Field(30.0, ge=0)
    pages_visited: int = Field(10, ge=0)
    fraud_risk_score: float = Field(30.0, ge=0, le=100)


class RecommendRequest(BaseModel):
    profile: Profile = Profile()
    model: str = "hybrid"
    k: int = Field(5, ge=1, le=48)
    profit_weight: float = Field(0.25, ge=0, le=1, description="lambda: weight of net profit vs relevance")
    return_penalty: float = Field(0.0, ge=0, le=5)
    fraud_penalty: float = Field(0.0, ge=0, le=5)
    sentiment_boost: float = Field(0.0, ge=0, le=5)
    exclude_high_risk: bool = False
    risk_threshold: float = Field(80.0, ge=0, le=100)
    category: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "items": len(rec().names), "models": list(rec().models), "train_rows": rec().meta["n_train"],
            "split_date": rec().meta["split_date"]}


@app.get("/options")
def options():
    return rec().options()


@app.get("/customers")
def customers(limit: int = Query(50, ge=1, le=300), value_tier: str | None = None):
    s = rec().samples
    if value_tier:
        s = [r for r in s if r["value_tier"] == value_tier]
    return s[:limit]


@app.get("/customers/summary")
def customers_summary():
    return rec().customer_intel


@app.post("/recommendations")
def recommendations(req: RecommendRequest):
    try:
        return rec().recommend(req.profile.model_dump(), req.model, req.k, req.profit_weight, req.return_penalty,
                               req.fraud_penalty, req.sentiment_boost, req.exclude_high_risk, req.risk_threshold, req.category)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/items")
def items():
    return rec().items


@app.get("/items/{item}/related")
def related(item: str):
    try:
        return rec().related(item)
    except KeyError:
        raise HTTPException(404, f"unknown item {item}")


@app.get("/insights")
def insights():
    return rec().insights


@app.get("/insights/{section}")
def insights_section(section: str):
    data = rec().insights
    if section not in data:
        raise HTTPException(404, f"unknown section {section}; choose from {list(data)}")
    return data[section]


@app.get("/models/metrics")
def metrics():
    return rec().metrics
