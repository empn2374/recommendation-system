"""Offline evaluation on a temporal split (train < 2025-11-01 <= test). Writes reports/evaluation.txt and artifacts/metrics.json."""
import json
import math

import numpy as np
import polars as pl

from rec.config import ARTIFACTS_DIR, PROCESSED_DIR, REPORTS_DIR
from rec.features import SPLIT_DATE, SPLIT_DATE_STR, build_item_table
from rec.models import business_rerank, fit_models

KS = (5, 10)


def topk(S: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(S, idx, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(idx, order, axis=1)


def metrics(S: np.ndarray, y: np.ndarray, net_profit: np.ndarray, item_value: np.ndarray, k: int) -> dict:
    top = topk(S, k)
    match = top == y[:, None]
    hit = match.any(axis=1)
    rank = match.argmax(axis=1)
    ndcg = np.where(hit, 1 / np.log2(rank + 2), 0.0)
    p = hit.mean()
    return {
        "hit": float(p),
        "hit_ci95": float(1.96 * math.sqrt(p * (1 - p) / len(y))),
        "ndcg": float(ndcg.mean()),
        "coverage": float(len(np.unique(top)) / S.shape[1]),
        "captured_profit_per_customer": float(np.where(hit, net_profit, 0.0).mean()),
        "avg_list_item_value": float(item_value[top].mean()),
        "_hit_vec": hit,
    }


def main() -> None:
    df = pl.read_parquet(PROCESSED_DIR / "model_table.parquet")
    train = df.filter(pl.col("order_date") < SPLIT_DATE)
    test = df.filter((pl.col("order_date") >= SPLIT_DATE) & pl.col("is_positive_purchase"))
    items_df = build_item_table(train)
    items = items_df["item_key"].to_list()
    value = items_df["net_profit_per_order"].to_numpy()
    category = items_df["category"].to_list()
    n_items = len(items)
    print(f"train rows={train.height:,} test purchases={test.height:,} items={n_items}")

    models = fit_models(train, items)
    pop = models["popularity"]

    y = pop.y(test)
    net_profit = test["profit_usd"].to_numpy()
    scores = {n: m.score(test) for n, m in models.items()}

    out: dict = {"split_date": SPLIT_DATE_STR, "n_train": train.height, "n_test": test.height, "models": {}, "pareto": []}
    lines = [f"train rows={train.height:,} | test completed purchases={test.height:,} | items={n_items} | random hit@5={5 / n_items:.4f}\n"]

    pop_hits = {}
    for k in KS:
        lines.append(f"=== Relevance metrics @K={k} (95% CI on hit rate) ===")
        lines.append(f"{'model':28s} {'hit':>8s} {'+-CI':>8s} {'NDCG':>8s} {'cover':>7s} {'captured$/cust':>15s} {'lift vs pop':>12s}")
        for name, S in scores.items():
            m = metrics(S, y, net_profit, value, k)
            if name == "popularity":
                pop_hits[k] = m["_hit_vec"]
            lift = (m["hit"] - pop_hits[k].mean()) / pop_hits[k].mean() if k in pop_hits else float("nan")
            lines.append(f"{name:28s} {m['hit']:8.4f} {m['hit_ci95']:8.4f} {m['ndcg']:8.4f} {m['coverage']:7.2f} {m['captured_profit_per_customer']:15.2f} {lift:11.2%}")
            out["models"].setdefault(name, {})[f"@{k}"] = {kk: vv for kk, vv in m.items() if not kk.startswith("_")}
        lines.append("")

    # does personalisation beat popularity? paired difference on the same test rows
    k = 5
    best = "profile_logreg"
    d = metrics(scores[best], y, net_profit, value, k)["_hit_vec"].astype(float) - pop_hits[k].astype(float)
    se = d.std(ddof=1) / math.sqrt(len(d))
    lines.append(f"Paired test {best} vs popularity hit@5: diff={d.mean():+.5f}  95% CI [{d.mean() - 1.96 * se:+.5f}, {d.mean() + 1.96 * se:+.5f}]")
    out["paired_vs_popularity"] = {"model": best, "diff": float(d.mean()), "ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)]}

    # relevance vs profit trade-off for the business re-ranker on top of the hybrid
    lines.append("\n=== Business re-ranking (hybrid relevance + net profit per order), K=5 ===")
    lines.append(f"{'lambda':>6s} {'hit@5':>8s} {'NDCG@5':>8s} {'cover':>7s} {'captured$/cust':>15s} {'avg item value $':>17s}  top-5 categories")
    rel = scores["hybrid(logreg+popularity)"]
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        S = business_rerank(rel, value, lam)
        m = metrics(S, y, net_profit, value, 5)
        cats = {}
        for i in topk(S, 5)[0]:
            cats[category[i]] = cats.get(category[i], 0) + 1
        lines.append(f"{lam:6.2f} {m['hit']:8.4f} {m['ndcg']:8.4f} {m['coverage']:7.2f} {m['captured_profit_per_customer']:15.2f} {m['avg_list_item_value']:17.2f}  {cats}")
        out["pareto"].append({"lambda": lam, **{kk: vv for kk, vv in m.items() if not kk.startswith("_")}, "top5_categories": cats})

    # how much do item-level business stats actually differ?
    lines.append("\nItem net profit per order: min={:.2f} max={:.2f} | category means:".format(value.min(), value.max()))
    for c, g in items_df.group_by("category").agg(pl.col("net_profit_per_order").mean()).sort("net_profit_per_order", descending=True).iter_rows():
        lines.append(f"  {c:12s} {g:8.2f}")

    text = "\n".join(lines)
    print(text)
    (REPORTS_DIR / "evaluation.txt").write_text(text, encoding="utf-8")
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
