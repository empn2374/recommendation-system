import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Json, money, num, pct, useGet } from "../api";
import { Callout, Card, Kpi, State, Table } from "../ui";

export default function Evaluation() {
  const { data, error } = useGet("/models/metrics");
  if (error || !data) return <State error={error} loading={!error} />;
  const names = Object.keys(data.models);
  const random5 = data.models.random["@5"].hit;
  const pop5 = data.models.popularity["@5"].hit;
  const paired = data.paired_vs_popularity;
  const pareto: Json[] = data.pareto;
  return (
    <>
      <h1>Model Evaluation</h1>
      <p className="sub">
        Temporal split: trained on {num(data.n_train)} orders before {data.split_date}, tested on {num(data.n_test)} completed purchases from the last three months.
        The task is to rank the 48 products, so random guessing gives a hit@5 of about {pct(5 / 48)}.
      </p>

      <div className="grid g4" style={{ marginBottom: 16 }}>
        <Kpi label="Random hit@5" value={pct(random5)} />
        <Kpi label="Popularity hit@5" value={pct(pop5)} />
        <Kpi label="Best model hit@5" value={pct(Math.max(...names.map((n) => data.models[n]["@5"].hit)))} />
        <Kpi label="Personalization lift" value={`${paired.diff >= 0 ? "+" : ""}${(paired.diff * 100).toFixed(2)} pts`} sub={`95% CI [${(paired.ci95[0] * 100).toFixed(2)}, ${(paired.ci95[1] * 100).toFixed(2)}]`} />
      </div>

      <Callout kind="warn">
        <b>No model meaningfully beats popularity.</b> The confidence interval for the best profile model against popularity includes zero. This is a property of the data
        (one order per customer, product independent of customer attributes), not a modelling defect. Top-1 accuracy is about 2.3% for every real model, against 2.1% for chance.
      </Callout>

      <Card title="Relevance metrics">
        <Table head={["Model", "Hit@5", "±95% CI", "NDCG@5", "Hit@10", "Coverage@5", "Profit captured / customer"]}
          rows={names.map((n) => {
            const m5 = data.models[n]["@5"]; const m10 = data.models[n]["@10"];
            return [<b key="n">{n}</b>, pct(m5.hit, 2), pct(m5.hit_ci95, 2), m5.ndcg.toFixed(4), pct(m10.hit, 2), pct(m5.coverage, 0), money(m5.captured_profit_per_customer, 2)];
          })} />
        <p className="note">Coverage is the share of the 48-item catalogue that appears in any customer's top-5 list. The random model scores every item equally, so it returns one fixed arbitrary list; its profit figure only reflects which items happened to be in it.</p>
      </Card>

      <div className="grid g2">
        <Card title="Relevance vs profit trade-off (business re-ranker, top-5)" note="Profit captured = average actual profit of test purchases whose item appears in the recommended list.">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={pareto.map((p) => ({ lambda: p.lambda, hit: Number((p.hit * 100).toFixed(2)), profit: Number(p.captured_profit_per_customer.toFixed(2)) }))}>
              <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="lambda" label={{ value: "profit weight λ", position: "insideBottom", offset: -4 }} />
              <YAxis yAxisId="l" domain={[0, 14]} unit="%" /><YAxis yAxisId="r" orientation="right" /><Tooltip /><Legend verticalAlign="top" />
              <Line isAnimationActive={false} yAxisId="l" dataKey="hit" name="Hit@5 (%)" stroke="#4f46e5" strokeWidth={2} />
              <Line isAnimationActive={false} yAxisId="r" dataKey="profit" name="Profit captured ($)" stroke="#12805c" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Re-ranker by λ">
          <Table head={["λ", "Hit@5", "Profit / customer", "Coverage", "Top-5 categories"]}
            rows={pareto.map((p) => [p.lambda.toFixed(2), pct(p.hit, 2), money(p.captured_profit_per_customer, 2), pct(p.coverage, 0), Object.entries(p.top5_categories).map(([c, n]) => `${c} ${n}`).join(", ")])} />
          <p className="note"><b>λ = 0.25 is the sweet spot:</b> about +22% profit for roughly the same hit rate. Beyond 0.5 the list collapses into Electronics and the hit rate falls by about 25%.</p>
        </Card>
      </div>

      <Card title="Method">
        <ul className="clean">
          <li>Training and item statistics use only orders before {data.split_date}. Evaluation uses only later orders, so there is no leakage.</li>
          <li>Positive signal = Completed order. Returned, cancelled and pending orders are not purchases.</li>
          <li>Models: random, popularity, trending (90 days), segment popularity, naive Bayes and logistic regression on 16 profile/context features, and a hybrid blend with popularity.</li>
          <li>Collaborative filtering and co-purchase rules were not built because they need repeat customers or baskets, which the data does not contain.</li>
        </ul>
      </Card>
    </>
  );
}

