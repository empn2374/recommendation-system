import { Json, money, moneyM, num, pct, useGet } from "../api";
import { Callout, Card, Kpi, State } from "../ui";

export default function Overview() {
  const { data, error } = useGet("/insights");
  const m = useGet("/models/metrics");
  if (error || !data) return <State error={error} loading={!error} />;
  const k = data.kpis;
  const paired = m.data?.paired_vs_popularity;
  const pareto: Json[] = m.data?.pareto ?? [];
  const base = pareto.find((p) => p.lambda === 0);
  const sweet = pareto.find((p) => p.lambda === 0.25);
  return (
    <>
      <h1>Intelligent Product Recommendation Platform</h1>
      <p className="sub">
        End-to-end analysis and recommender for a 1M-order synthetic e-commerce dataset ({data.meta.start} to {data.meta.end}).
      </p>

      <div className="grid g5" style={{ marginBottom: 16 }}>
        <Kpi label="Orders" value={num(k.orders)} />
        <Kpi label="Net revenue" value={moneyM(k.net_revenue_completed)} sub="completed orders" />
        <Kpi label="Net profit" value={moneyM(k.net_profit_completed)} sub={`${k.avg_margin_pct.toFixed(1)}% avg margin`} />
        <Kpi label="Returned" value={pct(k.return_rate)} sub={`${moneyM(data.returns.totals.profit_lost)} profit lost`} />
        <Kpi label="Status conflicts" value={pct(k.status_conflict_rate)} sub="rows with contradictory states" />
      </div>

      <Callout kind="warn">
        <b>The honest headline.</b> This dataset is synthetic and almost entirely random: every customer has one order, and product choice is independent of
        customer attributes. Personalization therefore cannot beat popularity
        {paired && <> (paired difference in hit@5: {paired.diff >= 0 ? "+" : ""}{(paired.diff * 100).toFixed(2)} points, 95% CI [{(paired.ci95[0] * 100).toFixed(2)}, {(paired.ci95[1] * 100).toFixed(2)}], which includes zero)</>}.
        The measurable value lies in the <b>business layer</b>.
      </Callout>

      <div className="grid g3">
        <Card title="1. Profit-aware ranking works">
          {base && sweet ? (
            <p>At profit weight 0.25, profit captured per customer rises from <b>{money(base.captured_profit_per_customer, 2)}</b> to{" "}
              <b>{money(sweet.captured_profit_per_customer, 2)}</b> (<span className="up">+{pct(sweet.captured_profit_per_customer / base.captured_profit_per_customer - 1, 0)}</span>)
              while the hit rate stays at {pct(sweet.hit, 1)}.</p>
          ) : <p>Loading...</p>}
        </Card>
        <Card title="2. Category mix is the real driver">
          <p>Electronics earns <b>{money(data.products.by_category[0].net_profit_per_order)}</b> per order against{" "}
            <b>{money(data.products.by_category[data.products.by_category.length - 1].net_profit_per_order)}</b> for Clothing, so steering toward higher-value
            categories matters more than predicting taste.</p>
        </Card>
        <Card title="3. Discounts destroy margin">
          <p>Margin falls from 45% at no discount to 27% at 25% off, with <b>no conversion gain</b>. Capping discounts at 10% is worth about{" "}
            <b>{moneyM(data.opportunities.find((o: Json) => o.id === "discount_cap")?.value_usd ?? 0)}</b> if this holds in an A/B test.</p>
        </Card>
      </div>

      <h2>Key findings</h2>
      <Card>
        <ul className="clean">{data.findings.map((f: string, i: number) => <li key={i}>{f}</li>)}</ul>
      </Card>

      <h2>What was built</h2>
      <div className="grid g3">
        <Card title="Data and models"><ul className="clean">
          <li>Profiling, quality flags and cleaning of 1,000,123 rows</li>
          <li>Seven recommenders compared on a temporal split</li>
          <li>Business re-ranker with profit, return, fraud and sentiment terms</li>
        </ul></Card>
        <Card title="Customer and business analysis"><ul className="clean">
          <li>Value tiers, clustering and a CLV scenario model</li>
          <li>Product, marketing, behavior and returns insights</li>
          <li>Five sized revenue opportunities</li>
        </ul></Card>
        <Card title="Platform"><ul className="clean">
          <li>FastAPI service serving precomputed artifacts</li>
          <li>React interface (this app)</li>
          <li>Scalable architecture design, with a zero-cost deployment</li>
        </ul></Card>
      </div>
    </>
  );
}
