import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Json, money, num, pct, useGet } from "../api";
import { Callout, Card, Kpi, State, Table } from "../ui";

export default function Customers() {
  const { data, error } = useGet("/customers/summary");
  if (error || !data) return <State error={error} loading={!error} />;
  const tiers: Json[] = data.value_tiers;
  const pareto = Object.entries(data.pareto as Record<string, number>).map(([k, v]) => ({ top: `Top ${k}%`, share: Math.round(v * 1000) / 10 }));
  const beh = Object.entries(data.behaviour as Json);
  const clv = data.clv;
  return (
    <>
      <h1>Customer Intelligence</h1>
      <p className="sub">Value tiers, clustering, behavior tests and a lifetime-value scenario model.</p>
      <Callout kind="warn">
        Every customer has a single order, so RFM and probabilistic CLV models cannot be fitted. Value tiers use realized net profit, and CLV is a
        transparent scenario with stated assumptions.
      </Callout>

      <div className="grid g4" style={{ marginBottom: 16 }}>
        <Kpi label="Top 10% of customers" value={pct(data.pareto["10"], 0)} sub="of net profit" />
        <Kpi label="High-value threshold" value={money(data.high_value_threshold)} sub="net profit per customer" />
        <Kpi label="Predictability (AUC)" value={data.high_value_auc_profile_only.toFixed(3)} sub="0.5 = no signal" />
        <Kpi label="Base-case 3-year CLV" value={money(clv.mean)} sub={`median ${money(clv.median)}`} />
      </div>

      <div className="grid g2">
        <Card title="Value tiers" note="Net profit = profit on completed orders only.">
          <Table head={["Tier", "Customers", "Avg profit", "Share of profit"]}
            rows={tiers.map((t) => [t.value_tier, num(t.customers), money(t.avg_profit, 1), pct(t.profit_share)])} />
        </Card>
        <Card title="Pareto: share of net profit by top customers (%)">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={pareto}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="top" /><YAxis unit="%" /><Tooltip /><Bar dataKey="share" fill="#4f46e5" radius={[4, 4, 0, 0]} /></BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Callout kind="info">
        High-value customers <b>cannot be identified from who they are</b> (AUC {data.high_value_auc_profile_only.toFixed(3)} using age, loyalty, tenure, session and fraud score).
        Value comes from what is bought: {pct(data.high_value_by_category.Electronics, 0)} of Electronics customers are in the top 10%, against {pct(data.high_value_by_category.Clothing, 0)} of Clothing customers.
      </Callout>

      <div className="grid g2">
        <Card title={`Clusters (k = ${data.cluster_k})`} note={`Silhouette is only ${Math.min(...Object.values(data.silhouette as Record<string, number>)).toFixed(2)} to ${Math.max(...Object.values(data.silhouette as Record<string, number>)).toFixed(2)} for every k from 3 to 8 (near 0 = no real cluster structure), so these are slices of random data, not distinct customer types.`}>
          <Table head={["Cluster", "Customers", "Avg net profit", "Qty", "Avg price", "Age"]}
            rows={data.clusters.map((c: Json) => [c.cluster, num(c.customers), money(c.avg_net_profit), c.qty.toFixed(1), money(c.price), c.age.toFixed(0)])} />
        </Card>
        <Card title="CLV scenarios (3-year)" note={`Assumes retention ${clv.retention}, ${money(clv.mean_profit_per_order, 2)} net profit per order. Retention and order rate are assumptions, not estimates.`}>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={Object.entries(clv.scenarios as Record<string, number>).map(([k, v]) => ({ scenario: k, clv: Math.round(v) }))}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="scenario" /><YAxis /><Tooltip /><Bar dataKey="clv" fill="#12805c" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <div className="grid g2">
        <Card title="Does the supplied VIP / Premium / Regular label mean anything?">
          <Table head={["Label", "Customers", "Avg net profit", "Loyalty score"]}
            rows={data.given_segment.map((g: Json) => [g.customer_segment, num(g.n), money(g.avg_net_profit, 1), g.loyalty.toFixed(1)])} />
          <p className="note">No. A VIP is worth the same as a Regular customer.</p>
        </Card>
        <Card title="Behavior tests (p-values)" note="With nine dimensions tested, one p-value below 0.05 is expected by chance.">
          <Table head={["Dimension", "Conversion p", "Profit p", "Profit spread"]}
            rows={beh.map(([k, v]: [string, Json]) => [k.replace(/_/g, " "), v.p_conversion.toFixed(3), v.p_profit.toFixed(3), pct(v.profit_spread)])} />
        </Card>
      </div>

      <Card title="Net profit per order and unit price by cluster">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data.clusters.map((c: Json) => ({ name: c.cluster, profit: Math.round(c.avg_net_profit), price: Math.round(c.price) }))}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="name" /><YAxis /><Tooltip /><Legend />
            <Bar dataKey="profit" name="Avg net profit" fill="#4f46e5" radius={[4, 4, 0, 0]} />
            <Bar dataKey="price" name="Avg unit price" fill="#b45309" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </>
  );
}
