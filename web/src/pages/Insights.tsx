import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Json, money, moneyM, num, pct, useGet } from "../api";
import { Callout, Card, State, Table } from "../ui";

export default function Insights() {
  const { data, error } = useGet("/insights");
  if (error || !data) return <State error={error} loading={!error} />;
  const cats: Json[] = data.products.by_category;
  const disc: Json[] = data.marketing.discount_depth.rows;
  const camp: Json[] = data.marketing.campaign_source.rows;
  const reasons: Json[] = data.returns.by_reason;
  const monthly: Json[] = data.behaviour.monthly.slice(1, -1);
  const pm: Json[] = data.returns.payment_method.rows;
  const within: Json[] = data.products.within_category_item_spread;
  return (
    <>
      <h1>Business Insights</h1>
      <p className="sub">Product performance, marketing, behavior, returns and revenue opportunities.</p>

      <h2>Product performance</h2>
      <div className="grid g2">
        <Card title="Net profit by category ($M)" note="Every category has 20% of orders, so the ranking is a price effect.">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={cats.map((c) => ({ category: c.category, profit: Math.round(c.net_profit / 1e5) / 10 }))}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="category" /><YAxis /><Tooltip /><Bar dataKey="profit" fill="#4f46e5" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Items within a category are indistinguishable">
          <Table head={["Category", "Items", "ANOVA p", "Best vs worst"]}
            rows={within.map((w) => [w.category, w.items, w.anova_p_net_profit.toFixed(2), pct(w.best_vs_worst_item_gap_pct)])} />
          <p className="note">Category, not individual product, is the unit of performance.</p>
        </Card>
      </div>
      <Card title="Categories">
        <Table head={["Category", "Net profit", "Share of profit", "Profit per order", "Margin", "Return rate", "Rating"]}
          rows={cats.map((c) => [c.category, moneyM(c.net_profit), pct(c.profit_share), money(c.net_profit_per_order), `${c.avg_margin_pct.toFixed(1)}%`, pct(c.return_rate), c.avg_rating.toFixed(2)])} />
      </Card>

      <h2>Marketing effectiveness</h2>
      <div className="grid g2">
        <Card title="Discount depth: margin vs conversion" note={`Conversion does not change with discount depth (p = ${data.marketing.discount_depth.p_conversion.toFixed(2)}).`}>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={disc.map((d) => ({ discount: `${d.discount_percent}%`, margin: Number(d.avg_margin_pct.toFixed(1)), conversion: Number((d.conversion * 100).toFixed(1)) }))}>
              <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="discount" /><YAxis /><Tooltip />
              <Line isAnimationActive={false} dot={false} dataKey="margin" name="Margin %" stroke="#c0392b" strokeWidth={2} />
              <Line isAnimationActive={false} dot={false} dataKey="conversion" name="Conversion %" stroke="#12805c" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Campaign sources" note={`Conversion p = ${data.marketing.campaign_source.p_conversion.toFixed(2)}. No channel stands out.`}>
          <Table head={["Source", "Orders", "Conversion", "Profit / order"]}
            rows={camp.map((c) => [c.value, num(c.orders), pct(c.conversion), money(c.net_profit_per_order, 2)])} />
        </Card>
      </div>

      <h2>Customer behavior</h2>
      <Card title="Monthly orders" note={`Flat volume (coefficient of variation ${pct(data.behaviour.monthly_orders_cv)}), no seasonality. Partial first and last months are excluded.`}>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={monthly.map((m) => ({ month: m.ym, orders: m.orders }))}>
            <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="month" /><YAxis domain={[0, "auto"]} /><Tooltip /><Line isAnimationActive={false} dataKey="orders" stroke="#4f46e5" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <h2>Return patterns</h2>
      <div className="grid g2">
        <Card title="Returns by reason" note={`${pct(data.returns.totals.return_rate)} of orders returned, ${moneyM(data.returns.totals.profit_lost)} profit lost.`}>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={reasons.map((r) => ({ reason: r.return_reason, returns: r.returns }))} layout="vertical" margin={{ left: 60 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} /><XAxis type="number" /><YAxis type="category" dataKey="reason" width={150} /><Tooltip /><Bar dataKey="returns" fill="#b45309" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Return rate by payment method" note="The only detectable driver, and small, significant only because of the 1M-row sample.">
          <Table head={["Payment method", "Orders", "Return rate"]} rows={pm.map((p) => [p.value, num(p.orders), pct(p.return_rate, 2)])} />
        </Card>
      </div>

      <h2>Revenue opportunities</h2>
      <Callout kind="info">Scenario estimates with stated assumptions. They are not forecasts and each one needs validation.</Callout>
      <Card>
        <Table head={["Opportunity", "Estimated profit", "Basis", "Assumption"]}
          rows={[...data.opportunities].sort((a: Json, b: Json) => b.value_usd - a.value_usd).map((o: Json) => [<b key="t">{o.title}</b>, <b key="v" className="up">{moneyM(o.value_usd)}</b>, o.basis, o.assumption])} />
      </Card>
    </>
  );
}

