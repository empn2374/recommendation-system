import { useEffect, useMemo, useState } from "react";
import { api, Json, money, pct, useGet } from "../api";
import { Callout, Card, State } from "../ui";

const MODEL_LABELS: Record<string, string> = {
  hybrid: "Hybrid (profile model + popularity)",
  profile_logreg: "Profile: logistic regression",
  profile_naive_bayes: "Profile: naive Bayes",
  segment_popularity: "Segment popularity",
  popularity: "Popularity",
  trending_90d: "Trending (last 90 days)",
};

const SAMPLE_FIELDS = ["customer_segment", "country", "gender", "age", "customer_loyalty_score", "tenure_days", "device_type",
  "campaign_source", "traffic_source", "coupon_used", "abandoned_cart_before", "is_weekend", "order_hour", "order_month",
  "session_duration_minutes", "pages_visited", "fraud_risk_score"];

function Select({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (v: string) => void }) {
  return (
    <div className="field">
      <label>{label}</label>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => <option key={o}>{o}</option>)}
      </select>
    </div>
  );
}

function NumberField({ label, value, onChange, min, max, step = 1 }: { label: string; value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number }) {
  return (
    <div className="field">
      <label>{label}</label>
      <input type="number" value={value} min={min} max={max} step={step} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, hint }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; hint?: string }) {
  return (
    <div className="field">
      <label>{label}: <b>{value}</b></label>
      <input type="range" value={value} min={min} max={max} step={step} onChange={(e) => onChange(Number(e.target.value))} />
      {hint && <div className="note" style={{ margin: 0 }}>{hint}</div>}
    </div>
  );
}

export default function Playground() {
  const opts = useGet("/options");
  const samples = useGet("/customers?limit=300");
  const [profile, setProfile] = useState<Json>(null);
  const [sampleIdx, setSampleIdx] = useState(-1);
  const [model, setModel] = useState("hybrid");
  const [k, setK] = useState(5);
  const [lam, setLam] = useState(0.25);
  const [wReturn, setWReturn] = useState(0);
  const [wFraud, setWFraud] = useState(0);
  const [wSent, setWSent] = useState(0);
  const [block, setBlock] = useState(false);
  const [threshold, setThreshold] = useState(80);
  const [category, setCategory] = useState("All");
  const [result, setResult] = useState<Json>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { if (opts.data && !profile) setProfile(opts.data.defaults); }, [opts.data, profile]);

  const body = useMemo(() => profile && ({
    profile, model, k, profit_weight: lam, return_penalty: wReturn, fraud_penalty: wFraud, sentiment_boost: wSent,
    exclude_high_risk: block, risk_threshold: threshold, category: category === "All" ? null : category,
  }), [profile, model, k, lam, wReturn, wFraud, wSent, block, threshold, category]);

  useEffect(() => {
    if (!body) return;
    const t = setTimeout(() => {
      api("/recommendations", body).then((r) => { setResult(r); setError(null); }).catch((e) => setError(String(e.message)));
    }, 200);
    return () => clearTimeout(t);
  }, [body]);

  const pickSample = (i: number) => {
    setSampleIdx(i);
    if (i < 0) { setProfile(opts.data.defaults); return; }
    const s = samples.data[i];
    const p: Json = {};
    SAMPLE_FIELDS.forEach((f) => { p[f] = s[f]; });
    (["coupon_used", "abandoned_cart_before", "is_weekend"] as const).forEach((f) => { p[f] = s[f] === true || s[f] === "true"; });
    setProfile(p);
  };
  const set = (key: string) => (v: Json) => { setProfile({ ...profile, [key]: v }); setSampleIdx(-1); };

  if (opts.error || samples.error) return <State error={opts.error ?? samples.error} loading={false} />;
  if (!opts.data || !samples.data || !profile) return <State error={null} loading />;
  const o = opts.data;
  const cur = sampleIdx >= 0 ? samples.data[sampleIdx] : null;
  const maxScore = result?.items?.length ? Math.max(...result.items.map((i: Json) => i.relevance)) : 1;

  return (
    <>
      <h1>Recommendation Playground</h1>
      <p className="sub">Pick a customer profile, choose a model and tune the business objective. The list re-ranks live.</p>
      <div className="layout">
        <div>
          <Card title="Customer">
            <div className="field">
              <label>Real customer from the test period</label>
              <select value={sampleIdx} onChange={(e) => pickSample(Number(e.target.value))}>
                <option value={-1}>Custom profile</option>
                {samples.data.map((s: Json, i: number) => (
                  <option key={i} value={i}>{s.customer_id} · {s.country} · {s.value_tier}</option>
                ))}
              </select>
            </div>
            {cur && (
              <p className="note" style={{ marginTop: 0 }}>
                Actually bought: <b>{cur.actual_item}</b> ({cur.actual_category}), status {cur.order_status}, profit {money(cur.profit_usd, 0)}.
                CLV proxy {money(cur.clv_proxy)} · {cur.cluster}
              </p>
            )}
            <div className="grid g2">
              <Select label="Segment" value={profile.customer_segment} options={o.customer_segment} onChange={set("customer_segment")} />
              <Select label="Country" value={profile.country} options={o.country} onChange={set("country")} />
              <Select label="Gender" value={profile.gender} options={o.gender} onChange={set("gender")} />
              <NumberField label="Age" value={profile.age} min={18} max={100} onChange={set("age")} />
              <NumberField label="Loyalty (0-100)" value={profile.customer_loyalty_score} min={0} max={100} onChange={set("customer_loyalty_score")} />
              <NumberField label="Tenure (days)" value={profile.tenure_days} min={0} onChange={set("tenure_days")} />
              <Select label="Device" value={profile.device_type} options={o.device_type} onChange={set("device_type")} />
              <Select label="Campaign" value={profile.campaign_source} options={o.campaign_source} onChange={set("campaign_source")} />
              <Select label="Traffic" value={profile.traffic_source} options={o.traffic_source} onChange={set("traffic_source")} />
              <NumberField label="Hour" value={profile.order_hour} min={0} max={23} onChange={set("order_hour")} />
              <NumberField label="Month" value={profile.order_month} min={1} max={12} onChange={set("order_month")} />
              <NumberField label="Session (min)" value={profile.session_duration_minutes} min={0} onChange={set("session_duration_minutes")} />
              <NumberField label="Pages" value={profile.pages_visited} min={0} onChange={set("pages_visited")} />
              <NumberField label="Fraud score" value={profile.fraud_risk_score} min={0} max={100} onChange={set("fraud_risk_score")} />
            </div>
            {(["coupon_used", "abandoned_cart_before", "is_weekend"] as const).map((f) => (
              <label className="check" key={f}>
                <input type="checkbox" checked={profile[f]} onChange={(e) => set(f)(e.target.checked)} />{f.replace(/_/g, " ")}
              </label>
            ))}
          </Card>

          <Card title="Model and business objective">
            <div className="field">
              <label>Model</label>
              <select value={model} onChange={(e) => setModel(e.target.value)}>
                {o.models.map((m: string) => <option key={m} value={m}>{MODEL_LABELS[m] ?? m}</option>)}
              </select>
            </div>
            <Slider label="Profit weight (λ)" value={lam} min={0} max={1} step={0.05} onChange={setLam}
              hint="0 = pure relevance, 1 = pure net profit per order" />
            <Slider label="Return-risk penalty" value={wReturn} min={0} max={5} step={0.25} onChange={setWReturn} />
            <Slider label="Fraud penalty (item level)" value={wFraud} min={0} max={5} step={0.25} onChange={setWFraud} />
            <Slider label="Sentiment boost" value={wSent} min={0} max={5} step={0.25} onChange={setWSent} />
            <Slider label="Items to show (K)" value={k} min={1} max={10} step={1} onChange={setK} />
            <div className="field">
              <label>Category filter</label>
              <select value={category} onChange={(e) => setCategory(e.target.value)}>
                {["All", ...o.categories].map((c: string) => <option key={c}>{c}</option>)}
              </select>
            </div>
            <label className="check"><input type="checkbox" checked={block} onChange={(e) => setBlock(e.target.checked)} />Withhold recommendations for high-risk orders</label>
            {block && <Slider label="Risk threshold" value={threshold} min={0} max={100} step={5} onChange={setThreshold} />}
          </Card>
        </div>

        <div>
          {error && <div className="error">{error}</div>}
          {result?.blocked && <Callout kind="warn"><b>Withheld.</b> {result.message}</Callout>}
          {result && !result.blocked && (
            <>
              <div className="grid g3" style={{ marginBottom: 16 }}>
                <Card title="Profit of this list">
                  <div className="kpi-value">{money(result.comparison.avg_profit_reranked)}</div>
                  <div className="kpi-sub">avg net profit per order across {result.items.length} items</div>
                </Card>
                <Card title="vs. relevance-only list">
                  <div className={`kpi-value ${result.comparison.profit_uplift_pct >= 0 ? "up" : "down"}`}>
                    {result.comparison.profit_uplift_pct >= 0 ? "+" : ""}{pct(result.comparison.profit_uplift_pct, 0)}
                  </div>
                  <div className="kpi-sub">profit, from {money(result.comparison.avg_profit_baseline)}</div>
                </Card>
                <Card title="Relevance cost">
                  <div className={`kpi-value ${result.comparison.relevance_change_pct < -0.05 ? "down" : ""}`}>
                    {pct(result.comparison.relevance_change_pct, 1)}
                  </div>
                  <div className="kpi-sub">{result.comparison.overlap} of {result.items.length} items unchanged</div>
                </Card>
              </div>
              <Callout kind="info">
                <b>How to read this.</b> Item relevance differs only slightly between profiles, because the dataset has no learnable individual taste
                (see the Evaluation page). The strongest lever is the profit weight: around 0.25 lifts profit with almost no hit-rate loss,
                and above about 0.5 the list collapses into Electronics.
              </Callout>
              {result.items.map((it: Json) => (
                <div className="rec" key={it.item}>
                  <div className="rank">{it.rank}</div>
                  <div>
                    <div className="rec-name">{it.item}</div>
                    <div className="rec-meta">
                      <span className="badge">{it.category}</span> {money(it.avg_unit_price)} avg price · {it.margin_pct.toFixed(1)}% margin · {it.avg_rating.toFixed(1)}★ · {pct(it.return_rate)} returns
                    </div>
                    <div className="bar" title="relative relevance"><div style={{ width: `${(it.relevance / maxScore) * 100}%` }} /></div>
                    <ul>{it.reasons.map((r: string, i: number) => <li key={i}>{r}</li>)}</ul>
                    {it.upsell && (
                      <div className="pill-row">
                        <span className="badge good">Upsell: tier {it.upsell.tier} at {money(it.upsell.avg_unit_price)} adds {money(it.upsell.extra_profit)} profit/order</span>
                      </div>
                    )}
                  </div>
                  <div className="rec-right">
                    <div className="big">{money(it.net_profit_per_order)}</div>
                    <div className="note" style={{ margin: 0 }}>net profit / order</div>
                    <div className="note" style={{ margin: "4px 0 0" }}>relevance {pct(it.relevance, 2)}</div>
                    <div className="note" style={{ margin: 0 }}>score {it.final_score.toFixed(3)}</div>
                  </div>
                </div>
              ))}
              <Card title="Relevance-only baseline (λ = 0)" note="Top items if only predicted relevance mattered.">
                <div className="pill-row" style={{ marginTop: 0 }}>
                  {result.baseline.map((b: Json) => <span key={b.item} className="badge">{b.item} · {b.category} · {money(b.net_profit_per_order)}</span>)}
                </div>
              </Card>
            </>
          )}
        </div>
      </div>
    </>
  );
}
