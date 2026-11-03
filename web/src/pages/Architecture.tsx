import { Callout, Card, Table } from "../ui";

function Flow({ steps }: { steps: [string, string?][] }) {
  return (
    <div className="flow">
      {steps.map(([t, s], i) => (
        <span key={t} style={{ display: "contents" }}>
          <div className="box">{t}{s && <small>{s}</small>}</div>
          {i < steps.length - 1 && <span className="arrow">→</span>}
        </span>
      ))}
    </div>
  );
}

const ROWS: string[][] = [
  ["Data ingestion", "CSV to Parquet (391 MB to 81 MB)", "CDC and batch loaders (Data Factory or Airflow). Kafka or Event Hubs for clickstream."],
  ["Data storage", "Parquet files", "Lakehouse bronze / silver / gold on object storage with Delta tables, and a warehouse for BI."],
  ["Data quality", "Quality flags: duplicates, reused ids, status conflicts", "Rules as pipeline gates (Great Expectations or Delta constraints), with quarantine and alerts."],
  ["Feature engineering", "16 profile features plus a 48-row item table", "Spark jobs writing to a feature store (offline and online) to avoid training/serving skew."],
  ["Model training", "7 models trained with scikit-learn, exported as numpy-only inference classes (74 KB)", "Scheduled training, MLflow tracking and registry, and a promotion gate on hit rate and profit."],
  ["Serving", "FastAPI loads artifacts into memory and re-ranks per request", "Candidate generation (ANN) then ranking then business re-ranking behind a gateway, with Redis cache."],
  ["Monitoring", "Offline metrics endpoint", "Data and prediction drift, latency SLOs, CTR, conversion and profit per recommendation, with A/B tests."],
  ["Governance", "Documented", "PII minimization and pseudonymization, lineage, role-based access, audit logs, GDPR delete and opt-out."],
];

const BONUS: string[][] = [
  ["Profit-aware ranking", "Built", "λ re-ranker on net profit per order, exposed as a slider and an API parameter."],
  ["Fraud-aware filtering", "Built", "Withholds recommendations above a risk threshold, plus an optional item-level penalty. The score is not a validated risk signal in this data."],
  ["Sentiment-augmented", "Built", "Item positive-sentiment share as a re-ranking term. Sentiment is a deterministic function of rating here, so it adds no new information."],
  ["GenAI explanations", "Templates", "Rule-based reason codes. An LLM rewrite is a drop-in extension, off by default to avoid sending customer data to a third party."],
  ["Real-time architecture", "Design", "Event stream updates online features, and the serving flow already ranks per request. A session-based model would cover anonymous visitors."],
];

export default function Architecture() {
  return (
    <>
      <h1>Solution Architecture</h1>
      <p className="sub">Train offline, serve small. Relevance and business value are separate layers, so the business weights change without retraining.</p>

      <Card title="As built (this repository)">
        <div className="stack">
          <Flow steps={[["Raw CSV", "1M rows"], ["ingest", "Parquet"], ["clean", "types, keys, flags"], ["features", "item + profile"]]} />
          <Flow steps={[["features"], ["train", "7 models"], ["evaluate", "temporal split"], ["artifacts/", "models, tables, metrics"]]} />
          <Flow steps={[["artifacts/"], ["FastAPI", "in-memory"], ["React app", "static site"]]} />
        </div>
      </Card>

      <Card title="Production design">
        <div className="stack">
          <Flow steps={[["Sources", "orders, events, CRM"], ["Ingestion", "CDC, batch, Kafka"], ["Lakehouse", "bronze, silver, gold"], ["Feature store", "offline + online"]]} />
          <Flow steps={[["Feature store"], ["Training", "scheduled"], ["Model registry", "MLflow"], ["Evaluation gate", "hit rate and profit"]]} />
          <Flow steps={[["Gateway"], ["Recommendation service", "candidates, ranking, re-ranking"], ["Redis cache"], ["Monitoring and A/B", "drift, CTR, profit"]]} />
        </div>
      </Card>

      <Card title="Request flow">
        <Flow steps={[["React app", "profile, λ, filters"], ["POST /recommendations"], ["Model", "P(item | profile)"], ["Re-ranker", "relevance + profit − risk"], ["Top-K", "with reason codes"]]} />
        <p className="note">final = (1 − λ)·relevance + λ·net profit per order − return penalty − fraud penalty + sentiment boost</p>
      </Card>

      <Card title="Component by component">
        <Table head={["Area", "Built here", "Production"]} rows={ROWS} />
      </Card>

      <Callout kind="info">
        Every customer is a cold start in this dataset, so the built system is profile-based. The architecture keeps a slot for collaborative filtering
        (an ALS or two-tower candidate generator) that feeds the same re-ranker once repeat-purchase data exists.
      </Callout>

      <Card title="Bonus challenges">
        <Table head={["Bonus", "Status", "Approach"]} rows={BONUS.map((r) => [<b key="b">{r[0]}</b>, <span key="s" className={`badge ${r[1] === "Built" ? "good" : "warn"}`}>{r[1]}</span>, r[2]])} />
      </Card>

      <Card title="Zero-cost deployment">
        <ul className="clean">
          <li><b>Frontend:</b> static Vite build on Vercel, Netlify, Cloudflare Pages or GitHub Pages.</li>
          <li><b>Backend:</b> FastAPI on Render's free tier as a native Python web service (no Docker). The API needs only 4 packages and about 85 MB of memory.</li>
          <li><b>Database:</b> none. Precomputed artifacts (a few MB) ship with the API and load into memory at startup.</li>
          <li><b>Training:</b> runs locally. The raw 1M-row file never goes to production.</li>
        </ul>
      </Card>
    </>
  );
}
