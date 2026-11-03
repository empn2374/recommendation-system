import { ReactNode } from "react";

export function Card({ title, children, note }: { title?: string; children: ReactNode; note?: string }) {
  return (
    <section className="card">
      {title && <h3>{title}</h3>}
      {children}
      {note && <p className="note">{note}</p>}
    </section>
  );
}

export function Kpi({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="kpi">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

export function State({ error, loading }: { error: string | null; loading: boolean }) {
  if (error) return <div className="error">Could not reach the API: {error}</div>;
  if (loading) return <div className="loading">Loading...</div>;
  return null;
}

export function Callout({ kind = "info", children }: { kind?: "info" | "warn" | "good"; children: ReactNode }) {
  return <div className={`callout ${kind}`}>{children}</div>;
}

export function Table({ head, rows }: { head: string[]; rows: ReactNode[][] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead><tr>{head.map((h) => <th key={h}>{h}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}
