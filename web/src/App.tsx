import { Suspense, lazy, useEffect, useState } from "react";

const Overview = lazy(() => import("./pages/Overview"));
const Playground = lazy(() => import("./pages/Playground"));
const Customers = lazy(() => import("./pages/Customers"));
const Insights = lazy(() => import("./pages/Insights"));
const Evaluation = lazy(() => import("./pages/Evaluation"));
const Architecture = lazy(() => import("./pages/Architecture"));
const DataHealth = lazy(() => import("./pages/DataHealth"));

const PAGES = [
  { id: "overview", label: "Overview", el: <Overview /> },
  { id: "data", label: "Data Health", el: <DataHealth /> },
  { id: "playground", label: "Recommendations", el: <Playground /> },
  { id: "customers", label: "Customers", el: <Customers /> },
  { id: "insights", label: "Insights", el: <Insights /> },
  { id: "evaluation", label: "Evaluation", el: <Evaluation /> },
  { id: "architecture", label: "Architecture", el: <Architecture /> },
] as const;

export default function App() {
  const [page, setPage] = useState<string>(location.hash.slice(1) || "overview");
  useEffect(() => {
    const onHash = () => setPage(location.hash.slice(1) || "overview");
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const go = (id: string) => { location.hash = id; };
  const current = PAGES.find((p) => p.id === page) ?? PAGES[0];
  return (
    <div className="app">
      <header>
        <div className="brand">
          <span className="logo">R</span>
          <div>
            <div className="brand-title">Product Recommendation Platform</div>
            <div className="brand-sub">1M orders · 48 products · profit-aware ranking</div>
          </div>
        </div>
        <nav>
          {PAGES.map((p) => (
            <button key={p.id} className={p.id === current.id ? "active" : ""} onClick={() => go(p.id)}>{p.label}</button>
          ))}
        </nav>
      </header>
      <main><Suspense fallback={<div className="loading">Loading...</div>}>{current.el}</Suspense></main>
    </div>
  );
}
