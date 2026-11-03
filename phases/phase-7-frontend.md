# Phase 7: React Frontend

**Status:** Done, type-checked, built, and tested in a real browser.
**Code:** [web/](../web) (Vite, React 19, TypeScript, Recharts, plain CSS)
**Run locally:** start the API on port 8000, then `cd web && npm install && npm run dev` and open `http://localhost:5173`.
**Configuration:** `VITE_API_URL` points the app at the API (default `http://127.0.0.1:8000`). See `web/.env.example`.

## 1. Pages

| Page | What it shows |
|---|---|
| **Overview** | KPIs, the honest headline with the paired-test result, the three findings that matter, all key findings, and what was built |
| **Data Health** | Completeness and integrity summary, the ten Phase 1 findings with their impact, and the definitions used throughout |
| **Recommendations** (hero page) | Pick one of 300 real customers or edit a profile. Choose a model and tune the profit weight, return, fraud and sentiment terms, category filter, K, and a high-risk withholding switch. The list re-ranks live with reason codes, upsell hints, profit uplift against the relevance-only baseline, relevance cost, and the baseline list. |
| **Customers** | Value tiers, Pareto chart, clusters and their weak silhouette, CLV scenarios, whether the supplied VIP/Premium/Regular label means anything, and behavior tests |
| **Insights** | Category performance, within-category spread, discount depth vs margin and conversion, campaigns, monthly orders, returns by reason and payment method, and five sized revenue opportunities |
| **Evaluation** | Model comparison with confidence intervals, the personalization lift, and the relevance-vs-profit trade-off chart and table |
| **Architecture** | As-built and production flows, the request flow and formula, a component table, bonus-challenge status, and the zero-cost deployment |

Design choices: every claim in the UI comes from an API field (no hardcoded results). Caveats are shown next to the numbers they qualify, such as weak silhouette, assumption-based CLV, and effect size against p-values.

## 2. Verification

- `tsc --noEmit` and `vite build` pass with no warnings.
- **Real-browser checks (Edge via Playwright):** all seven pages render. In the Recommendations page, selecting real customers updates the profile and list. Setting λ=1 produces an all-Electronics list (+102% profit, −32% relevance). Raising the fraud score above the threshold with the withholding switch on shows the explanation and zero cards.
- The UI numbers match the offline evaluation: at λ=0.25 the default profile shows +24% profit for −2.3% relevance, consistent with the +22% measured on held-out data.
- A fresh page load reports no console errors.

## 3. Optimization

The first build was a single 647 KB JavaScript file (192 KB gzipped). Each page is now lazy-loaded, so the 338 KB charting chunk only downloads when a chart page opens.

| Measure | Before | After |
|---|---|---|
| Initial JavaScript | 647 KB (192 KB gzipped) | **224 KB (70 KB gzipped)**, −63% gzipped |
| Large-chunk build warning | yes | gone |
| Chart library | in the initial load | loaded only on Customers, Insights and Evaluation |

## 4. Problems encountered and fixes

| Problem | Fix |
|---|---|
| Customers page cluster chart drew dots but no lines, and a line chart across unordered clusters is the wrong chart type | Replaced it with grouped bars. |
| Monthly orders line missing in screenshots (animation still running) | Disabled line animation. |
| A key finding exposed an internal field name (`within_category_item_spread`) | Reworded the finding in the pipeline and regenerated the insights. |
| Two figures (24.8% and 1.0%) were hardcoded in the Customers page, so they could drift from the data | Exported `high_value_by_category` from the pipeline and read it from the API. |
| Silhouette note said "about 0.06" when the range is 0.06 to 0.07 | Computed the range from the data. |
| The `random` baseline shows more profit captured than popularity, which looked like an error | Added a note: the random model scores ties, so it returns one fixed arbitrary list and its profit figure is luck. |
| Checkboxes showed the wrong state for sample customers (string "false" treated as true) and React logged a console error | Fixed at the data source and defensively in the UI (see Phase 6). |
| Browser back and forward did not change pages | Added a `hashchange` listener. |
| Missing favicon caused a console 404 | Added an inline favicon. |

## 5. Next

Phase 8: deployment (no Docker) and the optimization summary.
