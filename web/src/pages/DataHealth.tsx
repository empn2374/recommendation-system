import { Callout, Card, Table } from "../ui";

const FINDINGS: [string, string, string, string][] = [
  ["F1", "Practically no repeat customers", "1,000,122 distinct customers for 1,000,123 orders", "Collaborative filtering has no history to learn from"],
  ["F2", "product_id is not a product key", "26% of ids map to several names and prices; only 48 real product names", "The catalogue is 48 items (by name)"],
  ["F3", "customer_id collisions", "About 8,100 ids reused by different people", "Customer key = id + name"],
  ["F4", "Product choice ignores customer attributes", "Cramér's V of 0.006 to 0.008 against segment, country, age and more", "No personalization signal to learn"],
  ["F5", "All 48 products look identical", "Margin 39.5%, return rate 10%, rating 3.0 for every product", "Real differences exist only between categories"],
  ["F6", "Segment, campaign, device and country do not move outcomes", "AOV, margin and return rate are flat across them", "Segmentation and marketing insights are mostly flat"],
  ["F7", "Discount is the only mechanical driver", "Margin 45.0% at 0% off down to 26.7% at 25% off", "A valid insight on discount policy"],
  ["F8", "Reference columns are noise", "total_orders_by_customer says 1 to 50, product_rating_avg has about 0 correlation with row ratings", "Not used as ground truth"],
  ["F9", "Status fields contradict each other", "23.2% of rows, e.g. Completed but delivery Failed", "Flagged. Revenue uses carefully defined statuses"],
  ["F10", "Duplicated order_id", "8,140 ids across 16,333 rows", "Flagged, rows kept"],
];

export default function DataHealth() {
  return (
    <>
      <h1>Data Health</h1>
      <p className="sub">Profiling and quality assessment of the 1,000,123-row, 62-column dataset (full write-up in the Phase 1 document).</p>
      <div className="grid g3" style={{ marginBottom: 16 }}>
        <Card title="Completeness"><p>Only three columns have nulls, and each null has a meaning: return reason (90%, not returned), coupon code (50%, no coupon), feedback (20%).</p></Card>
        <Card title="Arithmetic integrity"><p>Total, profit and margin reconcile on every row. There are no negative or zero prices, quantities or profits, and ratings are all within 1 to 5.</p></Card>
        <Card title="Cleaning approach"><p>All rows were kept. Problems are exposed as flags (duplicate id, reused customer id, status conflict) instead of dropping 24.5% of the data.</p></Card>
      </div>
      <Callout kind="warn">The data is synthetic and mostly random. The findings below explain why the recommender focuses on business value, not individual taste.</Callout>
      <Card title="Key findings">
        <Table head={["#", "Finding", "Evidence", "Impact"]} rows={FINDINGS.map((f) => [<b key="i">{f[0]}</b>, f[1], f[2], f[3]])} />
      </Card>
      <Card title="Definitions used throughout">
        <ul className="clean">
          <li><b>Item</b> = product name (48 items in 5 categories).</li>
          <li><b>Customer key</b> = customer id plus customer name.</li>
          <li><b>Net profit</b> = profit on Completed orders only. Returned, cancelled and pending orders count as zero.</li>
          <li><b>Realized sale</b> = Completed or Returned, and Paid (80% of rows).</li>
          <li><b>Temporal split</b> = train before 2025-11-01, test on the last three months.</li>
        </ul>
      </Card>
    </>
  );
}
