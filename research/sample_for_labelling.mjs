/**
 * Draws the hand-labelling sample the RQ2 study's second half needs.
 *
 *   cd research && node sample_for_labelling.mjs results/site_audit/site_audit_<stamp>
 *
 * For §6.4's TODO: "a hand-labelled verdict on a sample of ~50 skipped and ~50 unchanged
 * items". The audit rates say how often the filters act, not whether each call was right.
 *
 * - Stratified by page: a flat sample would be mostly Wikipedia and MDN, which skip
 *   thousands.
 * - Deterministic: ranked by a hash of each item's text (no clock/RNG), so re-runs give
 *   the same sample and recorded labels stay attached to their items.
 *
 * Writes a JSONL with an empty `verdict` field per row, to be filled in by hand.
 */
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const stem = process.argv[2];
if (!stem)
  throw new Error(
    "usage: node sample_for_labelling.mjs results/site_audit/site_audit_<stamp>",
  );
const base = stem
  .replace(/\.json$/, "")
  .replace(/_items\.jsonl$/, "")
  .replace(/_units\.jsonl$/, "");
const readJsonl = (f) =>
  fs
    .readFileSync(f, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l));

const items = readJsonl(`${base}_items.jsonl`);
const units = readJsonl(`${base}_units.jsonl`);

const rank = (s) => createHash("sha256").update(s).digest("hex");

// `perPage` items per (site, cut), lowest hash first; smaller pages are not padded.
function stratify(rows, perPage, keyOf) {
  const groups = new Map();
  for (const r of rows) {
    const k = `${r.site}/${r.cut}`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(r);
  }
  const out = [];
  for (const [group, rs] of [...groups.entries()].sort()) {
    const ordered = [...rs].sort((a, b) =>
      rank(keyOf(a)).localeCompare(rank(keyOf(b))),
    );
    for (const r of ordered.slice(0, perPage)) out.push({ group, ...r });
  }
  return out;
}

// Skipped side: only prose-length text inside the content area; skipped nav labels would
// be trivially correct.
const skippedCandidates = items.filter(
  (i) =>
    i.status === "skipped" && i.in_scope && !i.in_boilerplate && i.words >= 8,
);
// section headings are skipped by design in document mode, not by a filter
const skipped = stratify(
  skippedCandidates.filter((i) => i.reason !== "section-heading"),
  3,
  (i) => i.text,
);

// Unchanged side: returned as-is by the model, either already simple or a missed
// simplification.
const unchanged = stratify(
  units.filter((u) => !u.changed),
  3,
  (u) => u.input,
);

const outFile = `${base}_labelling.jsonl`;
const rows = [
  ...skipped.map((i) => ({
    kind: "skipped",
    group: i.group,
    site: i.site,
    category: i.category,
    cut: i.cut,
    tag: i.tag,
    words: i.words,
    reason: i.reason,
    text: i.text,
    // to be filled in by hand: "correct" | "should-have-been-sent" | "unclear"
    verdict: "",
    note: "",
  })),
  ...unchanged.map((u) => ({
    kind: "unchanged",
    group: u.group,
    site: u.site,
    category: u.category,
    cut: u.cut,
    words: u.words,
    guard: u.guard,
    text: u.input,
    // "already-simple" | "should-have-been-simplified" | "unclear"
    verdict: "",
    note: "",
  })),
];
fs.writeFileSync(outFile, rows.map((r) => JSON.stringify(r)).join("\n") + "\n");

const byReason = {};
for (const s of skipped)
  byReason[s.reason ?? "null"] = (byReason[s.reason ?? "null"] || 0) + 1;
console.log(
  `skipped pool: ${skippedCandidates.length} -> sampled ${skipped.length}`,
);
console.log(`  by reason: ${JSON.stringify(byReason)}`);
console.log(
  `unchanged pool: ${units.filter((u) => !u.changed).length} -> sampled ${unchanged.length}`,
);
console.log(
  `wrote ${path.relative(".", outFile)} (${rows.length} rows, verdict fields empty)`,
);
