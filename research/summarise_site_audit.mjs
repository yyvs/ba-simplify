/**
 * The site-audit artifact, as the tables RESULTS.md quotes.
 *
 *   cd research && node summarise_site_audit.mjs results/site_audit/site_audit_<stamp>.json
 *
 * Prints every figure the write-up quotes, from the artifact, in the shape it is quoted
 * in, so none is hand-copied (see RESULTS.md §4).
 */
import fs from "node:fs";

const file =
  process.argv[2] ??
  (() => {
    const dir = "results/site_audit";
    const candidates = fs.readdirSync(dir).filter((f) => f.endsWith(".json")).sort();
    if (!candidates.length) throw new Error(`no artifact in ${dir}`);
    return `${dir}/${candidates[candidates.length - 1]}`;
  })();
const report = JSON.parse(fs.readFileSync(file, "utf8"));
const rows = report.rows.filter((r) => !r.error);
const failed = report.rows.filter((r) => r.error);

const pct = (n, d) => (d ? `${Math.round((n / d) * 100)}%` : "—");
const sum = (xs) => xs.reduce((a, b) => a + b, 0);

console.log(`# ${file}`);
console.log(`# ${report.run.started_at} · commit ${report.run.git_commit} · ${report.backend.seq2seq_device}`);
console.log(`# models: ${JSON.stringify(report.backend.model_names)}`);
if (failed.length) {
  console.log(`# ${failed.length} row(s) errored: ${failed.map((r) => `${r.site?.name}/${r.cut}`).join(", ")}`);
}
console.log();

console.log("## Coverage");
console.log();
console.log("| Category | Page | Cut | Scope | Found | Skipped | Sent | Changed | Changed/sent | Reqs with a guarded sentence |");
console.log("|---|---|---|---|---|---|---|---|---|---|");
for (const r of rows) {
  console.log(
    `| ${r.site.category} | ${r.site.name} | ${r.cut} | \`${r.content_scope ?? "—"}\` | ${r.units.found} | ` +
      `${r.units.skipped} (${pct(r.units.skipped, r.units.found)}) | ${r.units.sent} | ${r.units.changed} | ` +
      `${pct(r.units.changed, r.units.sent)} | ${r.requests.requests_with_a_guarded_sentence} |`
  );
}
console.log();

console.log("## Preservation");
console.log();
console.log("| Page | Cut | Anchors | Labels emptied | ...misplaced | ...model-deleted | Labels rewritten | Tags lost | Images | Revert (text) | Revert (attrs) |");
console.log("|---|---|---|---|---|---|---|---|---|---|---|");
for (const r of rows) {
  const loss = r.anchor_loss ?? {};
  const attrs = r.revert.attributes ?? {};
  const attrCell =
    attrs.comparable === false
      ? "n/a"
      : (attrs.elements_with_gained_attributes ?? 0) + (attrs.elements_with_lost_attributes ?? 0) === 0
        ? "clean"
        : `${attrs.elements_with_gained_attributes ?? 0}+/${attrs.elements_with_lost_attributes ?? 0}-`;
  console.log(
    `| ${r.site.name} | ${r.cut} | ${r.dom.anchors_before} | ${r.dom.anchor_labels_emptied} | ` +
      `${loss.misplaced_by_write_back ?? "—"} | ${loss.deleted_by_model ?? "—"} | ` +
      `${r.dom.anchor_labels_rewritten} | ${Object.keys(r.dom.tags_lost).length ? JSON.stringify(r.dom.tags_lost) : "none"} | ` +
      `${r.dom.images_before === r.dom.images_after ? "intact" : "CHANGED"} | ` +
      `${r.revert.text_identical ? "clean" : "DIRTY"} | ${attrCell} |`
  );
}
console.log();

console.log("## Why content was skipped (items)");
console.log();
const itemReasons = [...new Set(rows.flatMap((r) => Object.keys(r.skips.items)))].sort();
console.log(`| Page | Cut | ${itemReasons.join(" | ")} |`);
console.log(`|---|---|${itemReasons.map(() => "---").join("|")}|`);
for (const r of rows) {
  console.log(`| ${r.site.name} | ${r.cut} | ${itemReasons.map((k) => r.skips.items[k] ?? 0).join(" | ")} |`);
}
console.log();

console.log("## Why content was skipped (chunks within walked elements)");
console.log();
const chunkReasons = [...new Set(rows.flatMap((r) => Object.keys(r.skips.chunks)))].sort();
console.log(`| Page | Cut | ${chunkReasons.join(" | ")} |`);
console.log(`|---|---|${chunkReasons.map(() => "---").join("|")}|`);
for (const r of rows) {
  console.log(`| ${r.site.name} | ${r.cut} | ${chunkReasons.map((k) => r.skips.chunks[k] ?? 0).join(" | ")} |`);
}
console.log();

console.log("## Prose in the content area that was skipped (candidate false negatives)");
console.log();
console.log("| Page | Cut | Prose skipped (>=12 words, in scope, not boilerplate) | By reason | Boilerplate sent |");
console.log("|---|---|---|---|---|");
for (const r of rows) {
  console.log(
    `| ${r.site.name} | ${r.cut} | ${r.selection.prose_skipped_in_scope} | ` +
      `${Object.keys(r.selection.prose_skipped_by_reason).length ? JSON.stringify(r.selection.prose_skipped_by_reason) : "—"} | ` +
      `${r.selection.boilerplate_sent} |`
  );
}
console.log();

console.log("## Totals, by cut");
console.log();
for (const cut of ["sentence", "document"]) {
  const rs = rows.filter((r) => r.cut === cut);
  if (!rs.length) continue;
  const found = sum(rs.map((r) => r.units.found));
  const skipped = sum(rs.map((r) => r.units.skipped));
  const sent = sum(rs.map((r) => r.units.sent));
  const changed = sum(rs.map((r) => r.units.changed));
  const emptied = sum(rs.map((r) => r.dom.anchor_labels_emptied ?? 0));
  const misplaced = sum(rs.map((r) => r.anchor_loss?.misplaced_by_write_back ?? 0));
  const deleted = sum(rs.map((r) => r.anchor_loss?.deleted_by_model ?? 0));
  const anchors = sum(rs.map((r) => r.dom.anchors_before));
  const prose = sum(rs.map((r) => r.selection.prose_skipped_in_scope));
  const boiler = sum(rs.map((r) => r.selection.boilerplate_sent));
  const guards = sum(rs.map((r) => r.requests.requests_with_a_guarded_sentence));
  const cached = sum(rs.map((r) => r.requests.served_from_cache ?? 0));
  console.log(`**${cut}** (${rs.length} pages)`);
  console.log(`- found ${found}, skipped ${skipped} (${pct(skipped, found)}), sent ${sent} (${pct(sent, found)}), changed ${changed} (${pct(changed, sent)} of sent)`);
  console.log(`- anchors ${anchors}; labels emptied ${emptied} (${pct(emptied, anchors)}) = ${misplaced} misplaced by write-back + ${deleted} deleted by the model`);
  console.log(`- prose skipped inside the content area: ${prose}; boilerplate sent: ${boiler}`);
  console.log(`- requests where at least one sentence hit a backend guard: ${guards} of ${sent} sent (${pct(guards, sent)}); sentences sent: ${sum(rs.map((r) => r.requests.sentences_total ?? 0))}; requests served from cache: ${cached}`);
  console.log(`- pages where the revert was not text-identical: ${rs.filter((r) => !r.revert.text_identical).map((r) => r.site.name).join(", ") || "none"}`);
  console.log();
}
