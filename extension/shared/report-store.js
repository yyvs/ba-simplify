// shared/report-store.js - saved reports: shape, storage and shared operations.
// Loaded by history.html (creates them) and reports.html (shows, evaluates, deletes).
//
// A report is a kept run. The History log is a rolling window (last 20 pages, see
// background.js's MAX_HISTORY_PAGES); a report holds its own copy of the rows, so
// deleting or pushing the page out of History doesn't affect it.
//
// Shape is a superset of a stored history page (same method/modelKey/modelId/
// granularity/audience/status/stats/totalTimeMs/entries, plus report fields), so
// shared/analysis.js's readers and summary lines run on it unchanged.
const REPORTS_STORAGE_KEY = "simplifyReports";
// Storage-quota backstop, not a rolling window: ~50 runs of 200 rows is already a large
// share of chrome.storage.local. Oldest dropped only at the cap.
const MAX_REPORTS = 50;

// Explicit list so a report doesn't inherit fields later builds add to history pages.
// `entries` is cloned, not aliased.
const RUN_SNAPSHOT_KEYS = [
  "url",
  "title",
  "timestamp",
  "method",
  "modelKey",
  "modelId",
  "granularity",
  "audience",
  "status",
  "stats",
  "totalTimeMs",
];

function runSnapshot(page) {
  const snapshot = {};
  RUN_SNAPSHOT_KEYS.forEach((key) => {
    if (page[key] !== undefined) snapshot[key] = page[key];
  });
  snapshot.entries = (page.entries || []).map((entry) => ({ ...entry }));
  return snapshot;
}

// "Wikipedia — Recycling · Fine-tuned model · whole sections": page, model, scope,
// derived from the run instead of prompting. Timestamp is a separate field.
function reportName(page) {
  const parts = [(page.title || page.url || "Untitled page").trim()];
  if (page.modelKey) {
    parts.push(modelDisplayName(page.modelKey));
    parts.push(modelScopeText(page.modelKey, page.granularity ? { [page.modelKey]: page.granularity } : null));
  } else if (page.granularity) {
    parts.push(granularityLabel(page.granularity).toLowerCase());
  }
  return parts.join(" · ");
}

function newReportId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return `report-${crypto.randomUUID()}`;
  return `report-${Date.now()}-${Math.round(Math.random() * 1e9)}`;
}

function reportFromPage(page) {
  return {
    reportId: newReportId(),
    name: reportName(page),
    createdAt: Date.now(),
    // source run in the History log, followed while still running (see refreshedReport)
    sourceSessionId: page.sessionId || null,
    ...runSnapshot(page),
    // computed on demand, then stored
    evaluation: null,
  };
}

async function readReports() {
  const result = await chrome.storage.local.get(REPORTS_STORAGE_KEY);
  const reports = result[REPORTS_STORAGE_KEY];
  return Array.isArray(reports) ? reports : [];
}

async function writeReports(reports) {
  await chrome.storage.local.set({ [REPORTS_STORAGE_KEY]: reports.slice(0, MAX_REPORTS) });
}

// Idempotent per run: saving the same page again updates its report. `already` lets
// the caller say which happened.
async function saveReportForPage(page) {
  const reports = await readReports();
  const existing = page.sessionId
    ? reports.findIndex((r) => r.sourceSessionId === page.sessionId)
    : -1;
  if (existing >= 0) {
    const merged = { ...reports[existing], ...runSnapshot(page) };
    const next = reports.slice();
    next[existing] = merged;
    await writeReports(next);
    return { report: merged, already: true };
  }
  const report = reportFromPage(page);
  await writeReports([report, ...reports]);
  return { report, already: false };
}

// --- comparison reports ---------------------------------------------------
// Each arm-run is also logged to History as its own page (one per model and cut), and
// cached units are attributed to it. The report records what the log can't: that the
// runs got the same text, in the same order, from one reading of one page.
//
// One report per comparison, entries stored once, results keyed by cut then arm:
//
//   arms:    { A: { modelKey, modelId, method, granularity, cuts: { <cut>: {...} } }, B: {…} }
//   entries: [ { input, results: { whole_sections:       { A: {...}, B: {...} },
//                                  sentence_by_sentence: { A: {...}, B: {...} } } } ]
//
// An arm is a model, not a (model, cut) pair: with both cuts one arm answers twice under
// one letter. Per-cut timings and counts live under `arms[label].cuts[cut]`.
//
// Written as each arm-run finishes (runs take minutes), as an upsert keyed by the
// group id so columns accumulate into one report. Labels come from shared/model-labels.js
// (comparisonArmLabel) because the run is in a content script, where this file isn't
// loaded.

// Same cap as background.js's MAX_ENTRIES_PER_PAGE, but keeps the first entries, not the
// last: rows are aligned across arms and cuts by position, and dropping the head would
// offset the columns.
const MAX_COMPARISON_UNITS = 200;

function comparisonReportName(page, armCount) {
  const parts = [(page.title || page.url || "Untitled page").trim()];
  parts.push(armCount === 1 ? "1 model compared" : `${armCount} models compared`);
  // all cuts, e.g. "sections and sentences"
  const cuts = comparisonCutOrder(page.cutOrder || [page.unitGranularity]);
  if (cuts.length > 0) parts.push(cuts.map((cut) => cutShortLabel(cut)).join(" and "));
  return parts.join(" · ");
}

// Checks for arms rather than `kind`, so older reports are read by their content.
function isComparisonReport(report) {
  return !!(report && report.arms && typeof report.arms === "object");
}

function comparisonArms(report) {
  if (!isComparisonReport(report)) return [];
  return Object.keys(report.arms)
    .map((label) => ({ label, ...report.arms[label] }))
    .sort((a, b) => (a.index || 0) - (b.index || 0));
}

function comparisonCuts(report) {
  if (!isComparisonReport(report)) return [];
  return comparisonCutOrder(report.cutOrder || [report.unitGranularity]);
}

// One column per (cut, arm) that ran, in run order. Used by both the evaluation and the
// report block.
function comparisonColumns(report) {
  const arms = comparisonArms(report);
  const columns = [];
  comparisonCuts(report).forEach((cut) => {
    arms.forEach((arm) => {
      if (!arm.cuts || !arm.cuts[cut]) return;
      columns.push({ cut, arm, run: arm.cuts[cut] });
    });
  });
  return columns;
}

// Folds one arm-run's results into its comparison report. `payload.results` is aligned
// to `payload.inputs` by position, with null where an entry got no complete answer.
async function saveComparisonRun(payload) {
  const reports = await readReports();
  const existingIndex = reports.findIndex(
    (r) => isComparisonReport(r) && r.comparisonGroup === payload.group
  );
  const inputs = (payload.inputs || []).slice(0, MAX_COMPARISON_UNITS);
  const base =
    existingIndex >= 0
      ? { ...reports[existingIndex], arms: { ...reports[existingIndex].arms } }
      : {
          reportId: newReportId(),
          createdAt: Date.now(),
          kind: "comparison",
          comparisonGroup: payload.group,
          // written by the run itself, not refreshed from the log (see refreshedReport)
          sourceSessionId: null,
          url: payload.page.url,
          title: payload.page.title,
          timestamp: payload.page.timestamp,
          // coarsest cut given; a section contains the sentence cut's units, so its
          // entry can hold both cuts' answers
          unitGranularity: payload.unitGranularity,
          cutOrder: payload.cutOrder || [payload.unitGranularity],
          arms: {},
          // planned arm-runs, so a partial report doesn't look finished
          models: payload.models,
          units: (payload.inputs || []).length,
          entries: inputs.map((input) => ({ input, results: {} })),
          evaluation: null,
        };

  const { label } = payload.arm;
  const cut = payload.cut;
  base.entries = base.entries.map((entry, i) => {
    const result = (payload.results || [])[i];
    if (!result) return entry;
    return {
      ...entry,
      results: { ...entry.results, [cut]: { ...(entry.results[cut] || {}), [label]: result } },
    };
  });
  // arm identity is per model; each cut adds its run record
  const existingArm = base.arms[label] || {};
  base.arms[label] = {
    ...existingArm,
    ...payload.arm,
    cuts: { ...(existingArm.cuts || {}), [cut]: payload.run },
  };
  base.status = payload.runStatus;
  base.name = comparisonReportName(base, Object.keys(base.arms).length);
  // a new column makes any stored evaluation stale
  if (existingIndex >= 0) base.evaluation = null;

  const rest = existingIndex >= 0 ? reports.filter((_, i) => i !== existingIndex) : reports;
  await writeReports([base, ...rest]);
  return base;
}

// Progress in columns and entries; the run finishes one column before the next.
function comparisonProgress(report) {
  const columns = comparisonColumns(report);
  const units = report.units || (report.entries || []).length;
  const answered = columns.reduce(
    (sum, column) => sum + ((column.run.stats && column.run.stats.processedItems) || 0),
    0
  );
  const total = (report.models || columns.length) * units;
  return {
    columns: columns.length,
    runs: report.models || columns.length,
    arms: comparisonArms(report).length,
    cuts: comparisonCuts(report).length,
    units,
    done: Math.min(answered, total),
    total,
    state: report.status === "running" ? "running" : report.status === "stopped" ? "stopped" : "complete",
  };
}

async function deleteReport(reportId) {
  const reports = await readReports();
  await writeReports(reports.filter((r) => r.reportId !== reportId));
}

async function deleteAllReports() {
  await chrome.storage.local.remove(REPORTS_STORAGE_KEY);
}

async function storeEvaluation(reportId, evaluation) {
  const reports = await readReports();
  await writeReports(reports.map((r) => (r.reportId === reportId ? { ...r, evaluation } : r)));
}

// --- following a run that hasn't finished --------------------------------
// Runs log in batches (content.js's flushHistoryBuffer), so a report saved mid-run is
// refreshed from its source page while that page is still in the log. Once it leaves
// the log, the last snapshot is kept.
function refreshedReport(report, pages) {
  if (!report.sourceSessionId) return report;
  const live = pages.find((p) => p.sessionId === report.sourceSessionId);
  if (!live) return report;
  const merged = { ...report, ...runSnapshot(live) };
  // return the same object when unchanged; runs on every render and a new object
  // would trigger a storage write
  return JSON.stringify(merged) === JSON.stringify(report) ? report : merged;
}

// Progress from recorded counts, not rows on hand (capped at MAX_ENTRIES_PER_PAGE).
// `total` is null for older builds that didn't record it. `live` (source still in the
// log) distinguishes "still running" from "stopped early".
function reportProgress(report, live) {
  const stats = pageStats(report);
  const total = typeof stats.totalItems === "number" ? stats.totalItems : null;
  const done = (stats.processedItems || 0) + (stats.skippedItems || 0);
  if (total == null) return { done, total: null, state: live ? "running" : "unknown" };
  const settled = done >= total;
  return {
    done: Math.min(done, total),
    total,
    state: settled ? "complete" : live ? "running" : "stopped",
  };
}
