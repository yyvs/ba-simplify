// shared/reports.js - the Reports page (reports.html): kept runs, their progress,
// output and evaluation.
//
// No analysis of its own: reports are shaped like history pages (see
// shared/report-store.js), so summary lines, metrics and tables come from
// shared/analysis.js, the same code as History's "Run advanced analysis".
//
// Adds progress (a report can follow a run still going), stored evaluations, and
// explicit deletion.

// --- rendering one report -------------------------------------------------

// "Running — 24 of 78 items", with a bar. complete: finished; running: still in the
// log; stopped: left the log unfinished; unknown: logged before counts were recorded.
const PROGRESS_TEXT = {
  complete: (p) => `Complete — ${p.total} ${p.total === 1 ? "item" : "items"}`,
  running: (p) => (p.total == null ? "Running" : `Running — ${p.done} of ${p.total} items`),
  stopped: (p) => `Stopped at ${p.done} of ${p.total} items — the run left the History log`,
  unknown: () => "This run recorded no item counts",
};

function buildProgressLine(report, live) {
  const progress = reportProgress(report, live);
  const line = document.createElement("div");
  line.className = `report-progress report-progress-${progress.state}`;

  const label = document.createElement("div");
  label.className = "page-aggregate";
  label.textContent = PROGRESS_TEXT[progress.state](progress);
  line.appendChild(label);

  // bar only with a known total (including complete, shown full)
  if (progress.total) {
    const track = document.createElement("div");
    track.className = "report-bar";
    const fill = document.createElement("div");
    fill.className = "report-bar-fill";
    fill.style.width = `${Math.round((progress.done / progress.total) * 100)}%`;
    track.appendChild(fill);
    line.appendChild(track);
  }
  return line;
}

// --- comparison report ----------------------------------------------------
// One report holding every (cut, arm) column over one reading of one page (see
// shared/report-store.js). Needs named columns, progress in columns and entries, and
// one evaluation table across all columns.

const COMPARISON_PROGRESS_TEXT = {
  running: (p) => `Running — run ${p.columns} of ${p.runs}, ${p.done} of ${p.total} answers`,
  stopped: (p) => `Stopped — ${p.columns} of ${p.runs} runs, ${p.done} of ${p.total} answers`,
  complete: (p) =>
    `Complete — ${p.columns} ${p.columns === 1 ? "model/method" : "models/methods"} × ${p.units} ${p.units === 1 ? "entry" : "entries"}`,
};

function buildComparisonProgressLine(report) {
  const progress = comparisonProgress(report);
  const line = document.createElement("div");
  line.className = `report-progress report-progress-${progress.state}`;

  const label = document.createElement("div");
  label.className = "page-aggregate";
  label.textContent = COMPARISON_PROGRESS_TEXT[progress.state](progress);
  line.appendChild(label);

  if (progress.total) {
    const track = document.createElement("div");
    track.className = "report-bar";
    const fill = document.createElement("div");
    fill.className = "report-bar-fill";
    fill.style.width = `${Math.round((progress.done / progress.total) * 100)}%`;
    track.appendChild(fill);
    line.appendChild(track);
  }
  return line;
}

// One line per column: heading, model, verbatim model id, and what the run was given.
// The id (e.g. `yunvs/bart-base-wikilarge-simplification`) identifies the actual
// artifact for citation; monospace as in buildSelectionLine.
//
// Per column, not arm: over both cuts one model has two lines, differing only in the
// last clause.
function buildArmsLines(report) {
  const wrap = document.createElement("div");
  wrap.className = "report-arms";
  const columns = comparisonColumns(report);
  const cutCount = comparisonCuts(report).length;
  columns.forEach((column) => {
    const row = document.createElement("div");
    row.className = "page-aggregate report-arm";

    const label = document.createElement("span");
    label.className = "report-arm-label";
    label.textContent = comparisonColumnLabel(column, cutCount);
    row.append(label, document.createTextNode(` ${modelDisplayName(column.arm.modelKey)} · `));

    const id = document.createElement("code");
    id.className = "page-model-id";
    id.textContent = column.arm.modelId || "model id not recorded";
    row.appendChild(id);

    const tail = [
      column.arm.method ? methodLabel(column.arm.method) : null,
      cutReadingLabel(column.cut, column.arm.granularity),
    ].filter(Boolean);
    if (tail.length > 0) row.appendChild(document.createTextNode(` · ${tail.join(" · ")}`));

    wrap.appendChild(row);
  });
  return wrap;
}

function buildReportMenu(report) {
  const wrap = document.createElement("div");
  wrap.className = "page-menu-wrap";

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "hist-btn page-menu-btn";
  btn.textContent = "⋮";
  btn.setAttribute("aria-label", "Report actions");

  const menu = document.createElement("div");
  menu.className = "page-menu";

  const downloadBtn = document.createElement("button");
  downloadBtn.type = "button";
  downloadBtn.textContent = "Download (JSON)";
  downloadBtn.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    wrap.classList.remove("open");
    // as stored, evaluation included, so the file matches what the page shows
    downloadJSON(report, `simplification-report-${sanitizeFilename(report.name)}.json`);
  });

  const deleteBtn = document.createElement("button");
  deleteBtn.type = "button";
  deleteBtn.textContent = "Delete";
  deleteBtn.addEventListener("click", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    wrap.classList.remove("open");
    if (!confirm(`Delete the report "${report.name}"? The run itself stays in the History log while it is still there.`)) return;
    await deleteReport(report.reportId);
    showNotice("Report deleted.");
  });

  menu.append(downloadBtn, deleteBtn);
  wrap.append(btn, menu);

  btn.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    wrap.classList.toggle("open");
  });
  document.addEventListener("click", (event) => {
    if (!wrap.contains(event.target)) wrap.classList.remove("open");
  });
  return wrap;
}

// Same reference-free metrics as History, over the report's own rows, then stored.
// Re-running is offered because a report following a live run gains rows afterwards.
function buildEvaluation(report) {
  const section = document.createElement("div");
  section.className = "report-evaluation";

  const head = document.createElement("div");
  head.className = "report-evaluation-head";

  const status = document.createElement("div");
  status.className = "page-aggregate";

  const runBtn = document.createElement("button");
  runBtn.type = "button";
  runBtn.className = "hist-btn primary";

  const results = document.createElement("div");
  results.className = "compare-results analyze-results";

  const comparison = isComparisonReport(report);
  const stored = report.evaluation;
  if (stored && Array.isArray(stored.sections)) {
    const unit = stored.entryCount === 1 ? "row" : "rows";
    status.textContent = comparison
      ? `Evaluated over ${stored.entryCount} ${stored.entryCount === 1 ? "entry" : "entries"} × ${stored.armCount} models/methods · ${formatTimestamp(stored.computedAt)}`
      : `Evaluated over ${stored.entryCount} ${unit} · ${formatTimestamp(stored.computedAt)}`;
    runBtn.textContent = "Re-run evaluation";
    renderEvaluationSections(results, stored.sections);
  } else {
    // for comparisons, explain what the evaluation does
    status.textContent = comparison
      ? "Not evaluated yet — every model/method is scored side by side against the one original, over the entries all of them answered."
      : "Not evaluated yet.";
    runBtn.textContent = "Run evaluation";
  }

  runBtn.addEventListener("click", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    runBtn.disabled = true;
    status.textContent = "Computing advanced metrics…";
    // yield once so the status paints before the long synchronous pass (as history.js)
    await new Promise((resolve) => setTimeout(resolve, 0));
    try {
      // single run: original vs one output; comparison: original vs N outputs in one table
      const columns = comparison ? comparisonColumns(report) : null;
      const entries = comparison
        ? comparableComparisonUnits(report.entries, columns)
        : analyzeEntries(report);
      const sections = comparison
        ? computeComparisonMetrics(columns, report.entries)
        : computeAdvancedMetrics(entries);
      if (!sections) {
        status.textContent = comparison
          ? "No entry was answered by every model/method, so there is nothing they can be compared over."
          : "Not enough text to compute metrics on.";
        return;
      }
      await storeEvaluation(report.reportId, {
        computedAt: Date.now(),
        entryCount: entries.length,
        // comparisons only
        ...(comparison ? { armCount: columns.length } : {}),
        sections,
      });
      showNotice("Evaluation saved to the report.");
    } catch (e) {
      console.error("report evaluation failed", e);
      status.textContent = "Evaluation failed — see the console for details.";
    } finally {
      runBtn.disabled = false;
    }
  });

  head.append(status, runBtn);
  section.append(head, results);
  return section;
}

// Stored sections are plain data (labels, cells, verdicts), rendered with
// shared/analysis.js's table builder.
function renderEvaluationSections(container, sections) {
  container.textContent = "";
  sections.forEach((section) => {
    const block = document.createElement("div");
    block.className = "compare-row analyze-section";

    const heading = document.createElement("h3");
    heading.textContent = section.title;
    block.appendChild(heading);

    if (section.note) {
      const note = document.createElement("p");
      note.className = "analyze-section-note";
      note.textContent = section.note;
      block.appendChild(note);
    }

    block.appendChild(buildMetricsTable(section));
    container.appendChild(block);
  });

  const footnote = document.createElement("p");
  footnote.className = "analyze-note";
  footnote.textContent =
    "Reference-free metrics only. SARI, BERTScore and LENS each need human reference simplifications, which a page of web text doesn't have — those are computed offline against ASSET and D-Wikipedia.";
  container.appendChild(footnote);
}

function buildReportSection(report, live, index, focusId) {
  const details = document.createElement("details");
  details.className = "page-block report-block";
  // anchor, so the producing run can open the page at this report
  details.id = `report-${report.reportId}`;
  if (focusId ? report.reportId === focusId : index === 0) details.open = true;
  if (report.reportId === focusId) details.classList.add("report-block-focused");

  const summary = document.createElement("summary");
  summary.className = "page-summary";

  const marker = document.createElement("span");
  marker.className = "page-toggle";
  marker.setAttribute("aria-hidden", "true");
  marker.textContent = "▸";

  const left = document.createElement("div");
  left.className = "page-summary-left";

  const heading = document.createElement("h2");
  heading.appendChild(buildStatusDot(report.status));
  heading.appendChild(document.createTextNode(report.name));
  left.appendChild(heading);

  const meta = document.createElement("div");
  meta.className = "page-meta";
  meta.textContent = `Saved ${formatTimestamp(report.createdAt)} · from ${truncateText(report.url || "an unrecorded page", 80)}`;
  left.appendChild(meta);

  if (isComparisonReport(report)) {
    // no selection/outcome/speed lines (single-run); per-arm equivalents are in the
    // evaluation
    left.append(buildArmsLines(report), buildComparisonProgressLine(report));
  } else {
    const stats = pageStats(report);
    left.append(buildSelectionLine(report), buildProgressLine(report, live), buildOutcomeLine(stats));
    const speed = buildSpeedLine(report, stats);
    if (speed) left.appendChild(speed);
    left.appendChild(buildMetricsLine(report.entries || []));
  }

  summary.append(marker, left, buildReportMenu(report));
  details.append(summary, buildEvaluation(report));
  return details;
}

// --- opening at one report --------------------------------------------------
// background.js's generateReportForTab and notifyComparisonFinished open this page with
// `#report-<id>` or `#comparison-<group>`; the content script knows only the group, not
// the report id the background page wrote.
function requestedReportId(reports) {
  const hash = (location.hash || "").replace(/^#/, "");
  if (!hash) return null;
  const byId = reports.find((report) => `report-${report.reportId}` === hash || report.reportId === hash);
  if (byId) return byId.reportId;
  const group = hash.replace(/^comparison-/, "");
  const byGroup = reports.find((report) => report.comparisonGroup && report.comparisonGroup === group);
  return byGroup ? byGroup.reportId : null;
}

// Scroll once per fragment: each comparison column write re-renders the page.
let scrolledTo = null;

function scrollToRequestedReport(focusId) {
  if (!focusId || scrolledTo === focusId) return;
  const block = document.getElementById(`report-${focusId}`);
  if (!block || typeof block.scrollIntoView !== "function") return;
  scrolledTo = focusId;
  block.scrollIntoView({ block: "start" });
}

// --- the page -------------------------------------------------------------

async function downloadAllReports() {
  const reports = await readReports();
  if (reports.length === 0) {
    showNotice("There are no reports to download.", { error: true });
    return;
  }
  downloadJSON({ exportedAt: Date.now(), reports }, `simplification-reports-${Date.now()}.json`);
}

async function confirmDeleteAllReports() {
  const reports = await readReports();
  if (reports.length === 0) {
    showNotice("There are no reports to delete.", { error: true });
    return;
  }
  if (!confirm(`Delete all ${reports.length} reports? The runs themselves stay in the History log while they are still there.`)) return;
  await deleteAllReports();
  showNotice("All reports deleted.");
}

async function render() {
  const [reports, history] = await Promise.all([
    readReports(),
    chrome.storage.local.get(HISTORY_STORAGE_KEY),
  ]);
  const pages = history[HISTORY_STORAGE_KEY] || [];

  // refresh reports following live runs; write only on change, since the write
  // re-renders (an unconditional one would loop)
  const refreshed = reports.map((report) => refreshedReport(report, pages));
  const changed = refreshed.some((report, i) => report !== reports[i]);
  if (changed) {
    await writeReports(refreshed);
    return; // the storage change re-renders with the new copies
  }

  const container = document.getElementById("reports-container");
  container.textContent = "";

  if (refreshed.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent =
      "No reports yet. Open History, press ⋮ on the run you want to keep and choose \"Save as report\" — the report keeps its own copy of the rows, so it survives the log rolling over.";
    container.appendChild(empty);
    return;
  }

  const focusId = requestedReportId(refreshed);
  refreshed.forEach((report, i) => {
    const live = pages.some((p) => p.sessionId === report.sourceSessionId);
    container.appendChild(buildReportSection(report, live, i, focusId));
  });
  scrollToRequestedReport(focusId);
}

document.getElementById("download-all-reports").addEventListener("click", downloadAllReports);
document.getElementById("delete-all-reports").addEventListener("click", confirmDeleteAllReports);

// History too: a report may be following a run that logs in batches.
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local") return;
  if (changes[REPORTS_STORAGE_KEY] || changes[HISTORY_STORAGE_KEY]) render();
});

// an already-open tab navigated to another report only changes the fragment
window.addEventListener("hashchange", () => {
  scrolledTo = null;
  render();
});

render();
