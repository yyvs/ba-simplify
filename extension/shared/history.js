// shared/history.js - the History page: last 20 simplified pages, opened from the
// toolbar icon's right-click "History" item (background.js's recordHistory /
// MENU_HISTORY_ID) via openInfoTab with an empty payload; the log is read from storage.
//
// Run readers, summary lines and metrics live in shared/analysis.js (shared with
// reports.html). This file: table, menus, the two modals, exports.

// column widths in % of the table. Input and Simplification are always equal; the
// remainder is split evenly across SCORE_COLUMNS.
const NUM_WIDTH = 3;
const ACTION_WIDTH = 5;
const TIME_WIDTH = 7;
const TEXT_COL_WIDTH = 33; // Input and Simplification, each
const SCORE_COLUMNS_WIDTH =
  100 - NUM_WIDTH - ACTION_WIDTH - TIME_WIDTH - TEXT_COL_WIDTH * 2;

// --- exports ---------------------------------------------------------------
// Session-level fields first, entries last. Fields are the stored ones: an export is
// a copy of the log, not a separate format.

const SESSION_IDENTITY_KEYS = [
  "sessionId",
  "timestamp",
  "title",
  "url",
  "method",
  "modelKey",
  "modelId",
  "granularity",
  "audience",
];

function sessionIdentity(page) {
  const out = {};
  SESSION_IDENTITY_KEYS.forEach((key) => {
    if (page[key] !== undefined) out[key] = page[key];
  });
  return out;
}

function pageExport(page) {
  return {
    ...sessionIdentity(page),
    ...(page.totalTimeMs !== undefined
      ? { totalTimeMs: page.totalTimeMs }
      : {}),
    ...(page.status !== undefined ? { status: page.status } : {}),
    ...(page.stats ? { stats: page.stats } : {}),
    entries: page.entries || [],
  };
}

// One row in the same envelope, without run-level totals, time and status (they
// describe rows not in the file). Keeps page and model identity.
function entryExport(page, entry) {
  return { ...sessionIdentity(page), entries: [entry] };
}

async function getPages() {
  const result = await chrome.storage.local.get(HISTORY_STORAGE_KEY);
  return result[HISTORY_STORAGE_KEY] || [];
}

async function downloadAllHistory() {
  const pages = await getPages();
  downloadJSON(pages.map(pageExport), `simplification-history-all.json`);
}

// Deleting a run also drops the cached simplifications it produced; otherwise a
// deleted page would be served again from the cache, unlogged.
//
// Routed through the service worker so the backend origin and endpoint live only in
// background.js's dropCachedPages. Best-effort: the history deletion has already
// happened, and a stopped backend has no cache.
function dropCachedPages(sessionIds) {
  const pages = (sessionIds || []).filter(Boolean);
  if (pages.length === 0) return Promise.resolve({ dropped: 0, pages: 0 });
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ cmd: "dropCachedPages", pages }, (resp) => {
      if (chrome.runtime.lastError || !resp) {
        resolve({ dropped: 0, pages: 0 });
        return;
      }
      resolve(resp);
    });
  });
}

// Zero is common: the backend restarted since the run (in-memory cache), every unit
// was skipped as too short, or entries are still held by a remaining page. The message
// doesn't guess which.
function describeCacheDrop(result) {
  const dropped =
    result && typeof result.dropped === "number" ? result.dropped : 0;
  if (dropped === 0)
    return "Nothing in the backend's cache belonged only to it.";
  return `${dropped} cached ${dropped === 1 ? "simplification" : "simplifications"} dropped with it.`;
}

// both delete functions only write storage; the onChanged listener below re-renders.
async function deleteAllHistory() {
  if (
    !confirm(
      "Delete all simplification history? The cached simplifications those runs produced " +
        "are dropped from the backend too. This can't be undone.",
    )
  )
    return;
  // read before removal: the ids are sent to the backend
  const pages = await getPages();
  await chrome.storage.local.remove(HISTORY_STORAGE_KEY);
  const result = await dropCachedPages(pages.map((p) => p.sessionId));
  showNotice(`History deleted. ${describeCacheDrop(result)}`);
}

async function deletePage(sessionId) {
  if (
    !confirm(
      "Delete this page's history? The cached simplifications this run produced are " +
        "dropped from the backend too. This can't be undone.",
    )
  )
    return;
  const pages = await getPages();
  await chrome.storage.local.set({
    [HISTORY_STORAGE_KEY]: pages.filter((p) => p.sessionId !== sessionId),
  });
  const result = await dropCachedPages([sessionId]);
  showNotice(`Page deleted. ${describeCacheDrop(result)}`);
}

// --- rendering -----------------------------------------------------------

function getFileLabel(page) {
  return (page.title || page.url || "page").trim() || "page";
}

function buildColgroup() {
  const colgroup = document.createElement("colgroup");
  const widths = [NUM_WIDTH, TEXT_COL_WIDTH, TEXT_COL_WIDTH, TIME_WIDTH]
    .concat(SCORE_COLUMNS.map(() => SCORE_COLUMNS_WIDTH / SCORE_COLUMNS.length))
    .concat([ACTION_WIDTH]);
  widths.forEach((w) => {
    const col = document.createElement("col");
    col.style.width = `${w}%`;
    colgroup.appendChild(col);
  });
  return colgroup;
}

function buildRowMenu(entry, page) {
  const wrap = document.createElement("div");
  wrap.className = "page-menu-wrap row-menu-wrap";

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "hist-btn page-menu-btn";
  btn.textContent = "⋮";
  btn.setAttribute("aria-label", "Entry actions");

  const menu = document.createElement("div");
  menu.className = "page-menu";

  const copyInputOutputBtn = document.createElement("button");
  copyInputOutputBtn.type = "button";
  copyInputOutputBtn.textContent = "Copy input and output (JSON)";
  copyInputOutputBtn.addEventListener("click", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    wrap.classList.remove("open");
    // the whole row, in the same shape "Download (JSON)" writes to a file
    const ok = await copyText(
      JSON.stringify(entryExport(page || {}, entry), null, 2),
    );
    if (ok) showNotice("Row copied to clipboard as JSON.");
    else showNotice("Couldn't copy the row to the clipboard.", { error: true });
  });

  const downloadBtn = document.createElement("button");
  downloadBtn.type = "button";
  downloadBtn.textContent = "Download (JSON)";
  downloadBtn.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    wrap.classList.remove("open");
    downloadJSON(
      entryExport(page || {}, entry),
      `simplification-entry-${Date.now()}.json`,
    );
  });

  const analyzeBtn = document.createElement("button");
  analyzeBtn.type = "button";
  analyzeBtn.textContent = "Analyze";
  analyzeBtn.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    wrap.classList.remove("open");
    openAnalyzeModal({ ...(page || {}), entries: [entry] }, "entry");
  });

  const compareBtn = document.createElement("button");
  compareBtn.type = "button";
  compareBtn.textContent = "Compare with different model";
  compareBtn.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    wrap.classList.remove("open");
    // wrapped in its page: the compare run needs the page-level model and audience
    openCompareModal({ ...(page || {}), entries: [entry] });
  });

  const deleteBtn = document.createElement("button");
  deleteBtn.type = "button";
  deleteBtn.textContent = "Delete";
  deleteBtn.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    wrap.classList.remove("open");
    if (page && page.sessionId && confirm("Delete this page's history?")) {
      deletePage(page.sessionId);
    }
  });

  menu.append(
    copyInputOutputBtn,
    downloadBtn,
    analyzeBtn,
    compareBtn,
    deleteBtn,
  );
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

// Reader-facing wording for the backend's SimplifyResponse.fallback_reason codes.
// Unknown codes are shown as-is.
const FALLBACK_REASON_LABELS = {
  no_meaningful_change: "the model's edit changed no words",
  hallucination: "the model's output was unrelated to the input",
  corpus_artifact: "the model answered with boilerplate from its training data",
  empty: "the model returned nothing",
  prompt_echo: "the model echoed the instruction back",
  refusal: "the model declined to answer",
  degenerate_repetition: "the model repeated itself",
  no_words: "the model's output had no words in it",
  too_long: "the model's output was longer than the input",
  unrelated_short_output: "the model's output was unrelated to the input",
  timeout: "the request timed out",
  error: "the request failed",
};

function fallbackReasonLabel(reason) {
  return FALLBACK_REASON_LABELS[reason] || reason;
}

// A cache hit shows "Cached" rather than its lookup time, which would read as a
// generation time.
function buildTimeCell(entry) {
  const td = document.createElement("td");
  td.className = "cell-time";
  if (entry.cached) {
    td.classList.add("cell-muted");
    td.textContent = "Cached";
    td.title =
      "Served from the backend's cache — this text had been simplified before.";
    return td;
  }
  td.textContent = formatLatency(entry.requestTimeMs);
  return td;
}

function buildRow(entry, page, index) {
  const input = entryInput(entry);
  const output = entryOutput(entry);

  const tr = document.createElement("tr");
  if (!entry.changed) tr.classList.add("unchanged");

  const num = document.createElement("td");
  num.textContent = String(index + 1);

  const inputCell = document.createElement("td");
  inputCell.className = "history-text";
  inputCell.textContent = input;

  // Unchanged rows show why the original was kept and the rejected model answer,
  // instead of repeating the input.
  const outputCell = document.createElement("td");
  outputCell.className = "history-text";
  if (entry.changed) {
    outputCell.textContent = output;
  } else {
    outputCell.classList.add("cell-muted");
    const note = document.createElement("div");
    note.className = "entry-meta entry-meta-warn";
    // older builds recorded no reason
    note.textContent = entry.fallbackReason
      ? `Kept the original — ${fallbackReasonLabel(entry.fallbackReason)}.`
      : "Kept the original.";
    if (entry.fallbackReason)
      note.title = `fallback_reason: ${entry.fallbackReason}`;
    outputCell.appendChild(note);

    // The rejected answer, untruncated: reasons like degenerate repetition or corpus
    // artifact are claims about the whole answer. Skipped for pre-`modelResult`
    // entries (falls back to the served text) and empty generations.
    const modelResult = entryModelResult(entry);
    if (modelResult && modelResult !== output) {
      const rejected = document.createElement("div");
      rejected.className = "entry-meta entry-meta-rejected";
      rejected.textContent = modelResult.trim();
      outputCell.appendChild(rejected);
    }
  }

  tr.append(num, inputCell, outputCell, buildTimeCell(entry));

  SCORE_COLUMNS.forEach((col) => {
    const td = document.createElement("td");
    // Changed rows only: an unchanged row would score its input against itself.
    const result = entry.changed ? col.perSentence(input, output) : null;
    td.textContent = result ? result.text : "—";
    if (!entry.changed) td.classList.add("cell-muted");
    tr.appendChild(td);
  });

  const actionCell = document.createElement("td");
  actionCell.appendChild(buildRowMenu(entry, page));
  tr.appendChild(actionCell);

  return tr;
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (e) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "true");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    try {
      document.execCommand("copy");
    } catch (copyError) {
      console.error("copy failed", copyError);
      return false;
    } finally {
      textarea.remove();
    }
    return true;
  }
}

function makePageMenu(page) {
  const wrap = document.createElement("div");
  wrap.className = "page-menu-wrap";

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "hist-btn page-menu-btn";
  btn.textContent = "⋮";
  btn.setAttribute("aria-label", "Page actions");

  const menu = document.createElement("div");
  menu.className = "page-menu";

  const downloadBtn = document.createElement("button");
  downloadBtn.type = "button";
  downloadBtn.textContent = "Download (JSON)";
  downloadBtn.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    wrap.classList.remove("open");
    downloadJSON(
      pageExport(page),
      `simplification-history-${sanitizeFilename(getFileLabel(page))}.json`,
    );
  });

  const analyzeBtn = document.createElement("button");
  analyzeBtn.type = "button";
  analyzeBtn.textContent = "Analyze";
  analyzeBtn.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    wrap.classList.remove("open");
    openAnalyzeModal(page, "page");
  });

  // The log silently drops its oldest page after 20; a report keeps its own copy of
  // the rows on reports.html (see shared/report-store.js).
  const reportBtn = document.createElement("button");
  reportBtn.type = "button";
  reportBtn.textContent = "Save as report";
  reportBtn.addEventListener("click", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    wrap.classList.remove("open");
    try {
      const { already } = await saveReportForPage(page);
      showNotice(
        already
          ? "Report updated with this run's latest rows."
          : "Saved to Reports.",
      );
    } catch (e) {
      console.error("saving a report failed", e);
      showNotice("Couldn't save this run as a report.", { error: true });
    }
  });

  const compareBtn = document.createElement("button");
  compareBtn.type = "button";
  compareBtn.textContent = "Compare with different model";
  compareBtn.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    wrap.classList.remove("open");
    openCompareModal(page);
  });

  const deleteBtn = document.createElement("button");
  deleteBtn.type = "button";
  deleteBtn.textContent = "Delete";
  deleteBtn.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    wrap.classList.remove("open");
    deletePage(page.sessionId);
  });

  menu.append(downloadBtn, reportBtn, analyzeBtn, compareBtn, deleteBtn);
  wrap.append(btn, menu);

  btn.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    wrap.classList.toggle("open");
  });

  document.addEventListener("click", (event) => {
    if (!wrap.contains(event.target)) {
      wrap.classList.remove("open");
    }
  });

  return wrap;
}

function fillCompareModelSelect() {
  const select = document.getElementById("compare-model");
  select.innerHTML = "";
  MODEL_KEYS.forEach((key) => {
    const option = document.createElement("option");
    option.value = key;
    // name and scope in one string (an <option> holds only text); without scope the two
    // fine-tuned checkpoints share a name
    option.textContent = modelDisplayNameWithScope(key);
    select.appendChild(option);
  });
}

function setCompareStatus(text) {
  const el = document.getElementById("compare-status");
  el.textContent = text;
}

function appendCompareRows(rows) {
  const container = document.getElementById("compare-results");
  container.textContent = "";
  if (!rows.length) {
    container.textContent = "No comparison rows to show.";
    return;
  }

  rows.forEach((row, index) => {
    const block = document.createElement("div");
    block.className = "compare-row";

    const heading = document.createElement("h3");
    heading.textContent = `Sentence ${index + 1}`;

    const grid = document.createElement("div");
    grid.className = "compare-grid";

    const original = document.createElement("div");
    original.className = "compare-box";
    original.textContent = row.original;
    const simplified = document.createElement("div");
    simplified.className = "compare-box";
    simplified.textContent = row.simplified;

    const originalHeader = document.createElement("div");
    originalHeader.textContent = "Original";
    const simplifiedHeader = document.createElement("div");
    simplifiedHeader.textContent = "Selected model output";

    const originalWrap = document.createElement("div");
    originalWrap.appendChild(originalHeader);
    originalWrap.appendChild(original);
    const simplifiedWrap = document.createElement("div");
    simplifiedWrap.appendChild(simplifiedHeader);
    simplifiedWrap.appendChild(simplified);

    grid.append(originalWrap, simplifiedWrap);
    block.append(heading, grid);
    container.appendChild(block);
  });
}

async function runCompare(page, modelKey) {
  setCompareStatus(
    `Running comparison with ${modelDisplayNameWithScope(modelKey)}…`,
  );
  const rows = [];

  for (const entry of page.entries) {
    const input = entryInput(entry);
    try {
      const result = await chrome.runtime.sendMessage({
        cmd: "fetchSimplifyWithModel",
        text: input,
        model: modelKey,
        // recorded only for prompted models; otherwise the default, which the backend
        // normalizes to anyway
        audience: page.audience || DEFAULT_AUDIENCE,
      });

      if (
        !result ||
        !result.ok ||
        !result.data ||
        typeof result.data.simplified !== "string"
      ) {
        rows.push({
          original: input,
          simplified: `Comparison failed: ${result && result.error ? result.error : "unknown error"}`,
        });
        continue;
      }

      rows.push({ original: input, simplified: result.data.simplified });
    } catch (e) {
      rows.push({
        original: input,
        simplified: `Comparison failed: ${String(e)}`,
      });
    }
  }

  appendCompareRows(rows);
  setCompareStatus(
    `${rows.length} requests fired against ${modelDisplayNameWithScope(modelKey)}.`,
  );
}

function openCompareModal(page) {
  const modal = document.getElementById("compare-modal");
  const select = document.getElementById("compare-model");
  const results = document.getElementById("compare-results");
  const status = document.getElementById("compare-status");

  fillCompareModelSelect();
  results.textContent = "";
  status.textContent = "";
  modal.classList.remove("hidden");
  modal.setAttribute("aria-hidden", "false");

  // open the dropdown on the page's own model
  const current = page.modelKey || "online";
  select.value = current;

  const runBtn = document.getElementById("compare-run");
  runBtn.onclick = () => runCompare(page, select.value);
}

function closeCompareModal() {
  const modal = document.getElementById("compare-modal");
  modal.classList.add("hidden");
  modal.setAttribute("aria-hidden", "true");
}

function buildAnalyzeSummary(pageOrEntry) {
  const entries = analyzeEntries(pageOrEntry);
  const changed = entries.filter((entry) => entry && entry.changed);
  const beforeFkgl = entries
    .map((entry) => fleschKincaidGrade(entry.input))
    .filter((v) => v != null);
  const afterFkgl = entries
    .map((entry) => fleschKincaidGrade(entry.output))
    .filter((v) => v != null);
  const beforeWords = entries.reduce(
    (sum, entry) => sum + (entry.input.match(/[A-Za-z']+/g) || []).length,
    0,
  );
  const afterWords = entries.reduce(
    (sum, entry) => sum + (entry.output.match(/[A-Za-z']+/g) || []).length,
    0,
  );
  const fkgl =
    beforeFkgl.length && afterFkgl.length
      ? `${(beforeFkgl.reduce((a, b) => a + b, 0) / beforeFkgl.length).toFixed(1)} → ${(afterFkgl.reduce((a, b) => a + b, 0) / afterFkgl.length).toFixed(1)}`
      : "n/a";

  const wordPct =
    beforeWords > 0 ? Math.round((1 - afterWords / beforeWords) * 100) : null;

  return {
    total: entries.length,
    changed: changed.length,
    fkgl,
    words: wordPct == null ? "n/a" : formatPercentDelta(wordPct),
  };
}

function renderAnalyzeSummary(target) {
  const stats = buildAnalyzeSummary(target);
  const container = document.getElementById("analyze-summary");
  container.innerHTML = "";

  const grid = document.createElement("div");
  grid.className = "analyze-grid";

  [
    ["FKGL", stats.fkgl],
    ["Words changed", stats.words],
    ["Rows simplified", `${stats.changed}/${stats.total}`],
  ].forEach(([label, value]) => {
    const stat = document.createElement("div");
    stat.className = "analyze-stat";
    const labelEl = document.createElement("span");
    labelEl.className = "label";
    labelEl.textContent = label;
    const valueEl = document.createElement("span");
    valueEl.className = "value";
    valueEl.textContent = value;
    stat.append(labelEl, valueEl);
    grid.appendChild(stat);
  });

  container.appendChild(grid);
}

// Returns the number of entries analyzed, so the caller can say what was covered.
function renderAdvancedAnalysis(target) {
  const container = document.getElementById("analyze-results");
  container.textContent = "";

  const entries = analyzeEntries(target);
  const sections = computeAdvancedMetrics(entries);
  if (!sections) {
    container.textContent = "Not enough text to compute metrics on.";
    return 0;
  }

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
    "Reference-free metrics only. SARI, BERTScore and LENS each need human reference simplifications, which the history doesn't hold — those are computed offline against ASSET and D-Wikipedia.";
  container.appendChild(footnote);

  return entries.length;
}

function openAnalyzeModal(target, mode = "page") {
  const modal = document.getElementById("analyze-modal");
  const title = document.getElementById("analyze-title");
  const summary = document.getElementById("analyze-summary");
  const status = document.getElementById("analyze-status");
  const results = document.getElementById("analyze-results");

  if (mode === "entry") {
    title.textContent = "Analyze sentence";
  } else {
    title.textContent = "Analyze page";
  }

  summary.innerHTML = "";
  results.textContent = "";
  status.textContent =
    "Quick stats are shown immediately. Run advanced analysis for the full metric set.";
  renderAnalyzeSummary(target);

  modal.classList.remove("hidden");
  modal.setAttribute("aria-hidden", "false");

  const runBtn = document.getElementById("analyze-run");
  runBtn.disabled = false;
  runBtn.onclick = async () => {
    runBtn.disabled = true;
    results.textContent = "";
    status.textContent = "Computing advanced metrics…";
    // yield once so the status paints before the long synchronous pass
    await new Promise((resolve) => setTimeout(resolve, 0));
    try {
      const count = renderAdvancedAnalysis(target);
      status.textContent = count
        ? `Advanced metrics over ${count} ${count === 1 ? "row" : "rows"}.`
        : "Nothing to analyze.";
    } catch (e) {
      console.error("advanced analysis failed", e);
      status.textContent =
        "Advanced analysis failed — see the console for details.";
    } finally {
      runBtn.disabled = false;
    }
  };
}

function closeAnalyzeModal() {
  const modal = document.getElementById("analyze-modal");
  modal.classList.add("hidden");
  modal.setAttribute("aria-hidden", "true");
}

function buildPageSection(page, index) {
  const details = document.createElement("details");
  details.className = "page-block";
  if (index === 0) details.open = true;

  const summary = document.createElement("summary");
  summary.className = "page-summary";

  // A `display: flex` <summary> loses the browser's disclosure triangle, so draw one.
  // aria-hidden: <details> already exposes the expanded state.
  const marker = document.createElement("span");
  marker.className = "page-toggle";
  marker.setAttribute("aria-hidden", "true");
  marker.textContent = "▸";

  const left = document.createElement("div");
  left.className = "page-summary-left";
  left.append(...buildPageSummary(page));

  const menu = makePageMenu(page);
  summary.append(marker, left, menu);

  const table = document.createElement("table");
  table.appendChild(buildColgroup());

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  const columns = [
    { label: "#" },
    { label: "Input" },
    { label: "Simplification" },
    {
      label: "Time",
      tooltip: "How long this unit's round trip to the backend took.",
    },
  ]
    .concat(SCORE_COLUMNS.map((c) => ({ label: c.label, tooltip: c.tooltip })))
    // no visible header over the ⋮ buttons; name kept for screen readers
    .concat([{ label: "", ariaLabel: "Actions" }]);
  columns.forEach(({ label, tooltip, ariaLabel }) => {
    const th = document.createElement("th");
    th.textContent = label;
    if (tooltip) th.title = tooltip;
    if (ariaLabel) th.setAttribute("aria-label", ariaLabel);
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);

  const tbody = document.createElement("tbody");
  page.entries.forEach((entry, i) =>
    tbody.appendChild(buildRow(entry, page, i)),
  );

  table.append(thead, tbody);
  details.append(summary, table);
  return details;
}

async function render() {
  const result = await chrome.storage.local.get(HISTORY_STORAGE_KEY);
  const pages = result[HISTORY_STORAGE_KEY] || [];
  const container = document.getElementById("history-container");
  container.textContent = "";

  if (pages.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent =
      "No pages simplified yet — simplify a page and it'll show up here.";
    container.appendChild(empty);
    return;
  }

  pages.forEach((page, i) => container.appendChild(buildPageSection(page, i)));
}

document
  .getElementById("download-all")
  .addEventListener("click", downloadAllHistory);
document
  .getElementById("delete-all")
  .addEventListener("click", deleteAllHistory);
document
  .getElementById("compare-close")
  .addEventListener("click", closeCompareModal);
document.getElementById("compare-modal").addEventListener("click", (event) => {
  if (event.target.id === "compare-modal") closeCompareModal();
});
document
  .getElementById("analyze-close")
  .addEventListener("click", closeAnalyzeModal);
document.getElementById("analyze-modal").addEventListener("click", (event) => {
  if (event.target.id === "analyze-modal") closeAnalyzeModal();
});

// live-update an open tab, as status.js does
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes[HISTORY_STORAGE_KEY]) {
    render();
  }
});

render();
