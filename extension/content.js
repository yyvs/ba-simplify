// In-page half of the extension: collects the page's text, sends it to the backend via
// the service worker, writes the answers back, and can restore the page exactly. Runs on
// every http(s) page; the shared modules in manifest.json's content_scripts load first
// and leave their constants as globals in this scope.

let isSimplified = false;
let observer = null;
// Running totals for the current session, reset in simplifyPage() and cumulative across
// the initial pass and every later observer batch.
// foundTotal (z): leaf-level candidates discovered; skippedCount (x): discovered but
// never sent (opted out, code-like, empty, already processed, ...); sentCount (y): sent,
// regardless of outcome; changedCount (w): sent and returned with different text
// (w <= y; short text such as nav links often comes back unchanged).
// foundTotal >= skippedCount + sentCount, with equality once every item has settled.
let foundTotal = 0;
let skippedCount = 0;
let sentCount = 0;
let changedCount = 0;
// Document mode only: prose that appeared after the run and was deliberately left alone
// (startDocumentWatcher). Not counted as found, since the run never intends to process it.
let unsimplifiedLateCount = 0;
// Document mode only: units sent inside a section the model answered that received none
// of the answer. Document models delete (nine paragraphs can come back as three
// sentences), and a section is one item however many elements it spans, so without this
// count the untouched paragraphs are invisible. Neither a skip nor a failure.
let unwrittenUnitCount = 0;
// Skip reasons, since skippedCount alone can't tell a filter working as designed from
// one eating the page. skipReasons: per discovered item never sent. chunkSkipReasons: per
// text chunk left behind inside an element that was walked (an element can be sent and
// still skip chunks, so merging the two would double-count).
// Each reason keeps a small, clipped text sample for judging the filter.
const SKIP_SAMPLE_LIMIT = 5;
const SKIP_SAMPLE_CHARS = 160;
let skipReasons = Object.create(null);
let chunkSkipReasons = Object.create(null);
let skipSamples = Object.create(null);

function noteSkip(tally, reason, text) {
  if (!reason) return;
  tally[reason] = (tally[reason] || 0) + 1;
  const samples = (skipSamples[reason] ||= []);
  const trimmed = (text || "").trim().replace(/\s+/g, " ");
  if (trimmed && samples.length < SKIP_SAMPLE_LIMIT) {
    samples.push(trimmed.slice(0, SKIP_SAMPLE_CHARS));
  }
}

// Only caller is research/audit_sites.mjs, which evaluates it in the page so the audit
// reports the guards' own tallies. Keep the shape stable.
function skipBreakdown() {
  return {
    items: { ...skipReasons },
    chunks: { ...chunkSkipReasons },
    samples: Object.fromEntries(
      Object.entries(skipSamples).map(([k, v]) => [k, [...v]]),
    ),
  };
}

function resetSkipBreakdown() {
  skipReasons = Object.create(null);
  chunkSkipReasons = Object.create(null);
  skipSamples = Object.create(null);
}

let lateContentTimer = 0;
// Each discovered item's outcome, so a later DOM removal (e.g. a virtualized list
// recycling rows) can un-count it from the right bucket; see forgetItem().
const itemStatus = new WeakMap();
// Separate from itemStatus: "changed" is independent of the outcome.
const changedItems = new WeakSet();
// Elements processed as their own unit this session; collectChunkNodes() excludes them
// from any ancestor's text. Filled by registerFound() before any extraction, replaced
// per run in simplifyPage().
let ownUnits = new WeakSet();
// Last request failure (e.g. backend down); shown once in the notice box.
let lastSimplifyError = null;

// --- run info for the notice box ---
// Which model, method and settings are doing the work (the selection lives in the
// service worker's storage and model ids come from /health, so background.js fills in
// `selection`), and how fast.
// Timing is split at the preflight: preflightMs is a fixed per-toggle cost (a real test
// generation, up to a minute on the prompted LLM path; thesis §3.4), the rest scales
// with page text.
const session = {
  phase: "idle", // "preflight" | "simplifying" | "settled"
  selection: null, // describeSelection() payload from background.js
  clickedAt: 0, // toolbar icon clicked — start of what the user actually waits
  preflightMs: 0, // health check + test generation, 0 until it resolves
  startedAt: 0, // first real page request — excludes the preflight above
  settledAt: 0, // when the last in-flight item settled (0 while still running)
  requests: 0, // completed backend round trips (one per chunk, not per item)
  requestMsTotal: 0,
  cachedRequests: 0, // of those, how many the backend served from its cache
};

// Re-render on a timer as well, so elapsed times keep moving during the preflight and
// slow requests. Stopped once everything has settled.
const NOTICE_TICK_MS = 1000;
let noticeTicker = null;

function startNoticeTicker() {
  if (noticeTicker) return;
  noticeTicker = setInterval(renderProgressNotice, NOTICE_TICK_MS);
}

function stopNoticeTicker() {
  if (!noticeTicker) return;
  clearInterval(noticeTicker);
  noticeTicker = null;
}

// --- one run at a time ---
// Rationale and record format: shared/active-run.js. Page side only: claim before
// starting, report while running, release when stopping. The service worker owns the
// lock and decides.

function claimRun(run) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ cmd: "claimRun", run }, (resp) => {
      if (chrome.runtime.lastError || !resp) {
        // No answer means a broken worker, not a refusal: run unlocked rather than
        // never simplifying.
        console.warn("no answer to the run claim; starting unlocked");
        resolve({ ok: true, holder: null });
        return;
      }
      resolve(resp);
    });
  });
}

function releaseRun() {
  chrome.runtime.sendMessage({ cmd: "releaseRun" }, () => {
    // Reading lastError marks it handled; the worker's staleness window covers a failed
    // release.
    if (chrome.runtime.lastError) return;
  });
}

// Sent from renderProgressNotice on every state change and tick. Doubles as the worker's
// heartbeat for telling a live run from an abandoned record, so it is sent even when the
// counts haven't changed.
function reportRunProgress(extra) {
  const progress = {
    phase: session.phase,
    done: sentCount,
    total: Math.max(foundTotal - skippedCount, sentCount),
    ...(extra || {}),
  };
  chrome.runtime.sendMessage({ cmd: "runProgress", progress }, () => {
    if (chrome.runtime.lastError) return;
  });
}

// This page's entry in the record other tabs read.
function runDescriptor(kind, extra) {
  return {
    kind,
    sessionId: pageSessionId,
    url: location.href,
    title: (document.title || "").trim(),
    host: location.hostname,
    ...(extra || {}),
  };
}

// --- "already running" notice ---
// Shown on the page that couldn't start: which page holds the run, its progress, and
// stop/wait. Deliberately no "queue" option: a queued run could start minutes later
// against a changed page with nobody watching.
let blockedByRun = null;
let blockedWatcherAttached = false;

const BLOCKED_STOP_ID = "simplify-stop-other";
const BLOCKED_WAIT_ID = "simplify-wait";

function showBlockedNotice(holder) {
  blockedByRun = holder;
  attachBlockedWatcher();
  renderBlockedNotice();
}

function hideBlockedNotice(message) {
  blockedByRun = null;
  if (message) showNotice(message, 5000);
}

// The holder's record is in storage, so the bar tracks the other tab's real progress
// without polling.
function attachBlockedWatcher() {
  if (blockedWatcherAttached) return;
  blockedWatcherAttached = true;
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local" || !blockedByRun) return;
    if (!(ACTIVE_RUN_STORAGE_KEY in changes)) return;
    const holder = changes[ACTIVE_RUN_STORAGE_KEY].newValue;
    if (!holder) {
      hideBlockedNotice(
        "The other page has finished. Click the icon to simplify this one.",
      );
      return;
    }
    blockedByRun = holder;
    renderBlockedNotice();
  });
}

function renderBlockedNotice() {
  const holder = blockedByRun;
  if (!holder) return;
  const total = holder.total || 0;
  const rows = [
    noticeRow("div", "sn-title", "Already simplifying another page"),
    noticeRow("div", "sn-outcome", activeRunSummary(holder)),
    total > 0
      ? progressBarRow(Math.min(holder.done || 0, total), total)
      : indeterminateBarRow(),
    noticeRow(
      "div",
      "sn-muted",
      "One page at a time: the models are one local service, so two runs at once make " +
        "both slower and neither one's timings mean anything.",
    ),
  ];

  const actions = noticeRow("div", "sn-actions");
  const stop = noticeRow("button", "sn-btn", "Stop that page");
  stop.id = BLOCKED_STOP_ID;
  stop.type = "button";
  stop.title = `Stop the run on ${activeRunLabel(holder)}. What it has already simplified stays.`;
  stop.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    chrome.runtime.sendMessage({ cmd: "stopActiveRun" }, (resp) => {
      if (chrome.runtime.lastError) return;
      // The resulting storage change normally clears this notice; this covers a holder
      // that was already gone.
      hideBlockedNotice(
        resp && resp.stopped
          ? "Stopped the other page. Click the icon to simplify this one."
          : "That run had already finished. Click the icon to simplify this page.",
      );
    });
  });

  const wait = noticeRow("button", "sn-btn", "Wait");
  wait.id = BLOCKED_WAIT_ID;
  wait.type = "button";
  wait.title =
    "Leave the other page running. This notice closes; nothing is queued.";
  wait.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    // Dismiss only; nothing is queued. Reappears on the next click if that run is
    // still going.
    hideBlockedNotice();
    const box = document.getElementById("simplify-notice");
    if (box) box.remove();
  });

  actions.append(stop, wait);
  rows.push(actions);
  showNotice(rows, 0);
}

// --- comparing several models on one page ---
// An evaluation run: the page is never written to, since rewriting it would destroy the
// text the next model reads.
// Inputs are collected once and reused for every model, so page changes in between (an
// ad slot, a lazy paragraph) can't show up as model differences. One collector per
// comparison, since collectors produce different units; the panel
// (shared/compare-picker.js) chooses the unit first.
// Arms may read a different unit than the page was cut into: a section sent to a
// sentence-scope checkpoint is split, simplified per sentence and rejoined by the backend
// (backend/main.py simplify_text). Each arm records the unit it read (`granularity`)
// next to the page's cut (`unitGranularity`).
// Models run sequentially: they share one local service, and concurrent requests would
// measure contention rather than the models.
const comparison = {
  running: false,
  cuts: [],
  models: [],
  // Every (cut, model) pair in order; a model that can read both chosen cuts appears
  // in both.
  runs: [],
  runIndex: 0, // 1-based, for reporting
  // One entry per unit of the coarsest chosen cut; every arm's answer is filed under
  // it. See collectComparisonPlan.
  entries: [],
  plan: null,
  done: 0,
  total: 0,
  stopRequested: false,
  // Cut the entries are units of (not necessarily the cut each arm read).
  unitGranularity: null,
};

function comparisonIsRunning() {
  return comparison.running;
}

// Called by the panel. Claims the lock like any other run.
function onComparisonRequested(cuts, models) {
  if (comparison.running) {
    showNotice("A comparison is already running on this page.", 5000);
    return;
  }
  claimRun(
    runDescriptor(ACTIVE_RUN_COMPARE, { models: models.length, modelIndex: 0 }),
  ).then((claim) => {
    if (!claim.ok) {
      showBlockedNotice(claim.holder);
      return;
    }
    runComparison(cuts, models);
  });
}

// The page as the chosen cuts read it, as plain strings (nothing is written back, so
// only the text matters).
// One entry per unit of the coarsest chosen cut, plus per cut the requests it makes for
// that entry. A section is built from the units the sentence cut sends, so one row can
// hold both the whole-section answer and the joined unit answers. With one cut, this is
// one request per entry.
//
//   entries:  ["Residents are requested to ...\n\nCollections will commence ...", ...]
//   requests: { whole_sections:       [["Residents ... Collections ..."], ...]
//               sentence_by_sentence: [["Residents ...", "Collections ..."], ...] }
//
// The sections cut sends `unit.text` joined with spaces, not the entry text: that is what
// the document path sends (simplifySection) and the concatenation of the sentence cut's
// requests, so both cuts get exactly the same words.
function collectComparisonPlan(cuts, maxTokens) {
  const plan = { unitGranularity: cuts[0], entries: [], requests: {} };
  cuts.forEach((cut) => (plan.requests[cut] = []));

  if (cuts.includes(GRANULARITY_WHOLE_SECTIONS)) {
    collectSections(findContentScope(), maxTokens).forEach((units) => {
      plan.entries.push(sectionOriginalText(units));
      plan.requests[GRANULARITY_WHOLE_SECTIONS].push([
        units.map((unit) => unit.text).join(" "),
      ]);
      if (plan.requests[GRANULARITY_SENTENCE_BY_SENTENCE]) {
        plan.requests[GRANULARITY_SENTENCE_BY_SENTENCE].push(
          units.map((unit) => unit.text),
        );
      }
    });
    return plan;
  }

  const elems = computeLeafCandidates(
    Array.from(document.querySelectorAll(CANDIDATE_SELECTOR)),
  );
  elems.forEach((el) => {
    if (!shouldSimplifyElement(el)) return;
    collectChunkNodes(el).forEach((nodes) => {
      const text = nodes
        .map((n) => n.data)
        .join("")
        .trim();
      if (!isSimplifiableChunk(text)) return;
      plan.entries.push(text);
      plan.requests[GRANULARITY_SENTENCE_BY_SENTENCE].push([text]);
    });
  });
  return plan;
}

// Whether a model can be an arm in this cut. Sentence-scope models read either (the
// backend splits sections into sentences and rejoins). Whole-sections checkpoints read
// the sections cut only: one leaf element is not a document-scope input.
function modelReadsCut(model, cut, granularities) {
  if (modelGranularity(model, granularities) !== GRANULARITY_WHOLE_SECTIONS)
    return true;
  return cut === GRANULARITY_WHOLE_SECTIONS;
}

// Includes the cut: the same model over both cuts is two runs, and one id would let the
// second overwrite the first's summary.
function comparisonSessionId(model, cut) {
  return `${pageSessionId}-compare-${cutShortLabel(cut)}-${model}`;
}

// Writes one arm's results to the History log under its own session id, as the same
// record as an ordinary run (the log is also where cached units are attributed per
// page). The comparison itself lives in flushComparisonArm's report: the log holds arms
// as unrelated pages with no link between them.
function flushComparisonModel(arm, cut, entries, stats, status) {
  const model = arm.modelKey;
  chrome.runtime.sendMessage({
    cmd: "recordHistory",
    page: {
      sessionId: comparisonSessionId(model, cut),
      url: location.href,
      title: document.title,
      timestamp: Date.now(),
    },
    session: {
      method: arm.method,
      modelKey: model,
      // Repo id / Ollama tag from /health, naming the checkpoint ("finetuned" is only a
      // key). The ordinary path gets it from the preflight (background.js
      // describeSelection); a comparison has none, so it reads /health once at the start.
      modelId: arm.modelId,
      granularity: arm.granularity,
      status,
      totalTimeMs: stats.totalTimeMs,
      stats: {
        totalItems: comparison.entries.length,
        skippedItems: 0,
        processedItems: stats.processed,
        simplifiedItems: stats.changed,
        unchangedItems: Math.max(stats.processed - stats.changed, 0),
        cachedResponses: stats.cached,
      },
      // Marks the row as a comparison arm. The group id also keys the combined report
      // (shared/report-store.js saveComparisonRun).
      comparison: {
        group: pageSessionId,
        label: arm.label,
        // with both cuts ticked, the only thing separating this row from the same model's
        // other one
        cut,
        modelIndex: comparison.runIndex,
        models: comparison.runs.length,
        unitGranularity: comparison.unitGranularity,
      },
    },
    entries,
  });
}

// The comparison as one report, rewritten each time an arm finishes.
// `results` is aligned to `comparison.inputs` by position, with null where this arm has
// no answer. Merging by input text would collide on repeated text; dropping unanswered
// entries would misalign the arm.
function flushComparisonArm(arm, cut, results, stats, runStatus) {
  chrome.runtime.sendMessage({
    cmd: "recordComparison",
    run: {
      group: pageSessionId,
      page: {
        url: location.href,
        title: document.title,
        timestamp: Date.now(),
      },
      unitGranularity: comparison.unitGranularity,
      cutOrder: comparison.cuts,
      // arm-runs, not models (a model ticked for both cuts counts twice)
      models: comparison.runs.length,
      inputs: comparison.entries,
      cut,
      arm,
      run: {
        status: runStatus === "running" ? "completed" : runStatus,
        totalTimeMs: stats.totalTimeMs,
        stats: {
          totalItems: comparison.entries.length,
          requests: stats.requests,
          processedItems: stats.processed,
          simplifiedItems: stats.changed,
          unchangedItems: Math.max(stats.processed - stats.changed, 0),
          cachedResponses: stats.cached,
        },
      },
      results,
      // whole run's state, not this arm's: a completed arm of a stopped comparison is
      // reported as such
      runStatus,
    },
  });
}

// One arm-run's answer to one entry: the facts of a History row (see historyEntry) minus
// the input, which the report stores once.
// `requests`: one item in the sections cut, one per unit in the sentence cut. Unit
// answers are joined into the section's answer and kept individually in `parts`.
// Null unless every request was answered: a partial join would silently miss a paragraph.
function comparisonResult(requests) {
  const answered = requests.filter((request) => request.data);
  if (answered.length === 0 || answered.length < requests.length) return null;

  const norm = (str) => String(str).replace(/\s+/g, " ").trim();
  const parts = answered.map(({ text, data }) => ({
    input: text,
    output: data.simplified,
    modelResult:
      typeof data.model_result === "string"
        ? data.model_result
        : data.simplified,
    changed: norm(data.simplified) !== norm(text),
    cached: !!data.cached,
    fallbackReason: data.fallback_reason || null,
    requestTimeMs:
      typeof data.requestTimeMs === "number" ? data.requestTimeMs : null,
  }));

  const result = {
    output: parts.map((part) => part.output).join(" "),
    modelResult: parts.map((part) => part.modelResult).join(" "),
    changed: parts.some((part) => part.changed),
    // an entry is only a cache hit if none of it had to be generated
    cached: parts.every((part) => part.cached),
    // first guard that fired, the same rule the backend uses for a multi-sentence request
    fallbackReason:
      (parts.find((part) => part.fallbackReason) || {}).fallbackReason || null,
    // summed: the entry's total cost to this arm
    requestTimeMs: parts.reduce(
      (total, part) => total + (part.requestTimeMs || 0),
      0,
    ),
  };
  // Only for multi-request entries, so sections-cut rows don't look split.
  if (parts.length > 1) result.parts = parts;

  // The backend's sentence split, for a sections-cut arm whose model reads sentences.
  const split = answered.length === 1 ? answered[0].data.sentence_split : null;
  if (split && Array.isArray(split.parts) && split.count > 1) {
    result.sentenceSplit = { count: split.count, parts: split.parts };
  }
  return result;
}

// Model ids, methods and granularities from /health, read once per run (a comparison has
// no preflight; cf. background.js describeSelection). Resolves to {} on failure: the run
// still works with keys alone, and shared/model-labels.js's fallback maps name the method.
function fetchComparisonModelFacts() {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ cmd: "health" }, (resp) => {
      if (chrome.runtime.lastError || !resp || !resp.ok || !resp.health) {
        resolve({});
        return;
      }
      const health = resp.health;
      resolve({
        names: health.model_names || null,
        methods: health.methods || null,
        granularities: health.granularities || null,
      });
    });
  });
}

function reportComparisonProgress() {
  reportRunProgress({
    phase: "simplifying",
    done: comparison.done,
    total: comparison.total,
    // differ once a model runs in two cuts: wording names models, "x of y" counts runs
    // (see activeRunSummary)
    models: comparison.models.length,
    runs: comparison.runs.length,
    modelIndex: comparison.runIndex,
  });
}

function renderComparisonNotice(run, finished, stopped) {
  const total = comparison.total;
  const models = comparison.models.length;
  const rows = [
    noticeRow(
      "div",
      "sn-title",
      !finished
        ? `Comparing models on ${pageLabel()}`
        : stopped
          ? `Stopped comparing models on ${pageLabel()}`
          : `Compared ${models} ${models === 1 ? "model" : "models"} ✓`,
    ),
  ];
  if (finished) {
    rows.push(processedRow(comparison.done, total));
  } else {
    rows.push(
      progressBarRow(comparison.done, total),
      noticeRow(
        "div",
        "sn-outcome",
        // with both cuts ticked the same model appears twice, so name the cut too
        `Run ${comparison.runIndex} of ${comparison.runs.length}: ${modelDisplayName(run.model)}` +
          `${comparison.cuts.length > 1 ? ` · ${cutShortLabel(run.cut)}` : ""}`,
      ),
    );
  }
  rows.push(
    noticeRow(
      "div",
      "sn-muted",
      finished
        ? "The arms are combined into one report. Open Reports from the toolbar icon's menu."
        : "The page is not being changed — every answer goes to its report.",
    ),
  );
  if (!finished) rows.push(stopButtonRow());
  showNotice(rows, finished ? 8000 : 0);
}

async function runComparison(cuts, models) {
  comparison.running = true;
  comparison.cuts = cuts;
  comparison.models = models;
  comparison.runIndex = 0;
  comparison.done = 0;
  comparison.stopRequested = false;
  // collectors tally skips into the page's counters; reset so a later run's notice
  // doesn't inherit them
  foundTotal = 0;
  skippedCount = 0;
  sentCount = 0;
  changedCount = 0;
  unwrittenUnitCount = 0;
  resetSkipBreakdown();
  ownUnits = new WeakSet();
  startNoticeTicker();

  // Collected once by the coarsest chosen cut's collector, reused for every arm-run.
  const plan = collectComparisonPlan(cuts, 512);
  comparison.plan = plan;
  comparison.unitGranularity = plan.unitGranularity;
  comparison.entries = plan.entries;
  if (comparison.entries.length === 0) {
    endComparison(`Nothing on ${pageLabel()} to compare.`);
    return;
  }

  // Awaited before the first request, or an arm finishing early is recorded with a null id.
  const facts = await fetchComparisonModelFacts();

  // Cuts outer, so the coarsest cut (the entries' unit) fills the report first.
  comparison.runs = [];
  cuts.forEach((cut) => {
    models.forEach((model, index) => {
      if (!modelReadsCut(model, cut, facts.granularities)) return;
      comparison.runs.push({
        cut,
        model,
        // Resolved from /health, else the fallback maps; same vocabulary as wire, storage
        // and log. The letter follows the model, not the run: one model read two ways is
        // one arm answering twice.
        arm: {
          label: comparisonArmLabel(index),
          index,
          modelKey: model,
          modelId: (facts.names && facts.names[model]) || null,
          method: modelMethod(model, facts.methods),
          granularity: modelGranularity(model, facts.granularities),
        },
      });
    });
  });
  comparison.total = comparison.runs.reduce(
    (total, run) =>
      total +
      plan.requests[run.cut].reduce((n, requests) => n + requests.length, 0),
    0,
  );

  const epoch = pageEpoch;
  for (let i = 0; i < comparison.runs.length; i += 1) {
    if (comparison.stopRequested || epoch !== pageEpoch) break;
    const run = comparison.runs[i];
    const { cut, model, arm } = run;
    comparison.runIndex = i + 1;
    renderComparisonNotice(run, false);
    reportComparisonProgress();

    const stats = {
      requests: 0,
      processed: 0,
      changed: 0,
      cached: 0,
      totalTimeMs: 0,
    };
    const startedAt = performance.now();
    // Flat, so the concurrency window spans entries (a sentence-cut entry can be a single
    // unit). Each item keeps its entry index for the fold below.
    const queue = [];
    plan.requests[cut].forEach((requests, entryIndex) => {
      requests.forEach((text) => queue.push({ entryIndex, text }));
    });

    // Requests within one arm-run go out concurrently, at the ordinary page path's width;
    // only runs are serialised. One at a time would starve the backend's BATCH_SIZE-8
    // batching (measured 2026-08-24, 24 sentences through `finetuned`: 455 ms/sentence
    // eight at a time vs. 1648 ms one at a time, 3.6x).
    const answers = await mapWithConcurrency(
      queue,
      MAX_CONCURRENT_SIMPLIFY,
      async (item) => {
        if (comparison.stopRequested || epoch !== pageEpoch) return undefined;
        const data = await requestComparisonUnit(item.text, model, cut);
        stats.requests += 1;
        comparison.done += 1;
        // per answer, so the notice moves while eight are open
        renderComparisonNotice(run, false);
        reportComparisonProgress();
        return data;
      },
    );

    stats.totalTimeMs = Math.round(performance.now() - startedAt);

    // Folded afterwards in page order (completion order is arbitrary), so rows line up
    // with the other arms'.
    const entries = [];
    // null where this arm-run has no complete answer, keeping its column aligned
    const results = new Array(comparison.entries.length).fill(null);
    let at = 0;
    plan.requests[cut].forEach((requests, entryIndex) => {
      const answered = requests.map((text, n) => ({
        text,
        data: answers[at + n],
      }));
      at += requests.length;
      const result = comparisonResult(answered);
      if (!result) return;
      results[entryIndex] = result;
      stats.processed += 1;
      if (result.changed) stats.changed += 1;
      if (result.cached) stats.cached += 1;
      // History records the page's text, not the joined request text (which in the
      // sections cut differs by the terminal punctuation this path adds).
      entries.push(
        historyEntry(
          comparison.entries[entryIndex],
          resultAsHistoryData(result),
          result.changed,
        ),
      );
    });

    // Flushed per arm-run so Reports can show a column while the next runs. The report
    // is written even for an arm that answered nothing, so its column isn't omitted.
    if (entries.length > 0) {
      flushComparisonModel(
        arm,
        cut,
        entries,
        stats,
        comparison.stopRequested ? "stopped" : "completed",
      );
    }
    flushComparisonArm(
      arm,
      cut,
      results,
      stats,
      comparison.stopRequested
        ? "stopped"
        : i === comparison.runs.length - 1
          ? "completed"
          : "running",
    );
  }

  const stopped = comparison.stopRequested;
  renderComparisonNotice(
    comparison.runs[Math.max(comparison.runIndex - 1, 0)] || {
      model: null,
      cut: null,
    },
    true,
    stopped,
  );
  endComparison(null, stopped);
}

// A folded entry result in the shape historyEntry() reads, so the row matches the
// ordinary path's.
function resultAsHistoryData(result) {
  return {
    simplified: result.output,
    model_result: result.modelResult,
    cached: result.cached,
    fallback_reason: result.fallbackReason,
    requestTimeMs: result.requestTimeMs,
    sentence_split: result.sentenceSplit
      ? { count: result.sentenceSplit.count, parts: result.sentenceSplit.parts }
      : null,
  };
}

// Runs `fn` over `items` with at most `width` calls outstanding; results in input order.
// Workers pull from a shared counter, so a slow unit delays only itself (unlike fixed
// groups).
async function mapWithConcurrency(items, width, fn) {
  const out = new Array(items.length);
  let next = 0;
  const worker = async () => {
    for (let i = next++; i < items.length; i = next++) {
      out[i] = await fn(items[i], i);
    }
  };
  await Promise.all(
    Array.from({ length: Math.min(width, items.length) }, worker),
  );
  return out;
}

// One unit against one named model. Resolves to the response body, or null on failure,
// so one failed request doesn't abandon the comparison.
function requestComparisonUnit(text, model, cut) {
  return new Promise((resolve) => {
    const startedAt = performance.now();
    chrome.runtime.sendMessage(
      {
        cmd: "fetchSimplifyWithModel",
        text,
        model,
        page: comparisonSessionId(model, cut),
      },
      (resp) => {
        if (chrome.runtime.lastError || !resp || !resp.ok || !resp.data) {
          console.warn(
            `comparison request failed for '${model}'`,
            chrome.runtime.lastError,
          );
          resolve(null);
          return;
        }
        resolve({
          ...resp.data,
          requestTimeMs: Math.round(performance.now() - startedAt),
        });
      },
    );
  });
}

function endComparison(message, stopped) {
  comparison.running = false;
  stopNoticeTicker();
  releaseRun();
  if (message) showNotice(message, 5000);
  // Runs take minutes on a page that never changes, so completion is announced outside
  // this tab.
  chrome.runtime.sendMessage({
    cmd: "comparisonFinished",
    models: comparison.models.length,
    runs: comparison.runs.length,
    units: comparison.entries.length,
    page: pageLabel(),
    stopped: !!stopped,
    nothingToDo: comparison.entries.length === 0,
    // Which report to open. The group id, not a report id: this side never sees the
    // report, which is keyed by group (shared/report-store.js saveComparisonRun).
    group: comparison.entries.length > 0 ? pageSessionId : null,
  });
}

// --- stopping a run ---
// Unlike toggling off (which reverts), stopping keeps what has been simplified and sends
// nothing further. A navigation ends a run the same way but reverts, since the page is
// gone.
// `runStopped` survives until the next beginSession(), so the following flush records
// the run as stopped.
let runStopped = false;

function runIsActive() {
  // a comparison holds the same lock and Stop, but has no session phases
  return (
    comparisonIsRunning() ||
    session.phase === "preflight" ||
    session.phase === "simplifying"
  );
}

// Ends the current run; returns whether there was one (Stop can be pressed just after
// the last item settled).
function stopCurrentRun() {
  // The comparison loop checks this between requests, so the in-flight answer is still
  // recorded.
  if (comparisonIsRunning()) {
    comparison.stopRequested = true;
    return true;
  }
  if (!runIsActive()) return false;
  runStopped = true;
  // In-flight requests can't be recalled; the epoch bump keeps their answers off the
  // page and also cancels a running preflight.
  pageEpoch += 1;
  abortQueuedSimplifications();
  stopFailureCountdown();
  // Before the notice: renderProgressNotice() won't draw over an idle session, so the
  // ticker can't overwrite it.
  endSession();
  flushHistoryBuffer();
  // written text stays, so the badge still says simplified
  isSimplified = isSimplified || changedCount > 0;
  notifyBadge(isSimplified);
  showStoppedNotice();
  return true;
}

// on icon click, before the preflight
function beginSession() {
  session.phase = "preflight";
  runStopped = false;
  session.selection = null;
  session.clickedAt = performance.now();
  session.preflightMs = 0;
  session.startedAt = 0;
  session.settledAt = 0;
  session.requests = 0;
  session.requestMsTotal = 0;
  session.cachedRequests = 0;
  startNoticeTicker();
  startNavWatcher();
}

function endSession() {
  session.phase = "idle";
  stopNoticeTicker();
  stopNavWatcher();
  // every way a run ends passes through here, so the lock is released here
  releaseRun();
}

// Asks the service worker for the selected model/audience without waiting for the
// preflight. Best-effort; the preflight overwrites it either way.
function requestSelectionInfo() {
  chrome.runtime.sendMessage({ cmd: "selectionInfo" }, (resp) => {
    if (chrome.runtime.lastError || !resp) return;
    // the preflight's (health-backed, more precise) answer wins if it arrived first
    if (session.selection) return;
    session.selection = resp;
    renderProgressNotice();
  });
}

// One completed backend round trip, timed around the message to the service worker:
// includes the backend's batching/queueing wait, unlike the isolated forward-pass time
// the Backend status vitality check reports against an idle backend.
function recordRequestTiming(elapsedMs, cached) {
  session.requests += 1;
  session.requestMsTotal += elapsedMs;
  if (cached) session.cachedRequests += 1;
}

// Identifies this page for the History page (extension/history/); all batches of a page
// append to the same entry. Re-minted on a same-document navigation
// (checkForNavigation), since an SPA route is a different page to the reader.
let pageSessionId = crypto.randomUUID();
// Whether anything has been logged under the current pageSessionId (re-minted with it).
// flushHistoryBuffer() uses it to tell a repeat run's all-cached batch from the page's
// only record.
let pageSessionRecorded = false;
// Entries awaiting flushHistoryBuffer(), batched per simplifyPage()/
// processObservedElements() pass. Shape: historyEntry().
let historyBuffer = [];
// Entries with a fallbackReason (model output rejected by a guard). Cumulative across
// the run's batches, since the status it decides is per page. Reset in simplifyPage().
let fallbackCount = 0;

// One log entry for one backend round trip: input -> modelResult -> output.
// `modelResult` is what the model returned even if a guard rejected it; `output` is what
// the page shows (the input, when a guard fired, with `fallbackReason` naming it).
// Model, audience, granularity and method are fixed per run and live on the session
// (flushHistoryBuffer).
// `coverage` (document mode): how many of the section's units the answer was written
// into; see unwrittenUnitCount.
function historyEntry(input, data, changed, coverage = null) {
  const entry = {
    input,
    // /simplify always sends this ("" when there was no generation, e.g. timeout). The
    // fallback covers partial payloads from the comparison path and test harness.
    modelResult:
      typeof data.model_result === "string"
        ? data.model_result
        : data.simplified,
    output: data.simplified,
    changed,
    cached: !!data.cached,
    fallbackReason: data.fallback_reason || null,
    requestTimeMs:
      typeof data.requestTimeMs === "number" ? data.requestTimeMs : null,
  };
  if (entry.fallbackReason) fallbackCount += 1;
  // only where the backend actually split the input (absent = never split)
  const split = data.sentence_split;
  if (split && Array.isArray(split.parts) && split.count > 1) {
    entry.sentenceSplit = { count: split.count, parts: split.parts };
  }
  // only for multi-unit sections where some unit was left unwritten
  if (coverage && coverage.units > 1 && coverage.written < coverage.units) {
    entry.coverage = { units: coverage.units, written: coverage.written };
  }
  return entry;
}

// Per-run facts, recorded once per page. Counts are the same tally the notice reports
// (renderProgressNotice), so History and the notice can't disagree.
function sessionSummary() {
  const sel = session.selection || {};
  const summary = {
    // Not converted: /health, storage and the log share shared/model-labels.js's
    // vocabulary. Fallbacks cover a run logged before the preflight answered.
    method: sel.method || modelMethod(sel.model),
    // picker key ("finetuned"): History's "Compare with different model" needs it to
    // build a request; an id alone can't be sent to the backend
    modelKey: sel.model || null,
    modelId: sel.modelId || null,
    granularity: sel.granularity || modelGranularity(sel.model),
    totalTimeMs: Math.round(
      (session.settledAt || performance.now()) - session.clickedAt,
    ),
    status: runStatus(),
    stats: {
      totalItems: foundTotal,
      skippedItems: skippedCount,
      processedItems: sentCount,
      simplifiedItems: changedCount,
      unchangedItems: Math.max(sentCount - changedCount, 0),
      cachedResponses: session.cachedRequests,
    },
  };
  // document mode only, and only if nonzero (always zero on the sentence path)
  if (unwrittenUnitCount > 0) summary.stats.unwrittenUnits = unwrittenUnitCount;
  // Only where it applies: seq2seq checkpoints ignore the audience and the backend
  // normalizes it away.
  if (sel.audienceApplies) summary.audience = sel.audience;
  return summary;
}

// Shown as one coloured dot on the History page. "Warnings" (a guard rejected at least
// one generation) is separate from "completed" because a page of fallbacks otherwise
// looks like a page of already-simple text.
function runStatus() {
  if (lastSimplifyError) return "failed";
  // before the fallback check: "stopped" matters more, its counts cover part of the page
  if (runStopped) return "stopped";
  return fallbackCount > 0 ? "completed_with_warnings" : "completed";
}

// A batch worth logging: not empty, and not a re-run of a page already recorded.
// Re-toggling a page serves every answer from the cache and would append verbatim
// duplicates (pageSessionId belongs to the page load, not the run). So an all-cached
// batch is dropped unless nothing has been recorded for this page yet (e.g. a route
// change whose text was all cached). Mixed batches are kept in full, cached entries
// included.
function isWorthRecording(entries) {
  if (entries.length === 0) return false;
  if (!pageSessionRecorded) return true;
  return entries.some((entry) => !entry.cached);
}

// `page` is passed when flushing after a navigation, when location.href is already
// the new URL.
function flushHistoryBuffer(page = null) {
  if (historyBuffer.length === 0) return;
  const entries = historyBuffer.splice(0);
  // dropped, not kept: kept entries would attach to a later, different run's batch
  if (!isWorthRecording(entries)) return;
  pageSessionRecorded = true;
  chrome.runtime.sendMessage({
    cmd: "recordHistory",
    page: page || {
      sessionId: pageSessionId,
      url: location.href,
      title: document.title,
      timestamp: Date.now(),
    },
    // recomputed per flush: later observer batches extend the same run
    session: sessionSummary(),
    entries,
  });
}

// Google-Translate-style DOM coverage. BLOCK_TAGS: a nested block descendant
// disqualifies an ancestor, so only the innermost is sent (<p> in <div> excludes the
// <div>). INLINE_LEAF_TAGS never disqualify an ancestor but become their own unit when
// no selected block leaf covers them (a lone `<button>Submit</button>`,
// `<nav><a>Home</a></nav>`). Formatting tags (strong/b/em/i/small/mark/del/ins) are in
// neither list: their text is part of the ancestor leaf's plain-text extraction
// (extractInnerText); the formatting is lost.
const BLOCK_TAGS = [
  "P",
  "H1",
  "H2",
  "H3",
  "H4",
  "H5",
  "H6",
  "LI",
  "TD",
  "TH",
  "BLOCKQUOTE",
  "FIGCAPTION",
  "SUMMARY",
  "HEADER",
  "FOOTER",
  "MAIN",
  "SECTION",
  "ARTICLE",
  "ASIDE",
  "DIV",
  "TABLE",
  "TR",
  "UL",
  "OL",
];
const INLINE_LEAF_TAGS = ["A", "BUTTON", "LABEL", "OPTION", "SPAN", "Q"];
const BLOCK_TAG_SET = new Set(BLOCK_TAGS);
const INLINE_LEAF_TAG_SET = new Set(INLINE_LEAF_TAGS);
const CANDIDATE_TAGS = BLOCK_TAGS.concat(INLINE_LEAF_TAGS);
const CANDIDATE_SELECTOR = CANDIDATE_TAGS.join(",").toLowerCase();
const PLACEHOLDER_SELECTOR = "input[placeholder], textarea[placeholder]";
// The extension's own injected UI, filtered out before candidate gathering. isOptedOut()
// would exclude the `translate="no"` notice only after counting it as found, which
// re-renders the notice and re-triggers the observer in an endless loop.
// PICKER_HOST_ID's subtree is in a shadow root, but its host DIV sits in document.body.
const OWN_UI_IDS = new Set([
  "simplify-notice",
  "simplifier-style",
  "simplify-model-picker",
  "simplify-compare-picker",
]);

// Clear a badge left showing "ON" from the previous page load.
notifyBadge(false);

(function injectStyles() {
  const styleEl = document.createElement("style");
  styleEl.id = "simplifier-style";
  styleEl.textContent = `
    .simplified {
      background-color: #ffffcc;
      transition: background-color 0.3s ease;
    }
    /* The notice box. Everything inside it is styled through this id rather
       than by class alone, both to outrank the host page's own rules (a page
       styling ".sn-title" or "code" would otherwise reach in) and because the
       notice is the only place these classes exist. Fonts, sizes and colors are
       all stated explicitly: the box is injected into an arbitrary page, so
       anything left to inherit inherits from that page. */
    #simplify-notice {
      position: fixed;
      top: 10px;
      right: 10px;
      width: 320px;
      max-width: calc(100vw - 20px);
      box-sizing: border-box;
      overflow-wrap: break-word;
      white-space: pre-line;
      line-height: 1.45;
      background: rgba(20,20,20,0.92);
      color: #fff;
      padding: 12px 14px;
      border-radius: 8px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.35);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      font-size: 13px;
      text-align: left;
      /* over anything a host page stacks, but deliberately under the model picker's
         2147483647 (shared/picker.js): the picker is a viewport takeover, and a
         progress box punching through its backdrop would read as page content that
         had escaped it. */
      z-index: 999999;
    }
    /* which page this is about. Truncated rather than wrapped: a long <title>
       would otherwise push the numbers -- the part that changes -- out of view. */
    #simplify-notice .sn-title {
      display: block;
      margin-bottom: 8px;
      font-size: 15px;
      font-weight: 600;
      line-height: 1.3;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    #simplify-notice .sn-bar-row {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    #simplify-notice .sn-bar {
      flex: 1;
      height: 7px;
      border-radius: 99px;
      background: rgba(255,255,255,0.18);
      overflow: hidden;
    }
    #simplify-notice .sn-bar-fill {
      height: 100%;
      width: 0;
      border-radius: 99px;
      background: #4688f1;
      transition: width 0.25s ease;
    }
    /* the preflight has no items to count yet, so its bar reports "something is
       happening" instead of how far along it is -- see renderProgressNotice(). */
    #simplify-notice .sn-bar-indeterminate {
      width: 40%;
      animation: simplify-notice-sweep 1.3s ease-in-out infinite;
    }
    @keyframes simplify-notice-sweep {
      from { transform: translateX(-100%); }
      to { transform: translateX(250%); }
    }
    /* tabular figures so the count doesn't jitter sideways as it climbs */
    #simplify-notice .sn-count {
      font-weight: 600;
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }
    #simplify-notice .sn-processed {
      white-space: normal;
    }
    #simplify-notice .sn-outcome {
      margin-top: 6px;
      white-space: normal;
    }
    #simplify-notice .sn-emph {
      font-weight: 600;
    }
    #simplify-notice .sn-muted {
      font-size: 12.5px;
      color: rgba(255,255,255,0.62);
      white-space: normal;
    }
    /* one line, always: three figures that are only comparable read together */
    #simplify-notice .sn-speed {
      margin-top: 7px;
      font-size: 12px;
      color: rgba(255,255,255,0.62);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    #simplify-notice .sn-speed + .sn-speed {
      margin-top: 0;
    }
    #simplify-notice .sn-meta {
      font-size: 11.5px;
      color: rgba(255,255,255,0.55);
      white-space: normal;
    }
    #simplify-notice .sn-gap {
      margin-top: 7px;
    }
    /* the exact model id: a string to copy, not prose */
    #simplify-notice code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
      overflow-wrap: anywhere;
      color: #ffffff;
      background: rgba(255, 255, 255, 0.08);
      padding: 1px 4px;
      border-radius: 4px;
    }
    #simplify-notice .sn-status {
      margin-top: 7px;
      font-size: 11.5px;
      font-style: italic;
      color: rgba(255,255,255,0.55);
      white-space: normal;
    }
    /* The one thing in this box that is pressable. It sits at the bottom, below the
       figures, because it is what you reach for after reading them -- and it is a real
       button rather than a link, since it acts on this run rather than going anywhere.
       Every property is stated: a host page's own button rules would otherwise reach
       a button the page never wrote. */
    #simplify-notice .sn-actions {
      display: flex;
      gap: 8px;
      margin-top: 10px;
    }
    #simplify-notice .sn-btn {
      flex: none;
      margin: 0;
      padding: 6px 12px;
      border: 1px solid rgba(255,255,255,0.28);
      border-radius: 6px;
      background: rgba(255,255,255,0.08);
      color: #fff;
      font-family: inherit;
      font-size: 12px;
      font-weight: 600;
      line-height: 1.2;
      letter-spacing: 0;
      text-transform: none;
      cursor: pointer;
    }
    #simplify-notice .sn-btn:hover { background: rgba(255,255,255,0.16); }
    #simplify-notice .sn-btn:focus-visible { outline: 2px solid #4688f1; outline-offset: 1px; }
  `;
  document.head.appendChild(styleEl);
})();

// Page-wide cap on in-flight simplification requests. Call sites fan out with
// Promise.all (one request per chunk), so a long page could open hundreds at once; the
// prompted LLM (roughly one sentence per second or two, single stream) would then look
// hung for minutes. 8 exactly fills the backend's BATCH_SIZE for the seq2seq path.
const MAX_CONCURRENT_SIMPLIFY = 8;
let inFlightSimplify = 0;
const simplifyQueue = [];

function pumpSimplifyQueue() {
  while (
    inFlightSimplify < MAX_CONCURRENT_SIMPLIFY &&
    simplifyQueue.length > 0
  ) {
    const run = simplifyQueue.shift();
    inFlightSimplify++;
    run().finally(() => {
      inFlightSimplify--;
      pumpSimplifyQueue();
    });
  }
}

// Delegated to the service worker: a content-script fetch is attributed to the page's
// origin, which trips Chrome's Local Network Access gate for 127.0.0.1.
// Every path funnels through here, so this is the single rate-limit choke point.
function fetchSimplify(text) {
  return new Promise((resolve, reject) => {
    const job = () => sendSimplifyRequest(text).then(resolve, reject);
    // Reject rather than drop (which would leave Promise.all pending forever); the marked
    // error lets call sites tell navigation from backend failure.
    job.cancel = () => reject(navigationAbort());
    simplifyQueue.push(job);
    pumpSimplifyQueue();
  });
}

// Rejection for requests still queued when the page navigated; not a failure.
function navigationAbort() {
  const err = new Error("page navigated before this request ran");
  err.simplifyNavigationAbort = true;
  return err;
}

function isNavigationAbort(e) {
  return Boolean(e && e.simplifyNavigationAbort);
}

function abortQueuedSimplifications() {
  simplifyQueue.splice(0).forEach((job) => job.cancel());
}

function sendSimplifyRequest(text) {
  return new Promise((resolve, reject) => {
    // started here, not in fetchSimplify(), so local queue wait isn't counted as
    // backend time
    const startedAt = performance.now();
    // `page` lets the backend attribute cache entries to this page load and drop them
    // when it leaves the History log's 20-page window. Read at send time: a navigation
    // re-mints the id and aborts everything still queued.
    chrome.runtime.sendMessage(
      { cmd: "fetchSimplify", text, page: pageSessionId },
      (resp) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
          return;
        }
        if (!resp || !resp.ok) {
          reject(new Error((resp && resp.error) || "fetchSimplify failed"));
          return;
        }
        // only successful round trips are timed
        const elapsedMs = performance.now() - startedAt;
        recordRequestTiming(elapsedMs, !!(resp.data && resp.data.cached));
        // The model is only known to background.js (read from storage there). Audience:
        // what the backend applied, else what was requested. `requestTimeMs` is only
        // measured here and is logged per History entry.
        resolve({
          ...resp.data,
          model: resp.model,
          audience: resp.data.audience || resp.audience,
          requestTimeMs: Math.round(elapsedMs),
        });
      },
    );
  });
}

// Preflight, once per toggle-on: backend reachable, model loaded, model returns output
// for a real request. Resolves to {ok:true} or {ok:false, stage, error}; stages are
// defined in background.js checkBackendHealth.
function checkBackend() {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ cmd: "healthCheck" }, (resp) => {
      if (chrome.runtime.lastError) {
        resolve({
          ok: false,
          stage: "reachable",
          error: chrome.runtime.lastError.message,
        });
        return;
      }
      resolve(
        resp || {
          ok: false,
          stage: "reachable",
          error: "no response from background script",
        },
      );
    });
  });
}

// Google Translate's opt-out convention (`translate="no"`, `.notranslate`), applied to
// the element and everything inside it. `aria-hidden="true"` marks decorative content
// such as icon glyphs and counters (observed: a shopping-bag badge's "0"/"+" came back
// as "2.0"/"Other websites").
function isOptedOut(el) {
  return (
    el.closest('[translate="no"], .notranslate, [aria-hidden="true"]') !== null
  );
}

// Short phrases (nav labels, "By", single-word buttons) rarely benefit and are the most
// likely to trip the backend's hallucination guard (backend/main.py
// _looks_like_hallucination).
const MIN_WORDS_TO_SIMPLIFY = 4;
function hasEnoughWords(text) {
  return wordCount(text) >= MIN_WORDS_TO_SIMPLIFY;
}

// The one length measure used throughout: the send/skip threshold, document mode's token
// estimate, and each unit's share of a section's answer.
function wordCount(text) {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

// Digits/symbols only (counters, prices, glyphs) can pass the word count ("2 / 5") and
// make the model hallucinate; require at least one letter.
function hasLetters(text) {
  return /\p{L}/u.test(text);
}

// Picks the innermost text-bearing units from CANDIDATE_SELECTOR matches, so nested
// containers aren't sent twice.
// Walks ancestor chains rather than querying descendant subtrees: cost is bounded by
// nesting depth, not by how much markup a leaf contains (subtree queries were slow on
// markup-heavy pages).
function computeLeafCandidates(elements) {
  const elementSet = new Set(elements);
  const nonLeafBlocks = new Set();

  for (const el of elements) {
    if (!BLOCK_TAG_SET.has(el.tagName)) continue;
    // a block candidate containing this one is not a leaf
    for (
      let ancestor = el.parentElement;
      ancestor;
      ancestor = ancestor.parentElement
    ) {
      if (BLOCK_TAG_SET.has(ancestor.tagName) && elementSet.has(ancestor)) {
        nonLeafBlocks.add(ancestor);
      }
    }
  }

  const leafBlocks = elements.filter(
    (el) => BLOCK_TAG_SET.has(el.tagName) && !nonLeafBlocks.has(el),
  );
  const leafBlockSet = new Set(leafBlocks);

  const orphanInline = elements.filter((el) => {
    if (!INLINE_LEAF_TAG_SET.has(el.tagName)) return false;
    for (
      let ancestor = el.parentElement;
      ancestor;
      ancestor = ancestor.parentElement
    ) {
      if (leafBlockSet.has(ancestor)) return false; // already covered by a leaf block
    }
    return true;
  });

  return leafBlocks.concat(orphanInline);
}

// <sup> is stripped mainly for citation markers (Wikipedia's <sup class="reference">),
// so "Chris Crawford.[4][5][6]" doesn't send "[4]" or "456". The opt-out selectors
// mirror isOptedOut() for nested opt-outs (e.g. an icon-only inline button) inside a
// block that isn't opted out.
const STRIP_FROM_TEXT_SELECTOR =
  'sup, script, style, [translate="no"], .notranslate, [aria-hidden="true"]';

// The part of that list a document request may read as context but never rewrite
// ("pinned"). An opt-out means "don't change", not "invisible". Stripping them outright
// left sections like
//
//     "Static test pages ... for each page.  : baseline toggle/revert behaviour.
//      : headings, lists, tables, semantic sectioning tags, quotes.  : ..."
//
// (every item's subject was an opted-out link), and the document model invented text
// for such input (four sentences about an "international association for the
// advancement of science and technology").
// <sup>, <script>, <style> and aria-hidden stay stripped: not prose, or decorative
// glyphs and duplicated labels.
const PINNABLE_SELECTOR = '[translate="no"], .notranslate';

// Leaf elements are sent as plain-prose chunks: descendant tags are flattened for the
// request, cut at each <br>. The tags stay in the page; collectChunkNodes keeps each
// chunk's text nodes and writeChunkText() writes only over those. Splitting at every
// inline tag instead sent context-free fragments like ") is a nonprofit " and garbled
// results.
//
// <br> must be honored here, not left to the backend's sentence segmenter: it carries
// no punctuation and contributes nothing to textContent, so "Founded: 1988<br>
// Headquarters: San Jose" would become one run-on (and "one<br>two" becomes "onetwo").
// Block-level breaks are already separate leaves (computeLeafCandidates).
//
// Empty chunks are kept: they preserve <br><br> gaps and trailing <br>s and keep chunks
// index-aligned with their nodes.
//
// A blank line inside a text node is the same kind of break (visible under
// `white-space: pre-wrap`/<pre>; e.g. demo/08-html-checker.html's output pane). Matched
// with the following whitespace, so the break stays at the end of the paragraph it
// closes, where writeChunkText's trailing-whitespace restore puts it back.
const BLANK_LINE = /\n[ \t]*\n[ \t\n]*/g;
const BLANK_LINE_AT_END = /\n[ \t]*\n[ \t\n]*$/;

// Splits text nodes at blank lines so each break is a node end, i.e. a chunk boundary.
// Serialization is unchanged, so the `originalHtml` revert is unaffected. The "\n" guard
// avoids a DOM write on every leaf. Idempotent: a break at a node's end is not split.
function splitAtBlankLines(el) {
  const texts = [];
  const collect = (node) => {
    for (const child of node.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) texts.push(child);
      else if (child.nodeType === Node.ELEMENT_NODE) collect(child);
    }
  };
  collect(el);
  for (const node of texts) {
    if (!node.data.includes("\n")) continue;
    const offsets = [];
    BLANK_LINE.lastIndex = 0;
    let match;
    while ((match = BLANK_LINE.exec(node.data)) !== null) {
      const end = match.index + match[0].length;
      if (end < node.data.length) offsets.push(end);
    }
    // from the back, so earlier offsets stay valid
    offsets.reverse().forEach((offset) => node.splitText(offset));
  }
}

// Returns, per chunk, its text nodes in document order. Callers read the untrimmed text
// directly: it is the only place the chunk-ending break survives.
// `pinned` (a Set, document path only; see PINNABLE_SELECTOR): opted-out subtrees are
// walked, and their text nodes are both collected and added to the Set as readable but
// not writable.
function collectChunkNodes(el, pinned = null) {
  splitAtBlankLines(el);
  const chunks = [];
  let current = [];
  const walk = (node, isPinned = false) => {
    for (const child of node.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) {
        current.push(child);
        if (isPinned) pinned.add(child);
        // blank line at node end (splitAtBlankLines): a boundary, like <br>
        if (BLANK_LINE_AT_END.test(child.data)) {
          chunks.push(current);
          current = [];
        }
      } else if (child.nodeType === Node.ELEMENT_NODE) {
        if (child.tagName === "BR") {
          chunks.push(current);
          current = [];
        } else if (ownUnits.has(child)) {
          // Its own unit, so not part of this one (not descended into, not even for
          // <br>). Otherwise `<label>…in the toolbar<button>Clear</button></label>`
          // sent "…in the toolbarClear", and the button lost its text.
        } else if (pinned && !isPinned && child.matches(PINNABLE_SELECTOR)) {
          // document-path opt-out: read as context; every text node inside is pinned
          walk(child, true);
        } else if (isPinned || !child.matches(STRIP_FROM_TEXT_SELECTOR)) {
          // Inside a pinned subtree, or ordinary. Stripped subtrees contribute nothing,
          // including any <br> inside them.
          walk(child, isPinned);
        }
      }
    }
  };
  walk(el);
  chunks.push(current);
  return chunks;
}

// An element's text with chunk breaks restored ("\n" for <br>, "\n\n" for a blank
// line), for the tooltip and History log. `raws` are the untrimmed chunks, where the
// break survives. Both paths record originals through here.
function joinChunkOriginals(chunks, raws) {
  return chunks.reduce(
    (text, chunk, i) =>
      i === 0
        ? chunk
        : text + (BLANK_LINE_AT_END.test(raws[i - 1]) ? "\n\n" : "\n") + chunk,
    "",
  );
}

// Real prose, not a bare URL or an empty <br><br> gap. Failing chunks are left as they
// were and keep their place.
function isSimplifiableChunk(text) {
  return chunkSkipReason(text) === null;
}

// Chunk-level counterpart of elementSkipReason().
function chunkSkipReason(text) {
  if (!text) return "empty-chunk";
  // plain-text URLs (anchor hrefs never reach the chunk text)
  if (/https?:\/\//.test(text)) return "bare-url";
  if (!hasEnoughWords(text)) return "under-min-words";
  if (!hasLetters(text)) return "no-letters";
  return null;
}

// Writing back: only the chunk's own text nodes are overwritten. Rebuilding the element
// as one text node (el.replaceChildren()) deleted everything not sent, including links
// and skipped subtrees: `<a>07 — Document sections</a>: heading-delimited sections.`
// lost its link.
//
// The output is spread across nodes by anchoring on fragments the model left alone:
// each node's text is searched for in the output, in order, from the previous match.
// Matching nodes keep their words (a surviving <b>/<a> keeps its text); text between
// matches goes to the first unmatched node before it (which restores " " in
// whitespace-only nodes between inline elements); leftovers join the last node. If
// nothing anchors, the first node takes everything and the rest go empty.
//
// Written through `.data`, never innerHTML: model output could contain a stray "<".
//
// Leading punctuation that joins a chunk to what precedes it (the ": " in `<a>01 — Basic
// paragraphs</a>: baseline toggle/revert behaviour.`, nav/byline dashes and pipes).
// Deliberately excludes opening quotes/brackets: re-adding an unmatched `"` or `(` reads
// as a typo.
const CHUNK_SEPARATOR = /^[\s:;,|–—-]+/;

// Prefix restored before the answer so it doesn't fuse with what precedes it.
// Leading whitespace was trimmed before sending and is always restored. A leading
// separator is sent but often dropped by the model: `": baseline toggle/revert
// behaviour."` -> `"Baseline toggle/revert function."` gave `<a>01 — Basic
// paragraphs</a>Baseline toggle/revert function.` This labelled-list shape is common
// (demo/index.html, nav and definition lists).
// Restored only if the answer lacks it, compared on punctuation only (":" vs ": " is
// not a loss).
function openingSeparator(whole, simplified) {
  const whitespace = whole.match(/^\s*/)[0];
  const separator = (whole.match(CHUNK_SEPARATOR) || [""])[0];
  // separator only: no sentence to precede
  if (separator.length === whole.length) return whitespace;
  const punctuation = separator.replace(/\s+/g, "");
  if (!punctuation || simplified.trim().startsWith(punctuation))
    return whitespace;
  return separator;
}

// `pinned`: the Set from collectChunkNodes(); those nodes are not written (see
// PINNABLE_SELECTOR) but still take part in alignment, which removes a pinned label's
// words from the answer so they aren't written a second time.
function writeChunkText(nodes, simplified, root, pinned = null) {
  if (nodes.length === 0) return;
  const isPinned = nodes.map((node) => (pinned ? pinned.has(node) : false));
  const first = isPinned.indexOf(false);
  // all pinned: nothing writable (and it shouldn't have been sent)
  if (first === -1) return;
  const last = isPinned.lastIndexOf(false);
  const whole = nodes.map((n) => n.data).join("");
  // restore the outer spacing trimmed before sending, or gaps between neighbouring
  // leaves close up
  const leading = whole.match(/^\s*/)[0];
  const trailing = whole.slice(leading.length).match(/\s*$/)[0];
  const parts = alignToNodes(
    nodes.map((n) => n.data.trim()),
    simplified.trim(),
    // Nodes inside an inline element (<a>, <b>) get a looser second anchoring attempt,
    // since an empty link is lost content. Direct children stay on exact matching, where
    // a partial hit would split a sentence at an arbitrary word.
    nodes.map((n) => n.parentNode !== root),
    isPinned,
  );
  // Only if the chunk's edge node is writable; a pinned edge node keeps the page's
  // spacing, and restoring it again would duplicate it mid-chunk.
  if (first === 0) parts[0] = openingSeparator(whole, simplified) + parts[0];
  if (last === nodes.length - 1) parts[last] += trailing;
  nodes.forEach((node, i) => {
    if (isPinned[i]) return;
    node.data = parts[i];
  });
}

// Splits `output` into one string per node, anchored on node texts that survive
// verbatim (see writeChunkText()). `nested[i]`: may anchor on part of its text.
// `pinned[i]`: never written.
function alignToNodes(texts, output, nested = [], pinned = []) {
  const parts = new Array(texts.length).fill("");
  // nodes since the last match, waiting for the text before the next one
  const unmatched = [];
  let cursor = 0;

  texts.forEach((text, i) => {
    if (pinned[i]) {
      // Keeps its own text; anchoring only advances the cursor past the label's words.
      // Never joins `unmatched`: text given to it would never reach the page.
      const match =
        exactMatch(text, output, cursor) || partialMatch(text, output, cursor);
      if (!match) return;
      const gap = output.slice(cursor, match.at);
      if (gap) {
        const target =
          unmatched.length > 0
            ? gapTarget(unmatched, nested, pinned)
            : writableBefore(i, pinned);
        if (target !== -1) parts[target] += gap;
        unmatched.length = 0;
      }
      cursor = match.at + match.text.length;
      return;
    }
    // 1-2 character fragments ("a", "—") match coincidentally almost anywhere.
    const match =
      (text.length >= 3 && exactMatch(text, output, cursor)) ||
      (nested[i] ? partialMatch(text, output, cursor) : null);
    if (!match) {
      unmatched.push(i);
      return;
    }
    const { at } = match;
    text = match.text;
    const gap = output.slice(cursor, at);
    if (unmatched.length > 0) {
      parts[gapTarget(unmatched, nested, pinned)] = gap;
      unmatched.length = 0;
    } else if (i > 0 && !pinned[i - 1] && !(nested[i - 1] && !nested[i])) {
      // to the previous node, outside this one's element (the space between "The"
      // and "<b>quick brown fox</b>")
      parts[i - 1] += gap;
    } else {
      // Previous node is pinned, or nested while this one isn't (the space after
      // "<mark>overall comprehensibility</mark>" belongs outside the mark), or there
      // is no previous node.
      parts[i] = gap;
    }
    parts[i] += text;
    cursor = at + text.length;
  });

  const leftover = output.slice(cursor);
  if (leftover) {
    if (unmatched.length > 0)
      parts[gapTarget(unmatched, nested, pinned)] = leftover;
    else {
      // last writable node (not the last node if the chunk ends on a pinned label)
      const end = writableBefore(texts.length - 1, pinned);
      if (end !== -1) parts[end] += leftover;
    }
  }
  return parts;
}

// Which waiting node gets a stretch of rewritten output: the first that is neither
// pinned nor nested, since putting new text inside a <b>/<a> would extend markup over a
// phrase the author never marked. A nested node takes it only if all are nested.
function gapTarget(candidates, nested, pinned = []) {
  const writable = candidates.filter((i) => !pinned[i]);
  const usable = writable.length > 0 ? writable : candidates;
  const plain = usable.find((i) => !nested[i]);
  return plain === undefined ? usable[0] : plain;
}

// Nearest non-pinned node at or before `from`, else after it, else -1.
function writableBefore(from, pinned) {
  for (let i = from; i >= 0; i--) if (!pinned[i]) return i;
  for (let i = from + 1; i < pinned.length; i++) if (!pinned[i]) return i;
  return -1;
}

// Verbatim first, then with whitespace runs collapsed: a source line break
// ("…checklist\nfor each page.") otherwise anchors nowhere in a one-line answer (this
// left a `<code>` label empty).
function exactMatch(text, output, cursor) {
  const at = output.indexOf(text, cursor);
  if (at !== -1) return { at, text };
  const flat = text.replace(/\s+/g, " ");
  if (flat === text) return null;
  const flatAt = output.indexOf(flat, cursor);
  return flatAt === -1 ? null : { at: flatAt, text: flat };
}

// Longest run of consecutive words from `text` found verbatim in `output`, so a
// half-rewritten link label keeps what survived: "Computer Game Developers Conference"
// vs. "…the Computer Game Conference, started in 1988" keeps "Computer Game";
// "Conference" stays outside the link.
// Runs need at least two words (common single words match anywhere); one word only as
// the last resort below. Cheap because writeChunkText()'s `nested` gate limits it to
// short inline labels.
const MAX_PARTIAL_MATCH_WORDS = 12;
function partialMatch(text, output, cursor) {
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length < 2 || words.length > MAX_PARTIAL_MATCH_WORDS) return null;
  for (let length = words.length - 1; length >= 2; length--) {
    for (let start = 0; start + length <= words.length; start++) {
      const phrase = words.slice(start, start + length).join(" ");
      const at = output.indexOf(phrase, cursor);
      if (at !== -1) return { at, text: phrase };
    }
  }
  // Last resort: one long word occurring exactly once in the output (a repeated word
  // could anchor on the wrong occurrence).
  for (const word of words
    .filter((w) => w.length >= 5)
    .sort((a, b) => b.length - a.length)) {
    const at = output.indexOf(word);
    if (at >= cursor && at === output.lastIndexOf(word))
      return { at, text: word };
  }
  return null;
}

function simplifyElement(el) {
  if (el.dataset.originalHtml || !shouldSimplifyElement(el)) {
    resolveItem(
      el,
      "skipped",
      false,
      el.dataset.originalHtml ? "already-simplified" : elementSkipReason(el),
    );
    return Promise.resolve({ changed: false, cached: false });
  }

  const origHtml = el.innerHTML;
  // One walk for both the chunk texts (sent) and their nodes (written back), so the two
  // can't disagree.
  // Only the sentence path keeps the original whitespace layout; document requests
  // collapse to plain prose.
  const chunkNodes = collectChunkNodes(el);
  const rawChunks = chunkNodes.map((nodes) =>
    nodes.map((n) => n.data).join(""),
  );
  const chunks = rawChunks.map((text) => text.trim());
  // Ineligible chunks are tallied even if the element is sent: a per-element count
  // can't see prose left behind inside it.
  const chunkReasons = chunks.map(chunkSkipReason);
  const eligible = chunkReasons.map((reason) => reason === null);
  chunkReasons.forEach((reason, i) =>
    noteSkip(chunkSkipReasons, reason, chunks[i]),
  );
  const origText = joinChunkOriginals(chunks, rawChunks);

  if (!eligible.some(Boolean)) {
    resolveItem(el, "skipped", false, "no-simplifiable-chunk");
    return Promise.resolve({ changed: false, cached: false });
  }

  // positional: ineligible chunks keep their text, so <br> positions survive
  const results = new Array(chunks.length);
  let cachedAll = true;
  // answers arriving after a navigation are discarded (see the navigation section)
  const epoch = pageEpoch;

  const promises = chunks.map((chunk, i) => {
    if (!eligible[i]) {
      results[i] = chunk;
      return Promise.resolve();
    }
    return fetchSimplify(chunk).then((data) => {
      const norm = (str) => str.replace(/\s+/g, " ").trim();
      results[i] = data.simplified;
      if (!data.cached) cachedAll = false;
      historyBuffer.push(
        historyEntry(chunk, data, norm(data.simplified) !== norm(chunk)),
      );
    });
  });

  return Promise.all(promises)
    .then(() => {
      if (epoch !== pageEpoch) return { changed: false, cached: cachedAll };
      const norm = (str) => str.replace(/\s+/g, " ").trim();
      const changed = results.some(
        (r, i) => eligible[i] && norm(r) !== norm(chunks[i]),
      );
      if (changed) {
        // in place: only the sent chunks' text nodes are rewritten, and only if changed
        results.forEach((text, i) => {
          if (eligible[i] && norm(text) !== norm(chunks[i]))
            writeChunkText(chunkNodes[i], text, el);
        });
        el.dataset.originalHtml = origHtml;
        el.dataset.originalText = origText;
        rememberTitle(el);
        rememberMissingAttributes(el);
        el.title = origText;
        el.classList.add("simplified");
        adjustBackgroundForContrast(el);
      }
      resolveItem(el, "sent", changed);
      return { changed, cached: cachedAll };
    })
    .catch((e) => {
      // navigation abort: not a failure, and the run has already ended
      if (isNavigationAbort(e)) return { changed: false, cached: false };
      console.error("simplification failed", e);
      lastSimplifyError = e.message || String(e);
      resolveItem(el, "sent", false); // still counts as sent — it just failed
      return { changed: false, cached: false };
    });
}

// <input>/<textarea> placeholder: an attribute swap, not a text-node write.
function simplifyPlaceholder(el) {
  const orig = el.getAttribute("placeholder") || "";
  if (
    el.dataset.originalPlaceholder !== undefined ||
    !shouldSimplifyPlaceholder(el) ||
    !hasEnoughWords(orig) ||
    !hasLetters(orig)
  ) {
    // same reasons as text skips, prefixed "placeholder:"
    const reason =
      el.dataset.originalPlaceholder !== undefined
        ? "already-simplified"
        : !shouldSimplifyPlaceholder(el)
          ? isOptedOut(el)
            ? "opted-out"
            : "placeholder-empty"
          : chunkSkipReason(orig.trim());
    resolveItem(el, "skipped", false, `placeholder:${reason}`, orig);
    return Promise.resolve({ changed: false, cached: false });
  }
  const epoch = pageEpoch;
  return fetchSimplify(orig)
    .then((data) => {
      if (epoch !== pageEpoch) return { changed: false, cached: !!data.cached };
      const norm = (str) => str.replace(/\s+/g, " ").trim();
      const changed = norm(data.simplified) !== norm(orig);
      resolveItem(el, "sent", changed);
      historyBuffer.push(historyEntry(orig, data, changed));
      if (changed) {
        el.dataset.originalPlaceholder = orig;
        el.setAttribute("placeholder", data.simplified);
        return { changed: true, cached: data.cached };
      }
      return { changed: false, cached: data.cached };
    })
    .catch((e) => {
      if (isNavigationAbort(e)) return { changed: false, cached: false };
      console.error("placeholder simplification failed", e);
      lastSimplifyError = e.message || String(e);
      resolveItem(el, "sent", false); // still counts as sent — it just failed
      return { changed: false, cached: false };
    });
}

// --- document-granularity collection ---
// A document-trained model expects a whole coherent document (its corpus averages ~180
// tokens per document, median 120). Below: find the main prose, cut it into
// heading-delimited sections, keep each under the model's ceiling.

// Main-prose containers in priority order, to skip site chrome. #mw-content-text is
// MediaWiki's, named because Wikipedia-style articles are the model's training domain.
const CONTENT_SCOPE_SELECTORS = [
  "article",
  "main",
  "#content",
  "#mw-content-text",
  "body",
];

function findContentScope() {
  for (const selector of CONTENT_SCOPE_SELECTORS) {
    const el = document.querySelector(selector);
    if (el && el.textContent.trim()) return el;
  }
  return document.body;
}

// Headings are section dividers only, never sent or rewritten: when included, the model
// echoed them into the body as duplicate prose. Also keeps the page's navigable
// structure for screen-reader and skim-reading users.
const HEADING_TAG_SET = new Set(["H1", "H2", "H3", "H4", "H5", "H6"]);
// Document body leaves. Narrower than CANDIDATE_TAGS: orphan links/buttons/labels are
// page chrome, not article prose.
const DOCUMENT_BODY_TAG_SET = new Set([
  "P",
  "LI",
  "BLOCKQUOTE",
  "FIGCAPTION",
  "TD",
  "TH",
]);
// ...plus a leaf <div> (no nested block, per computeLeafCandidates): body prose without
// a <p>. Omitting it silently dropped whole panes (e.g. demo/08-html-checker.html's
// output pane). A separate set because the document watcher reads DOCUMENT_BODY_TAG_SET
// without a leaf check, where a wrapper <div> would count late prose several times.
const DOCUMENT_BODY_LEAF_TAG_SET = new Set([...DOCUMENT_BODY_TAG_SET, "DIV"]);

// BART's BPE: ~1.3 tokens per English word (measured on the project's D-Wikipedia
// corpus). 1.6 leaves headroom, since the backend truncates silently and overshooting
// loses page content.
const TOKENS_PER_WORD = 1.6;
function estimateTokens(text) {
  return Math.ceil(wordCount(text) * TOKENS_PER_WORD);
}

// Mirrors the backend's document_text.ensure_terminal_punctuation. Unterminated segments
// fuse with the next (observed: an unpunctuated heading and the following paragraph
// came back as one phrase).
function ensureTerminalPunctuation(text) {
  const trimmed = text.trim();
  if (!trimmed) return "";
  return /[.?!]$/.test(trimmed) ? trimmed : `${trimmed}.`;
}

// Groups a page's prose into sections of units: one per heading-delimited run, split so
// none exceeds `maxTokens`.
// A unit is one chunk of one element, as on the sentence path, carrying the text nodes
// the answer is written back over (see writeChunkText); only the number of units per
// request differs.
// `text` is what the model is sent (whitespace collapsed, terminal punctuation added);
// `original` is what the page says, shown in the tooltip and History log.
// Leaves passed over (headings, chrome, too short) are counted as skipped items, as on
// the sentence path, so both paths' run summaries are comparable.
function collectSections(scope, maxTokens) {
  const leaves = computeLeafCandidates(
    Array.from(scope.querySelectorAll(CANDIDATE_SELECTOR)),
  );
  // restore document order (block leaves come before orphan inlines), since sections
  // depend on heading positions
  leaves.sort((a, b) =>
    a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1,
  );

  const sections = [];
  let current = [];
  const flushSection = () => {
    if (current.length > 0) sections.push(current);
    current = [];
  };
  // Resolved on the spot, like simplifyElement()'s skips. Leaf candidates never nest,
  // so registering them here can't hide one collected element's text from another.
  const skip = (el, reason) => {
    registerFound(el);
    resolveItem(el, "skipped", false, reason);
  };

  for (const el of leaves) {
    if (HEADING_TAG_SET.has(el.tagName)) {
      flushSection();
      // skipped by design (the delimiter), not a filter firing
      skip(el, "section-heading");
      continue;
    }
    if (
      !DOCUMENT_BODY_LEAF_TAG_SET.has(el.tagName) ||
      el.dataset.originalHtml ||
      !shouldSimplifyElement(el)
    ) {
      skip(
        el,
        !DOCUMENT_BODY_LEAF_TAG_SET.has(el.tagName)
          ? "not-document-body-tag"
          : el.dataset.originalHtml
            ? "already-simplified"
            : elementSkipReason(el),
      );
      continue;
    }
    // per element, not per page, so it doesn't keep every leaf's text nodes alive
    const pinned = new Set();
    const units = collectChunkNodes(el, pinned).reduce((kept, nodes) => {
      const raw = nodes.map((n) => n.data).join("");
      const original = raw.trim();
      // Collapsed: a <br> is a soft break within a document, not a sentence boundary.
      // `nodes` still records which side of it each word came from.
      const text = original.replace(/\s+/g, " ");
      const chunkReason = chunkSkipReason(text);
      if (chunkReason) {
        noteSkip(chunkSkipReasons, chunkReason, text);
        return kept;
      }
      // all pinned: no writable node, so no unit of its own
      if (nodes.every((node) => pinned.has(node))) {
        noteSkip(chunkSkipReasons, "all-text-pinned", text);
        return kept;
      }
      // `raw` keeps the following break, which only survives untrimmed (see
      // joinChunkOriginals)
      kept.push({
        el,
        nodes,
        pinned,
        raw,
        original,
        text: ensureTerminalPunctuation(text),
      });
      return kept;
    }, []);
    if (units.length === 0) {
      skip(el, "no-simplifiable-chunk");
      continue;
    }
    current.push(...units);
  }
  flushSection();

  // Split sections past the model's ceiling: the backend truncates, so an oversized
  // section would lose its tail.
  const groups = [];
  for (const section of sections) {
    let group = [];
    let tokens = 0;
    for (const unit of section) {
      const unitTokens = estimateTokens(unit.text);
      if (group.length > 0 && tokens + unitTokens > maxTokens) {
        groups.push(group);
        group = [];
        tokens = 0;
      }
      group.push(unit);
      tokens += unitTokens;
    }
    if (group.length > 0) groups.push(group);
  }
  return groups;
}

// Sentence-final punctuation (optionally followed by a closing quote/bracket), then
// whitespace. Deliberately naive next to the backend's pysbd: a wrong cut here only
// moves a sentence into a neighbouring paragraph.
//
// Second branch: the same boundary without whitespace ("boundaries.The Movement
// argued", a decoder dropping the space marker). Missing it shifted every later
// paragraph by one (a BBC article's nine paragraphs each held their neighbour's text).
// document_text.py restores that space, so this branch covers the prompted-LLM path,
// whose output skips the PTB de-tokenizer.
// Guards: the lookbehind is _PERIOD_NOT_IN_NUMBER's abbreviation guard shifted one
// character left (a lone letter closing "U.S." or "e.g."); and the next letter must be a
// capital (denormalize_from_model capitalises after every terminal mark), which spares
// "demo/README.md" (splitting it emptied its <code> element).
const OUTPUT_SENTENCE_BOUNDARY =
  /(?<=[.?!]["'”’)\]]?)\s+|(?<!(?<![^\W\d_])[^\W\d_][.?!])(?<=[.?!])(?=\p{Lu})/u;

// Removed before comparing shared vocabulary, leaving what rewrites tend to keep:
// names, numbers, terms.
const OVERLAP_STOP_WORDS = new Set(
  (
    "a an the and or but of in on at to for from by with as is are was were be been being " +
    "it its this that these those they them their he she his her him you your i we us our " +
    "not no nor so if then than there here which who whom whose what when where how why " +
    "can could will would shall should may might must do does did done have has had " +
    "also more most other such about into over under after before between"
  ).split(" "),
);

// Lowercased identifying words. Apostrophes, dots and hyphens stay inside a
// token: "demo/README.md" and "u.s." survive rewrites intact.
function contentWords(text) {
  const tokens =
    text.toLowerCase().match(/[\p{L}\p{N}][\p{L}\p{N}'’./\-]*/gu) || [];
  return new Set(tokens.filter((word) => !OVERLAP_STOP_WORDS.has(word)));
}

function sharedWordCount(words, vocabulary) {
  let shared = 0;
  words.forEach((word) => {
    if (vocabulary.has(word)) shared += 1;
  });
  return shared;
}

// Splits one section's answer into one piece per unit.
// Document models delete, merge and reorder, so there is no positional mapping. But
// sentence order follows unit order (a one-pointer walk), and rewrites keep most of
// their source's words: the pointer advances when the next unit shares more of the
// sentence's vocabulary. On ties (near-identical rows, no shared terms) it falls back
// on pacing: a unit that has taken its share of the answer (proportional to its input
// words) yields, and no unit is left empty while sentences remain.
// Folding the section into its first element instead lost the other elements' links,
// list items and labels; an approximate placement is better than a deleted paragraph.
function distributeSentences(units, output) {
  const parts = new Array(units.length).fill("");
  const sentences = output
    .trim()
    .split(OUTPUT_SENTENCE_BOUNDARY)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
  if (sentences.length === 0) return parts;

  const vocabulary = units.map((unit) => contentWords(unit.text));
  const inputWords = units.map((unit) => wordCount(unit.text));
  const totalInputWords =
    inputWords.reduce((sum, n) => sum + n, 0) || units.length;
  const answerWords = sentences.map(wordCount);
  const totalAnswerWords = answerWords.reduce((sum, n) => sum + n, 0) || 1;

  const buckets = units.map(() => []);
  let at = 0; // the unit being filled
  let taken = 0; // answer words handed to it so far

  sentences.forEach((sentence, i) => {
    const spoken = contentWords(sentence);
    // an empty unit keeps this sentence regardless of vocabulary
    while (at < units.length - 1 && buckets[at].length > 0) {
      const here = sharedWordCount(spoken, vocabulary[at]);
      const next = sharedWordCount(spoken, vocabulary[at + 1]);
      // this unit has had its proportional share of the answer
      const full =
        taken >= (totalAnswerWords * inputWords[at]) / totalInputWords;
      // ...and keeping this sentence here would leave a later unit with none
      const scarce = sentences.length - i < units.length - at;
      if (next > here || (next === here && (full || scarce))) {
        at += 1;
        taken = 0;
        continue;
      }
      break;
    }
    buckets[at].push(sentence);
    taken += answerWords[i];
  });

  return buckets.map((bucket) => bucket.join(" "));
}

// The section as the page wrote it, for the tooltip and History log (not the flattened,
// punctuated model input). Same rule as simplifyElement's `origText`: chunks joined by
// their break, elements by a blank line.
function sectionOriginalText(units) {
  const byElement = [];
  units.forEach((unit) => {
    const last = byElement[byElement.length - 1];
    if (last && last.el === unit.el) last.units.push(unit);
    else byElement.push({ el: unit.el, units: [unit] });
  });
  return byElement
    .map(({ units: own }) => elementOriginalText(own))
    .join("\n\n");
}

function elementOriginalText(units) {
  return joinChunkOriginals(
    units.map((unit) => unit.original),
    units.map((unit) => unit.raw),
  );
}

// Simplifies one section as a single document and writes the answer back across its
// units (distributeSentences, writeChunkText).
// One tallied item per section ("Whole sections": one request). The tally token is the
// first unit, not element, since an element split across two sections would otherwise
// be counted twice.
function simplifySection(units) {
  const documentText = units.map((unit) => unit.text).join(" ");
  const originalText = sectionOriginalText(units);

  const epoch = pageEpoch;
  return fetchSimplify(documentText)
    .then((data) => {
      if (epoch !== pageEpoch) return { changed: false, cached: !!data.cached };
      const norm = (str) => str.replace(/\s+/g, " ").trim();
      // against what was sent: the added terminal punctuation is not a change
      const changed = norm(data.simplified) !== norm(documentText);
      // units the answer reached; counted even when zero
      let written = 0;
      if (changed) {
        const parts = distributeSentences(units, data.simplified);
        // captured before each element's first write, only for elements written to;
        // the rest stay unmarked
        const touched = [];
        units.forEach((unit, i) => {
          if (!parts[i] || norm(parts[i]) === norm(unit.text)) return;
          if (unit.el.dataset.originalHtml === undefined) {
            unit.el.dataset.originalHtml = unit.el.innerHTML;
            touched.push(unit.el);
          }
          writeChunkText(unit.nodes, parts[i], unit.el, unit.pinned);
          written += 1;
        });
        touched.forEach((el) => {
          const elementText = elementOriginalText(
            units.filter((unit) => unit.el === el),
          );
          el.dataset.originalText = elementText;
          rememberTitle(el);
          rememberMissingAttributes(el);
          el.title = elementText;
          el.classList.add("simplified");
          adjustBackgroundForContrast(el);
        });
      }
      unwrittenUnitCount += units.length - written;
      historyBuffer.push(
        historyEntry(originalText, data, changed, {
          units: units.length,
          written,
        }),
      );
      resolveItem(units[0], "sent", changed);
      return { changed, cached: !!data.cached };
    })
    .catch((e) => {
      if (isNavigationAbort(e)) return { changed: false, cached: false };
      console.error("section simplification failed", e);
      lastSimplifyError = e.message || String(e);
      resolveItem(units[0], "sent", false);
      return { changed: false, cached: false };
    });
}

// `content`: a plain string (textContent) or an array of DOM nodes for mixed formatting.
function showNotice(content, duration = 3000) {
  const existing = document.getElementById("simplify-notice");
  if (existing) existing.remove();
  const div = document.createElement("div");
  div.id = "simplify-notice";
  // Opt-out, or the observer would simplify the notice's own text, which calls
  // showNotice() again: an infinite loop that floods the backend.
  div.setAttribute("translate", "no");
  if (typeof content === "string") {
    div.textContent = content;
  } else {
    div.append(...content);
  }
  document.body.appendChild(div);
  if (duration > 0) {
    setTimeout(() => div.remove(), duration);
  }
}

// Light-yellow highlight for dark text, dark brown for light text (dark themes).
function adjustBackgroundForContrast(el) {
  const color = getComputedStyle(el).color;
  const m = color.match(/(\d+),\s*(\d+),\s*(\d+)/);
  if (!m) return; // unparsable color (e.g. "transparent") — leave the CSS class's default yellow

  const [r, g, b] = m.slice(1).map((v) => parseInt(v, 10) / 255);
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  el.style.backgroundColor = lum > 0.5 ? "#5c3a21" : "#ffffcc";
}

function notifyBadge(isOn) {
  chrome.runtime.sendMessage({ cmd: "updateBadge", isSimplified: isOn });
}

// Which guard rejects this element, or null. Single source of truth for both the
// decision (shouldSimplifyElement()) and the reported reason, so they can't drift.
function elementSkipReason(el) {
  if (el.closest("pre, code, script, style")) return "code-or-preformatted";
  if (isOptedOut(el)) return "opted-out";
  const txt = el.textContent.trim();
  if (!txt) return "no-text";
  // common code indicators: braces, semicolons, arrow functions, keywords
  const codePattern = /\b(function|var|let|const|=>)\b|[{}<>;=]/;
  if (codePattern.test(txt)) return "code-like-text";
  return null;
}

function shouldSimplifyElement(el) {
  return elementSkipReason(el) === null;
}

function shouldSimplifyPlaceholder(el) {
  if (isOptedOut(el)) return false;
  const txt = (el.getAttribute("placeholder") || "").trim();
  return txt.length > 0;
}

function unitLabel(n) {
  return n === 1 ? "item" : "items";
}

// one decimal below 10s, whole seconds above
function formatDuration(ms) {
  const seconds = ms / 1000;
  return seconds < 10 ? `${seconds.toFixed(1)}s` : `${Math.round(seconds)}s`;
}

// ms below 1s: cache hits take single-digit ms, a 7B generation seconds
function formatLatency(ms) {
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
}

// --- notice box building blocks ---
// Separate elements rather than one text block, so rows can be weighted differently
// (styles in injectStyles()).
function noticeRow(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

// document.title is often empty (framesets, some app shells), hence the fallbacks.
function pageLabel() {
  const title = (document.title || "").trim();
  return title || location.hostname || "this page";
}

// Bar plus "19 / 23". A total of 0 fills the bar.
function progressBarRow(done, total) {
  const row = noticeRow("div", "sn-bar-row");
  const bar = noticeRow("div", "sn-bar");
  const fill = noticeRow("div", "sn-bar-fill");
  fill.style.width = `${total > 0 ? Math.min(100, (done / total) * 100) : 100}%`;
  bar.appendChild(fill);
  row.append(bar, noticeRow("span", "sn-count", `${done} / ${total}`));
  return row;
}

// Preflight bar: nothing to count yet, and 0% would read as stuck.
function indeterminateBarRow() {
  const row = noticeRow("div", "sn-bar-row");
  const bar = noticeRow("div", "sn-bar");
  bar.appendChild(noticeRow("div", "sn-bar-fill sn-bar-indeterminate"));
  row.appendChild(bar);
  return row;
}

// Settled run's replacement for the bar. One text line, not a flex row, so it reads as
// one sentence to a screen reader.
function processedRow(done, total) {
  const row = noticeRow("div", "sn-processed");
  row.append(
    noticeRow("span", "sn-count", `${done} / ${total}`),
    noticeRow("span", null, ` ${unitLabel(total)} processed`),
  );
  return row;
}

// "15 simplified · 4 unchanged", emphasis on simplified.
function outcomeRow(changed, unchanged) {
  const row = noticeRow("div", "sn-outcome");
  row.append(
    noticeRow("span", "sn-emph", `${changed} simplified`),
    noticeRow("span", null, ` · ${unchanged} unchanged`),
  );
  return row;
}

// Method, model and audience; same rows throughout the run so the box doesn't reflow.
// Method and model share one row (even when wrapped).
function selectionRows() {
  const sel = session.selection;
  if (!sel)
    return [
      noticeRow("div", "sn-meta sn-gap", "Reading your model selection…"),
    ];
  const row = noticeRow(
    "div",
    "sn-meta sn-gap",
    `Method: ${sel.methodLabel}; Model: `,
  );
  // verbatim id from /health in code style; the readable label only when there is no
  // id (a checkpoint the backend names by path)
  row.appendChild(noticeRow("code", null, sel.modelId || sel.modelLabel));
  const rows = [row];
  if (sel.audienceApplies)
    rows.push(noticeRow("div", "sn-meta", `Written for: ${sel.audienceLabel}`));
  return rows;
}

// Elapsed, throughput and per-request latency on one line; the latter two only once a
// request has completed.
function speedRow(elapsedText) {
  const parts = [elapsedText];
  if (session.startedAt && session.requests > 0) {
    const elapsedMs =
      (session.settledAt || performance.now()) - session.startedAt;
    parts.push(
      // items, not requests (a leaf split at <br>s is several requests)
      `${(sentCount / Math.max(elapsedMs / 1000, 0.001)).toFixed(1)} items/s`,
      `${formatLatency(session.requestMsTotal / session.requests)}/request`,
    );
  }
  return noticeRow("div", "sn-speed", parts.join(" · "));
}

// Rebuilt on every tick; renderProgressNotice() restores keyboard focus after each
// redraw, or the button couldn't be reached by tabbing.
const STOP_BUTTON_ID = "simplify-stop";

function stopButtonRow() {
  const row = noticeRow("div", "sn-actions");
  const button = noticeRow("button", "sn-btn", "Stop");
  button.id = STOP_BUTTON_ID;
  button.type = "button";
  button.title =
    "Stop simplifying this page. What has already been simplified stays.";
  button.addEventListener("click", (event) => {
    // keep the host page's click handlers out of it
    event.preventDefault();
    event.stopPropagation();
    stopCurrentRun();
  });
  row.appendChild(button);
  return row;
}

// Same rows as a settled run except the title and the "left as it was" line. Not a
// failure, so no countdown or troubleshooting.
function showStoppedNotice() {
  const plannedTotal = Math.max(foundTotal - skippedCount, sentCount);
  const remaining = Math.max(plannedTotal - sentCount, 0);
  const rows = [
    noticeRow("div", "sn-title", `Stopped simplifying ${pageLabel()}`),
  ];
  if (sentCount > 0) {
    rows.push(
      processedRow(sentCount, plannedTotal),
      outcomeRow(changedCount, Math.max(sentCount - changedCount, 0)),
    );
  }
  rows.push(
    noticeRow(
      "div",
      "sn-muted",
      remaining > 0
        ? `${remaining} ${unitLabel(remaining)} left as ${remaining === 1 ? "it was" : "they were"}. ` +
            `Click the icon twice to start again.`
        : "Nothing had been sent yet.",
    ),
    speedRow(
      `Stopped after ${formatDuration(performance.now() - session.clickedAt)}`,
    ),
    ...selectionRows(),
  );
  showNotice(rows, 8000);
}

// Notice for the current phase: preflight (indeterminate bar, selection, elapsed) or the
// run (sent / planned items). Stays visible while anything is in flight, auto-hides
// once settled, and reappears via registerFound()/resolveItem() if more content loads
// (e.g. infinite scroll).
function renderProgressNotice() {
  // reverted, or a failure notice is showing and a straggler just settled: don't
  // overwrite it
  if (session.phase === "idle") return;

  const label = pageLabel();
  // feeds other tabs' "already running" bar and the worker's heartbeat
  reportRunProgress();
  const stopWasFocused =
    document.activeElement && document.activeElement.id === STOP_BUTTON_ID;
  const restoreStopFocus = () => {
    if (!stopWasFocused) return;
    const button = document.getElementById(STOP_BUTTON_ID);
    if (button) button.focus();
  };

  if (session.phase === "preflight") {
    showNotice(
      [
        noticeRow("div", "sn-title", `Simplifying ${label}`),
        indeterminateBarRow(),
        speedRow(
          `Running for ${formatDuration(performance.now() - session.clickedAt)}`,
        ),
        ...selectionRows(),
        noticeRow(
          "div",
          "sn-status",
          "Checking the backend and warming up the model…",
        ),
        // longest single wait (up to a minute on the prompted-LLM path)
        stopButtonRow(),
      ],
      0,
    );
    restoreStopFocus();
    return;
  }

  const settled = skippedCount + sentCount >= foundTotal;
  // A settled run can resume when the observer picks up lazy content; elapsed time
  // keeps counting from the original click.
  if (settled && !session.settledAt) {
    session.settledAt = performance.now();
    session.phase = "settled";
    stopNoticeTicker();
    // Released now, not at session end (never, for a page left open). Later observer
    // work is incremental, not a new run, so it deliberately doesn't re-claim.
    releaseRun();
  } else if (!settled && session.settledAt) {
    session.settledAt = 0;
    session.phase = "simplifying";
    startNoticeTicker();
  }

  const totalMs = (session.settledAt || performance.now()) - session.clickedAt;
  // Discovered items not skipped. Skips are decided synchronously, so only later
  // batches move this. Math.max prevents "20 / 19" when a removal un-counts a sent item.
  const plannedTotal = Math.max(foundTotal - skippedCount, sentCount);
  const unchangedCount = Math.max(sentCount - changedCount, 0);
  const remaining = Math.max(plannedTotal - sentCount, 0);

  // everything skipped: the title says so and the skipped line explains it, instead of
  // "0 / 0" rows
  const nothingToDo = settled && plannedTotal === 0;

  const rows = [
    noticeRow(
      "div",
      "sn-title",
      nothingToDo
        ? `Nothing to simplify on ${label}`
        : settled
          ? `Simplified ${label} ✓`
          : `Simplifying ${label}`,
    ),
  ];
  if (!nothingToDo) {
    rows.push(
      settled
        ? processedRow(sentCount, plannedTotal)
        : progressBarRow(sentCount, plannedTotal),
      outcomeRow(changedCount, unchangedCount),
    );
  }
  // qualifying counts only when nonzero
  if (session.cachedRequests > 0) {
    rows.push(
      noticeRow(
        "div",
        "sn-muted",
        `${session.cachedRequests} reused from cache`,
      ),
    );
  }
  if (skippedCount > 0) {
    rows.push(
      noticeRow(
        "div",
        "sn-muted",
        `${skippedCount} / ${foundTotal} ${unitLabel(foundTotal)} skipped (e.g. too short)`,
      ),
    );
  }
  // document mode: paragraphs inside answered sections that the answer never reached
  // (see unwrittenUnitCount)
  if (unwrittenUnitCount > 0) {
    rows.push(
      noticeRow(
        "div",
        "sn-muted",
        `${unwrittenUnitCount} ${unitLabel(unwrittenUnitCount)} of prose left as-is ` +
          `(the model's answer did not cover ${unwrittenUnitCount === 1 ? "it" : "them"})`,
      ),
    );
  }
  // directly under the counts, above the fixed footer, where it is harder to miss
  if (!settled) {
    rows.push(
      noticeRow(
        "div",
        "sn-status",
        remaining > 0
          ? `Processing remaining ${remaining} ${unitLabel(remaining)}…`
          : "Waiting for the last items to settle…",
      ),
    );
  }
  rows.push(
    speedRow(
      settled
        ? `Completed in ${formatDuration(totalMs)}`
        : `Running for ${formatDuration(totalMs)}`,
    ),
  );
  rows.push(...selectionRows());
  if (!settled) rows.push(stopButtonRow());
  // 8s: the only place the run's speed figures are shown
  showNotice(rows, settled ? 8000 : 0);
  restoreStopFocus();
}

// --- notice box failure state ---
// Same box and rows as renderProgressNotice(): title, body, hint, status line.
// The raw error is deliberately left out; it goes to the troubleshooting tab's
// "Technical details" (via the `error` field below) and the console.
// Four states: what matters to a reader is whether the service is missing (start it)
// or misbehaved (retry, or read the tab that opens).
const FAILURE_STATES = {
  unreachable: {
    body: "The simplification service isn't available.",
    hint: "Please make sure the local service is running, then try again.",
    stage: "reachable",
  },
  request_failed: {
    body: "The simplification service returned an error while processing this page.",
    hint: "Please try again or check the troubleshooting information.",
    stage: "model_response",
  },
  timeout: {
    body: "The simplification service took too long to respond.",
    hint: "Please try again or check the troubleshooting information.",
    stage: "model_response",
  },
  invalid_response: {
    body: "The simplification service returned an unexpected response.",
    hint: "Please check the troubleshooting information.",
    stage: "model_response",
  },
};

// Must match background.js's INFO_TAB_OPEN_DELAY_MS: the box shows this countdown.
const TROUBLESHOOT_OPEN_DELAY_MS = 3000;
const COUNTDOWN_TICK_MS = 500;
// must outlive the countdown, so the box doesn't vanish as the tab opens
const NOTICE_FAILURE_MS = 8000;

let countdownTimer = null;

function stopFailureCountdown() {
  if (!countdownTimer) return;
  clearInterval(countdownTimer);
  countdownTimer = null;
}

// `stage` is set only for preflight failures (background.js checkBackendHealth); run
// failures have only the error text, so both are consulted.
function classifyFailure(stage, errorText) {
  // loading / not loaded: both "can't serve yet" here; the troubleshooting tab has
  // a section for each
  if (
    stage === "reachable" ||
    stage === "model_loaded" ||
    stage === "model_unavailable"
  )
    return "unreachable";

  const text = String(errorText || "").toLowerCase();
  if (
    /failed to fetch|not reachable|networkerror|load failed|connection refused|err_connection/.test(
      text,
    )
  ) {
    return "unreachable";
  }
  // fetchWithTimeout's timeout arrives as an AbortError; navigation aborts never
  // reach here (filtered before lastSimplifyError is set)
  if (/abort|timeout|timed out|took too long/.test(text)) return "timeout";
  if (
    /no usable output|unexpected token|unexpected end|json|syntaxerror/.test(
      text,
    )
  )
    return "invalid_response";
  return "request_failed";
}

// Renders the failure state and opens troubleshooting. `stage` is passed through when
// known (preflight), else the state's own stage.
function showFailureNotice(stage, errorText) {
  const kind = classifyFailure(stage, errorText);
  const failure = FAILURE_STATES[kind];
  console.error(
    `simplification failed (${kind}) at stage "${stage || failure.stage}": ${errorText}`,
  );

  stopFailureCountdown();
  const startedAt = performance.now();
  const render = () => {
    const secondsLeft = Math.ceil(
      (TROUBLESHOOT_OPEN_DELAY_MS - (performance.now() - startedAt)) / 1000,
    );
    const status =
      secondsLeft > 0
        ? `Troubleshooting will open automatically in ${secondsLeft} second${secondsLeft === 1 ? "" : "s"}…`
        : "Opening troubleshooting…";
    showNotice(
      [
        noticeRow("div", "sn-title", `Couldn't simplify ${pageLabel()}`),
        noticeRow("div", "sn-outcome", failure.body),
        noticeRow("div", "sn-muted", failure.hint),
        noticeRow("div", "sn-status", status),
      ],
      NOTICE_FAILURE_MS,
    );
    if (secondsLeft <= 0) stopFailureCountdown();
  };
  render();
  countdownTimer = setInterval(render, COUNTDOWN_TICK_MS);

  // sent now: the service worker owns the delay, so the tab still opens if this page
  // closes or navigates during the countdown
  chrome.runtime.sendMessage({
    cmd: "openErrorPage",
    stage: stage || failure.stage,
    error: errorText,
  });
}

// Called before any async work, so "Found" is correct immediately.
function registerFound(el) {
  itemStatus.set(el, "pending");
  ownUnits.add(el);
  // stale from a previous toggle cycle on the same element
  changedItems.delete(el);
  foundTotal += 1;
}

// Records an item's outcome (synchronously for skips, after the fetch otherwise).
// No-op if the item was removed while in flight: forgetItem() already un-counted it.
function resolveItem(
  el,
  outcome,
  changed = false,
  reason = null,
  sampleText = null,
) {
  if (itemStatus.get(el) === "removed") return;
  itemStatus.set(el, outcome);
  if (outcome === "skipped") {
    skippedCount += 1;
    // sampled here, not at call sites, so every skip path records a reason and sample
    noteSkip(skipReasons, reason, sampleText ?? el.textContent);
  } else {
    sentCount += 1;
    if (changed) {
      changedItems.add(el);
      changedCount += 1;
    }
  }
  renderProgressNotice();
}

// Un-counts an item removed from the page (e.g. a virtualized list recycling rows).
// Pending items are just marked "removed" so resolveItem() ignores them later.
function forgetItem(el) {
  const status = itemStatus.get(el);
  if (status === undefined || status === "removed") return;
  if (status === "skipped") {
    skippedCount -= 1;
  } else if (status === "sent") {
    sentCount -= 1;
    if (changedItems.has(el)) {
      changedItems.delete(el);
      changedCount -= 1;
    }
  }
  foundTotal -= 1;
  itemStatus.set(el, "removed");
  renderProgressNotice();
}

// `granularity` (from the health preflight; one of GRANULARITY_CHOICES in
// shared/model-labels.js): sentence-by-sentence walks every leaf element,
// whole-sections builds heading-delimited sections from the main content area.
function simplifyPage(
  granularity = GRANULARITY_SENTENCE_BY_SENTENCE,
  documentMaxTokens = 512,
) {
  // totals shared with the observer's later batches
  foundTotal = 0;
  skippedCount = 0;
  sentCount = 0;
  changedCount = 0;
  unwrittenUnitCount = 0;
  fallbackCount = 0;
  resetSkipBreakdown();
  ownUnits = new WeakSet();
  lastSimplifyError = null;

  if (granularity === GRANULARITY_WHOLE_SECTIONS) {
    return simplifyPageAsDocuments(documentMaxTokens);
  }

  const candidates = Array.from(document.querySelectorAll(CANDIDATE_SELECTOR));
  const elems = computeLeafCandidates(candidates);
  const placeholderEls = Array.from(
    document.querySelectorAll(PLACEHOLDER_SELECTOR),
  );

  startObserver();

  // all registered up front; skip decisions happen per item in
  // simplifyElement/simplifyPlaceholder
  elems.forEach(registerFound);
  placeholderEls.forEach(registerFound);
  renderProgressNotice();

  const wrapped = elems
    .map((el) => simplifyElement(el))
    .concat(placeholderEls.map((el) => simplifyPlaceholder(el)));

  return Promise.all(wrapped).then((results) => {
    flushHistoryBuffer();
    if (lastSimplifyError) {
      // end first, so no straggler re-renders progress over the failure notice
      endSession();
      showFailureNotice(null, lastSimplifyError);
      return false;
    }
    const visiblyChangedCount = results.filter((r) => r.changed).length;
    if (visiblyChangedCount > 0) notifyBadge(true);
    return visiblyChangedCount > 0;
  });
}

// Document-granularity counterpart to simplifyPage(): one request per section.
// No MutationObserver simplification here: a lazily added paragraph isn't a document,
// and re-sending its section would overwrite text being read. startDocumentWatcher()
// only reports late prose.
function simplifyPageAsDocuments(maxTokens) {
  const scope = findContentScope();
  // also tallies the skipped leaves
  const sections = collectSections(scope, maxTokens);

  if (sections.length === 0) {
    // before endSession(): renderProgressNotice() won't draw over an idle session;
    // shows the skipped tally
    renderProgressNotice();
    endSession();
    return Promise.resolve(false);
  }

  startDocumentWatcher(scope);

  // first unit stands in for the section (see simplifySection)
  sections.forEach((units) => registerFound(units[0]));
  renderProgressNotice();

  return Promise.all(sections.map((units) => simplifySection(units))).then(
    (results) => {
      flushHistoryBuffer();
      if (lastSimplifyError) {
        // end first, so no straggler re-renders progress over the failure notice
        endSession();
        showFailureNotice(null, lastSimplifyError);
        return false;
      }
      const visiblyChangedCount = results.filter((r) => r.changed).length;
      if (visiblyChangedCount > 0) notifyBadge(true);
      return visiblyChangedCount > 0;
    },
  );
}

// The original-text tooltip overwrites the page's own `title`; save it first so the
// revert can restore it.
function rememberTitle(el) {
  if (el.dataset.originalTitle === undefined && el.hasAttribute("title")) {
    el.dataset.originalTitle = el.getAttribute("title");
  }
}

function restoreTitle(el) {
  if (el.dataset.originalTitle !== undefined) {
    el.setAttribute("title", el.dataset.originalTitle);
    delete el.dataset.originalTitle;
  } else {
    el.removeAttribute("title");
  }
}

// Removing our class/highlight leaves an empty `class=""`/`style=""` the page never had,
// which `:not([class])` selectors and hasAttribute() checks can see.
// Recorded rather than inferred: pages can ship `<li class="">` themselves (nextjs.org
// does; the RQ2 site audit caught an earlier infer-at-revert version deleting it on four
// <li>s).
function rememberMissingAttributes(el) {
  if (el.dataset.addedClassAttr === undefined && !el.hasAttribute("class")) {
    el.dataset.addedClassAttr = "1";
  }
  if (el.dataset.addedStyleAttr === undefined && !el.hasAttribute("style")) {
    el.dataset.addedStyleAttr = "1";
  }
}

function dropAddedAttributes(el) {
  if (el.dataset.addedClassAttr !== undefined) {
    if (el.getAttribute("class") === "") el.removeAttribute("class");
    delete el.dataset.addedClassAttr;
  }
  if (el.dataset.addedStyleAttr !== undefined) {
    if (el.getAttribute("style") === "") el.removeAttribute("style");
    delete el.dataset.addedStyleAttr;
  }
}

function revertPage() {
  document.querySelectorAll("[data-original-html]").forEach((el) => {
    el.innerHTML = el.dataset.originalHtml;
    delete el.dataset.originalHtml;
    delete el.dataset.originalText;
    restoreTitle(el);
    el.classList.remove("simplified");
    el.style.backgroundColor = "";
    dropAddedAttributes(el);
  });
  document.querySelectorAll("[data-original-placeholder]").forEach((el) => {
    el.setAttribute("placeholder", el.dataset.originalPlaceholder);
    delete el.dataset.originalPlaceholder;
  });
  stopObserver();
}

function toggleSimplification() {
  // a leftover countdown would redraw the failure box over this run's notice
  stopFailureCountdown();
  if (isSimplified) {
    showNotice("reverting to original text...");
    revertPage();
    notifyBadge(false);
    isSimplified = false;
    endSession();
    showNotice("restored original text");
  } else {
    // Notice first, before the lock claim, so the click gets immediate feedback. If
    // refused, the same box becomes the "already running" notice.
    beginSession();
    requestSelectionInfo();
    renderProgressNotice();
    // An SPA can route even during the claim; a run started afterwards would target the
    // new page after its session had already ended, and nothing would end it.
    const clickEpoch = pageEpoch;
    claimRun(runDescriptor(ACTIVE_RUN_PAGE)).then((claim) => {
      if (clickEpoch !== pageEpoch) {
        endSession();
        return;
      }
      if (!claim.ok) {
        // before the notice, so the ticker can't redraw progress over it
        endSession();
        showBlockedNotice(claim.holder);
        return;
      }
      runPreflightAndSimplify();
    });
  }
}

// Runs once the lock is claimed.
function runPreflightAndSimplify() {
  // supersedes any earlier "already running" notice
  blockedByRun = null;
  // The preflight can take up to a minute (prompted-LLM path), so the notice is
  // already up, and its model/method comes from a storage-only lookup
  // (requestSelectionInfo). The page may also navigate meanwhile; the epoch check
  // prevents simplifying a page nobody clicked on.
  const epoch = pageEpoch;
  checkBackend().then((health) => {
    if (epoch !== pageEpoch) return;
    session.preflightMs = performance.now() - session.clickedAt;
    if (!health.ok) {
      endSession();
      showFailureNotice(health.stage, health.error);
      return;
    }
    // health-backed (real repo id / Ollama tag), replaces the storage-only lookup
    session.selection = health;
    session.phase = "simplifying";
    session.startedAt = performance.now();
    console.log(
      `backend check passed in ${Math.round(session.preflightMs)}ms, ` +
        `simplifying with '${health.model}' (${health.granularity} granularity)`,
    );
    simplifyPage(health.granularity, health.documentMaxTokens).then(
      (changed) => {
        if (changed) {
          isSimplified = true;
        }
      },
    );
  });
}

function processObservedElements(candidateEls, placeholderEls) {
  // filtering happens per item in simplifyElement/simplifyPlaceholder
  const toSimplify = computeLeafCandidates(Array.from(candidateEls));
  const toSimplifyPlaceholders = Array.from(placeholderEls);
  if (toSimplify.length === 0 && toSimplifyPlaceholders.length === 0) return;

  // extends the session's cumulative tally
  toSimplify.forEach(registerFound);
  toSimplifyPlaceholders.forEach(registerFound);
  renderProgressNotice();

  const promises = toSimplify
    .map((el) => simplifyElement(el))
    .concat(toSimplifyPlaceholders.map((el) => simplifyPlaceholder(el)));

  Promise.all(promises).then((results) => {
    flushHistoryBuffer();
    if (lastSimplifyError) {
      // end first, so no straggler re-renders progress over the failure notice
      endSession();
      showFailureNotice(null, lastSimplifyError);
      return;
    }
    if (results.some((r) => r.changed)) notifyBadge(true);
  });
}

function startObserver() {
  if (observer) return;
  const REMOVAL_SELECTOR = `${CANDIDATE_SELECTOR}, ${PLACEHOLDER_SELECTOR}`;
  observer = new MutationObserver((mutations) => {
    const candidateEls = new Set();
    const placeholderEls = new Set();
    const removedEls = new Set();
    for (const m of mutations) {
      m.addedNodes.forEach((node) => {
        if (node.nodeType !== Node.ELEMENT_NODE) return;
        const el = node;
        if (OWN_UI_IDS.has(el.id)) return;
        if (el.matches && el.matches(CANDIDATE_SELECTOR)) candidateEls.add(el);
        el.querySelectorAll &&
          el
            .querySelectorAll(CANDIDATE_SELECTOR)
            .forEach((c) => candidateEls.add(c));
        if (el.matches && el.matches(PLACEHOLDER_SELECTOR))
          placeholderEls.add(el);
        el.querySelectorAll &&
          el
            .querySelectorAll(PLACEHOLDER_SELECTOR)
            .forEach((c) => placeholderEls.add(c));
      });
      // virtualized/infinite-scroll lists recycle rows; keep "Found" in sync
      m.removedNodes.forEach((node) => {
        if (node.nodeType !== Node.ELEMENT_NODE) return;
        const el = node;
        if (OWN_UI_IDS.has(el.id)) return;
        if (el.matches && el.matches(REMOVAL_SELECTOR)) removedEls.add(el);
        el.querySelectorAll &&
          el
            .querySelectorAll(REMOVAL_SELECTOR)
            .forEach((c) => removedEls.add(c));
      });
    }
    // A routed navigation looks like lazy content except for the URL; check first.
    if (checkForNavigation()) return;
    processObservedElements(candidateEls, placeholderEls);
    removedEls.forEach(forgetItem);
  });
  observer.observe(document.body, { childList: true, subtree: true });
}

// Document mode's watch-only counterpart to startObserver(); deliberately simplifies
// nothing. A lone lazy paragraph is not the unit the document model was trained on
// (5.6), and re-sending its already rewritten section would replace text being read.
// Closing the gap is a design question (7.1).
function startDocumentWatcher(scope) {
  if (observer) return;
  // Threshold for a notice, not a simplification (the sentence path has no minimum,
  // 5.2): avoids a toast for every cookie banner, tooltip or swapped button label.
  const MIN_WORDS = 12;
  const isLateProse = (el) =>
    DOCUMENT_BODY_TAG_SET.has(el.tagName) &&
    !el.closest("[data-original-html]") &&
    !isOptedOut(el) &&
    el.textContent.trim().split(/\s+/).filter(Boolean).length >= MIN_WORDS;

  observer = new MutationObserver((mutations) => {
    // a new page's content is not late content on this one
    if (checkForNavigation()) return;
    let found = 0;
    for (const m of mutations) {
      m.addedNodes.forEach((node) => {
        if (node.nodeType !== Node.ELEMENT_NODE) return;
        if (OWN_UI_IDS.has(node.id)) return;
        if (!scope.contains(node)) return;
        if (isLateProse(node)) found += 1;
        node.querySelectorAll &&
          node.querySelectorAll(CANDIDATE_SELECTOR).forEach((el) => {
            if (isLateProse(el)) found += 1;
          });
      });
    }
    if (found === 0) return;
    unsimplifiedLateCount += found;
    // Coalesced, since infinite scroll mutates continuously; the count is cumulative.
    clearTimeout(lateContentTimer);
    lateContentTimer = setTimeout(() => {
      showNotice(
        `${unsimplifiedLateCount} new text ${unitLabel(unsimplifiedLateCount)} loaded after ` +
          `simplification and ${unsimplifiedLateCount === 1 ? "was" : "were"} left as-is — ` +
          `document mode rewrites whole sections, so it can't simplify content that arrives ` +
          `one paragraph at a time. Toggle off and on again to include it.`,
        8000,
      );
    }, 1200);
  });
  observer.observe(document.body, { childList: true, subtree: true });
}

function stopObserver() {
  if (observer) {
    observer.disconnect();
    observer = null;
  }
  clearTimeout(lateContentTimer);
  lateContentTimer = 0;
  unsimplifiedLateCount = 0;
  foundTotal = 0;
  skippedCount = 0;
  sentCount = 0;
  changedCount = 0;
  unwrittenUnitCount = 0;
  resetSkipBreakdown();
}

// --- navigation ---
// SPAs replace the page without reloading this script (router clicks, pushState,
// in-document redirects). A navigation ends the run and reverts; the new page waits
// for a click. Otherwise the old observer kept simplifying (or, in document mode,
// reporting) the new page's content with the badge still ON.
// history.pushState can't be hooked: content scripts run in an isolated world. So the
// URL is watched on popstate/hashchange, on every mutation batch (usually first), and
// on a slow poll.
let currentUrl = location.href;
// Bumped on every navigation; work started earlier writes nothing back.
let pageEpoch = 0;
const NAV_POLL_MS = 700;
let navWatcher = 0;

function startNavWatcher() {
  if (navWatcher) return;
  navWatcher = setInterval(checkForNavigation, NAV_POLL_MS);
}

function stopNavWatcher() {
  if (!navWatcher) return;
  clearInterval(navWatcher);
  navWatcher = 0;
}

// Returns whether the URL changed (ending any run); mutation callbacks then stop.
function checkForNavigation() {
  if (location.href === currentUrl) return false;
  const previous = {
    sessionId: pageSessionId,
    url: currentUrl,
    title: document.title,
  };
  currentUrl = location.href;
  pageEpoch += 1;
  pageSessionId = crypto.randomUUID();
  endRunForNavigation(previous);
  // after endRunForNavigation: its flush is judged against the old page's record
  pageSessionRecorded = false;
  return true;
}

function endRunForNavigation(previous) {
  const wasActive = isSimplified || session.phase !== "idle";
  abortQueuedSimplifications();
  // would otherwise redraw the failure box over the "new page" notice
  stopFailureCountdown();
  // Persistent parts the router left (header, sidebar) still show simplified text.
  // Also detaches the observer.
  revertPage();
  endSession();
  isSimplified = false;
  // flushed against the URL they came from
  flushHistoryBuffer({ ...previous, timestamp: Date.now() });
  if (!wasActive) return;
  notifyBadge(false);
  showNotice(
    "this is a new page — simplification is off. Click the icon to simplify it.",
    5000,
  );
}

window.addEventListener("popstate", checkForNavigation);
window.addEventListener("hashchange", checkForNavigation);

// Release on close/reload: the worker otherwise only notices a reload after the
// staleness window (fifteen seconds of a stuck lock).
window.addEventListener("pagehide", () => {
  if (runIsActive()) releaseRun();
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.cmd === "toggle") {
    toggleSimplification();
    // Must respond: otherwise Chrome closes the port ("The message port closed before
    // a response was received") and background.js misreads that as "no content script
    // in this tab".
    sendResponse({ ok: true });
  }

  // Panel is shared/picker.js, opened from the toolbar icon's menu. Responds for the
  // same reason as "toggle" (background.js would fall back to an extension page).
  if (message.cmd === "showModelPicker") {
    showModelPicker();
    sendResponse({ ok: true });
  }

  // shared/compare-picker.js; responds for the same reason as above.
  if (message.cmd === "showCompareModels") {
    showComparePicker();
    sendResponse({ ok: true });
  }

  // Stop from the toolbar menu or another tab. Reports whether a run was stopped (it
  // may have just finished).
  if (message.cmd === "stopRun") {
    sendResponse({ ok: true, stopped: stopCurrentRun() });
  }
});

// Called by shared/picker.js when the panel closes after the selection changed. A
// simplified page shows the previous model's output, so reload it (only then: a reload
// costs scroll position and form input). Reload rather than revert, since the page's
// scripts have already seen the rewritten DOM.
function onPickerSelectionCommitted() {
  if (!isSimplified) return;
  console.log(
    "simplification settings changed - reloading to drop the previous model's output",
  );
  location.reload();
}
