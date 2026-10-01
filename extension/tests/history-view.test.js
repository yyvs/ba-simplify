// History page rendering of a recorded run: the summary header (status, what ran,
// outcome, speed, readability change), the entry table's columns, and the three row
// states: simplified, unchanged/fallback, cached.
//
//   cd extension/tests && npm install && node history-view.test.js
//
// Same harness as history-analysis.test.js (vm context, jsdom from history.html's markup).
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { JSDOM } = require("jsdom");

const EXT = path.join(__dirname, "..");
const SHARED = path.join(EXT, "shared");
const SOURCES = ["model-labels.js", "model-selection.js", "analysis.js", "history.js"].map((f) =>
  fs.readFileSync(path.join(SHARED, f), "utf8")
);
const PAGE_HTML = fs
  .readFileSync(path.join(EXT, "history.html"), "utf8")
  .replace(/<script[^>]*>[\s\S]*?<\/script>/g, "");

let failures = 0;
function check(label, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${ok || !detail ? "" : ` — ${detail}`}`);
  if (!ok) failures += 1;
}

// Counts deliberately don't match the entries: a page is capped at MAX_ENTRIES_PER_PAGE
// while stats describe the whole run, so the header must read stats, not count rows.
const SIMPLIFIED_ENTRY = {
  input: "Residents are requested to position their receptacles at the kerbside.",
  modelResult: "Residents are asked to position their receptacles at the kerbside.",
  output: "Residents are asked to position their receptacles at the kerbside.",
  changed: true,
  cached: false,
  fallbackReason: null,
  requestTimeMs: 1400,
};

const FALLBACK_ENTRY = {
  input: "Collections will commence at approximately 6 a.m. on Tuesdays.",
  // only a full stop dropped: the change guard rejected it and the input was served back
  modelResult: "Collections will commence at approximately 6 a.m. on Tuesdays",
  output: "Collections will commence at approximately 6 a.m. on Tuesdays.",
  changed: false,
  cached: false,
  fallbackReason: "no_meaningful_change",
  requestTimeMs: 1400,
};

const CACHED_ENTRY = {
  input: "Please contact the council before the collection day.",
  modelResult: "Please contact the council before collection day.",
  output: "Please contact the council before collection day.",
  changed: true,
  cached: true,
  fallbackReason: null,
  requestTimeMs: 1400,
};

const PAGE = {
  sessionId: "session-1",
  timestamp: 1700000000000,
  title: "Demo 08 — HTML checker",
  url: "https://example.com/bins",
  method: "fine_tuned_seq2seq",
  modelKey: "finetuned",
  modelId: "yunvs/bart-base-wikilarge-simplification",
  granularity: "sentence_by_sentence",
  totalTimeMs: 4700,
  status: "completed_with_warnings",
  stats: {
    totalItems: 93,
    skippedItems: 70,
    processedItems: 23,
    simplifiedItems: 19,
    unchangedItems: 4,
    cachedResponses: 3,
  },
  entries: [SIMPLIFIED_ENTRY, FALLBACK_ENTRY, CACHED_ENTRY],
};

// Prompted LLM (the only kind with an audience), finished clean.
const LLM_PAGE = {
  sessionId: "session-2",
  timestamp: 1700000100000,
  title: "Bin collections",
  url: "https://example.com/other",
  method: "prompted_llm",
  modelKey: "llm_7b",
  modelId: "qwen2.5:7b-instruct",
  granularity: "whole_sections",
  audience: "children",
  totalTimeMs: 21000,
  status: "completed",
  stats: {
    totalItems: 4,
    skippedItems: 0,
    processedItems: 4,
    simplifiedItems: 4,
    unchangedItems: 0,
    cachedResponses: 0,
  },
  entries: [{ ...SIMPLIFIED_ENTRY, requestTimeMs: 5200 }],
};

// Legacy format: `original`/`simplified` per entry, no session-level fields. Must still render.
const LEGACY_PAGE = {
  sessionId: "session-3",
  timestamp: 1699000000000,
  title: "An older page",
  url: "https://example.com/legacy",
  entries: [
    {
      original: "Residents are requested to position their receptacles at the kerbside.",
      simplified: "Residents should put their bins at the kerb.",
      model: "finetuned",
      cached: false,
      changed: true,
    },
  ],
};

function load(pages) {
  const dom = new JSDOM(PAGE_HTML, { pretendToBeVisual: true, url: "chrome-extension://test/history.html" });
  const { window } = dom;
  const store = { simplifyHistory: pages };

  const context = {
    window,
    document: window.document,
    Blob: window.Blob,
    URL: window.URL,
    navigator: { clipboard: { writeText: () => Promise.resolve() } },
    confirm: () => true,
    setTimeout,
    clearTimeout,
    console,
    chrome: {
      runtime: { sendMessage: () => Promise.resolve({ ok: false, error: "not used here" }) },
      storage: {
        local: {
          get: (key) => Promise.resolve(key in store ? { [key]: store[key] } : {}),
          set: (values) => {
            Object.assign(store, values);
            return Promise.resolve();
          },
          remove: (key) => {
            delete store[key];
            return Promise.resolve();
          },
        },
        onChanged: { addListener() {} },
      },
    },
  };
  context.globalThis = context;
  vm.createContext(context);
  SOURCES.forEach((source) => vm.runInContext(source, context));
  return { context, window };
}

const { context, window } = load([PAGE, LLM_PAGE, LEGACY_PAGE]);
const tick = (ms = 0) => new Promise((resolve) => setTimeout(resolve, ms));

(async () => {
  // render() runs on load and reads storage
  await tick(10);

  const blocks = Array.from(window.document.querySelectorAll(".page-block"));
  check("every stored page renders a block", blocks.length === 3, `${blocks.length} blocks`);

  const [block, llmBlock, legacyBlock] = blocks;
  const lines = (el) => Array.from(el.querySelectorAll(".page-aggregate")).map((n) => n.textContent.trim());

  // --- the status indicator
  const dot = block.querySelector("h2 .status-dot");
  check("the title carries a round status indicator", !!dot, block.querySelector("h2").innerHTML);
  check("a run with fallbacks is marked orange", !!dot && dot.classList.contains("status-dot-warn"), dot && dot.className);
  check("the indicator explains itself on hover", !!dot && /fallback/i.test(dot.title), dot && dot.title);
  check("...and to a screen reader too", !!dot && dot.getAttribute("aria-label") === dot.title);

  const okDot = llmBlock.querySelector("h2 .status-dot");
  check("a clean run is marked green", !!okDot && okDot.classList.contains("status-dot-ok"), okDot && okDot.className);

  const failed = context.buildStatusDot("failed");
  check("a failed run is marked red", failed.classList.contains("status-dot-fail"), failed.className);
  const unknown = context.buildStatusDot(undefined);
  check("a page with no recorded status makes no claim", unknown.classList.contains("status-dot-unknown"), unknown.className);

  // --- the header
  const h2 = block.querySelector("h2");
  check("the title is an h2 and reads as the page's title", h2.textContent.includes("Demo 08 — HTML checker"), h2.textContent);

  const meta = block.querySelector(".page-meta");
  check("URL and timestamp share the page-meta line", /example\.com\/bins · /.test(meta.textContent), meta.textContent);

  const [selection, outcome, speed, metrics] = lines(block);
  check(
    "the selection line names method, model, granularity",
    selection === "Fine-tuned seq2seq · yunvs/bart-base-wikilarge-simplification · Sentence by sentence",
    selection
  );
  check(
    "the exact model id is set in monospace",
    !!block.querySelector(".page-aggregate .page-model-id"),
    selection
  );
  check(
    "a seq2seq run records no audience, so none is claimed",
    !/speaker|children|literacy/i.test(selection),
    selection
  );
  check(
    "the outcome line counts simplified, unchanged and skipped",
    outcome === "19 simplified · 4 unchanged · 70 items skipped (e.g. too short)",
    outcome
  );
  check(
    "the speed line reports elapsed, throughput and per-request latency",
    speed === "Completed in 4.7s · 4.9 items/s · 1.4s/request (3 reused from cache)",
    speed
  );
  check(
    "the metrics line reports readability and word counts",
    /^Readability \(FK grade\): avg [\d.]+ → [\d.]+ · Words: avg [-+]?\d+%$/.test(metrics),
    metrics
  );

  const llmSelection = lines(llmBlock)[0];
  check(
    "a prompted run names its audience, which is the only kind that has one",
    llmSelection === "Prompted LLM · qwen2.5:7b-instruct · Whole sections · Children · 8–12",
    llmSelection
  );
  check("a finished run reports its elapsed time as 'Completed in'", /^Completed in 21s/.test(lines(llmBlock)[2]), lines(llmBlock)[2]);
  const failedSpeed = context.buildSpeedLine({ ...PAGE, status: "failed" }, PAGE.stats);
  check("a failed run's elapsed time is not called 'completed'", /^Stopped after 4\.7s/.test(failedSpeed.textContent), failedSpeed.textContent);

  // --- the table
  const headers = Array.from(block.querySelectorAll("thead th")).map((th) => th.textContent);
  check(
    "the table's columns are # / Input / Simplification / Time / Readability / Words, and an unlabelled one for the row menu",
    headers.join(" | ") === "# | Input | Simplification | Time | Readability | Words | ",
    headers.join(" | ")
  );
  const actionsHeader = Array.from(block.querySelectorAll("thead th")).pop();
  check(
    "...the last one still named for a screen reader, which has no ⋮ to look at",
    actionsHeader.getAttribute("aria-label") === "Actions",
    actionsHeader.getAttribute("aria-label")
  );
  const readabilityHeader = Array.from(block.querySelectorAll("thead th")).find((th) => th.textContent === "Readability");
  check(
    "the Readability header explains Flesch-Kincaid on hover",
    /Flesch-Kincaid Grade Level/.test(readabilityHeader.title),
    readabilityHeader.title
  );

  const rows = Array.from(block.querySelectorAll("tbody tr"));
  const cells = (tr) => Array.from(tr.querySelectorAll("td"));

  // a simplified row
  const simplified = cells(rows[0]);
  check("a simplified row is numbered from 1", simplified[0].textContent === "1", simplified[0].textContent);
  check("...shows the input it was given", simplified[1].textContent === SIMPLIFIED_ENTRY.input, simplified[1].textContent);
  check("...shows the output the page ended up with", simplified[2].textContent === SIMPLIFIED_ENTRY.output, simplified[2].textContent);
  check("...reports its own request time", simplified[3].textContent === "1.4s", simplified[3].textContent);
  check("...and is not muted", !simplified[2].classList.contains("cell-muted"));
  check("...with readability and word figures", /→/.test(simplified[4].textContent) && /%$/.test(simplified[5].textContent),
    `${simplified[4].textContent} | ${simplified[5].textContent}`);
  check("...and a three-dot menu", !!simplified[6].querySelector(".page-menu-btn"));

  // the unchanged/fallback row
  const fallback = cells(rows[1]);
  check("an unchanged row is marked as such", rows[1].classList.contains("unchanged"));
  check("...mutes its Simplification cell", fallback[2].classList.contains("cell-muted"), fallback[2].className);
  check("...keeps the input readable rather than greying the whole row", !fallback[1].classList.contains("cell-muted"));
  check(
    "...doesn't print the input a second time, which is what the Input column is for",
    !fallback[2].textContent.includes(FALLBACK_ENTRY.output),
    fallback[2].textContent
  );
  check(
    "...says in words why the original was kept",
    /Kept the original — the model's edit changed no words\./.test(fallback[2].textContent),
    fallback[2].textContent
  );
  check(
    "...and carries the raw reason code for whoever needs it",
    /no_meaningful_change/.test(fallback[2].querySelector(".entry-meta").title),
    fallback[2].querySelector(".entry-meta").title
  );
  check("...mutes its metrics, since nothing was simplified", fallback[4].classList.contains("cell-muted") && fallback[5].classList.contains("cell-muted"));
  check(
    "...and reports no figures at all, rather than measuring the input against itself",
    fallback[4].textContent === "—" && fallback[5].textContent === "—",
    `${fallback[4].textContent} | ${fallback[5].textContent}`
  );
  check(
    "...and shows the answer the model actually gave, which the reason only describes",
    fallback[2].querySelector(".entry-meta-rejected").textContent === FALLBACK_ENTRY.modelResult,
    fallback[2].textContent
  );

  // The row's text is the input, so the rejected answer is the only record of what the model said.
  const artifactCell = cells(
    context.buildRow(
      {
        input: "Municipal recycling policy revisions",
        modelResult: "Other websites",
        output: "Municipal recycling policy revisions",
        changed: false,
        cached: false,
        fallbackReason: "corpus_artifact",
        requestTimeMs: 1028,
      },
      PAGE,
      0
    )
  )[2];
  check(
    "a corpus-artifact row shows the boilerplate the model answered with",
    artifactCell.querySelector(".entry-meta-rejected").textContent === "Other websites",
    artifactCell.textContent
  );

  // Pre-`modelResult` entries: quoting the served text would claim the model returned the input.
  const preModelResultCell = cells(
    context.buildRow(
      { input: FALLBACK_ENTRY.input, output: FALLBACK_ENTRY.input, changed: false, fallbackReason: "hallucination" },
      PAGE,
      0
    )
  )[2];
  check(
    "a row recorded before model_result existed shows no rejected answer",
    !preModelResultCell.querySelector(".entry-meta-rejected"),
    preModelResultCell.textContent
  );

  // older builds recorded no reason
  const noReasonCell = cells(
    context.buildRow({ input: FALLBACK_ENTRY.input, output: FALLBACK_ENTRY.input, changed: false }, PAGE, 0)
  )[2];
  check(
    "an unchanged row with no recorded reason still says the original was kept",
    noReasonCell.textContent.trim() === "Kept the original.",
    noReasonCell.textContent
  );

  const longCell = cells(
    context.buildRow(
      {
        input: "Collections start at 6 a.m.",
        modelResult: `Collections ${"start ".repeat(60)}`,
        output: "Collections start at 6 a.m.",
        changed: false,
        fallbackReason: "degenerate_repetition",
        requestTimeMs: 9000,
      },
      PAGE,
      0
    )
  )[2];
  // shown in full, not truncated to 160 chars, so it can be checked against the rejection reason
  const rejected = longCell.querySelector(".entry-meta-rejected");
  check(
    "a degenerate answer is shown in full rather than cut short",
    rejected.textContent === `Collections ${"start ".repeat(60)}`.trim(),
    `${rejected.textContent.length} chars`
  );
  check(
    "...and needs no label or quotation marks around it",
    !/Model returned|\u201c|\u201d/.test(rejected.textContent),
    rejected.textContent.slice(0, 40)
  );

  // the cached row
  const cached = cells(rows[2]);
  check("a cached row says so instead of reporting a latency", cached[3].textContent === "Cached", cached[3].textContent);
  check("...muted, so it doesn't read as a very fast generation", cached[3].classList.contains("cell-muted"));
  check("...and explains itself on hover", /cache/i.test(cached[3].title), cached[3].title);

  // --- a page written by an older build
  const legacyCells = Array.from(legacyBlock.querySelectorAll("tbody tr td"));
  check(
    "a page recorded before the rename still renders its text",
    legacyCells[1].textContent.startsWith("Residents are requested") &&
      legacyCells[2].textContent.startsWith("Residents should put"),
    legacyCells.map((c) => c.textContent).join(" | ")
  );
  check(
    "...and says what it doesn't know rather than inventing it",
    /weren't recorded/.test(lines(legacyBlock)[0]),
    lines(legacyBlock)[0]
  );
  check(
    "...leaving the skipped count off entirely, since none was recorded",
    !/skipped/.test(lines(legacyBlock)[1]),
    lines(legacyBlock)[1]
  );

  // --- exports
  const pageExport = context.pageExport(PAGE);
  check("a page export leads with the session fields", Object.keys(pageExport)[0] === "sessionId", Object.keys(pageExport).join(","));
  check("...carries the run's totals and elapsed time", !!pageExport.stats && pageExport.totalTimeMs === 4700);
  check("...and ends with the entries", Object.keys(pageExport).pop() === "entries", Object.keys(pageExport).join(","));

  const entryExport = context.entryExport(PAGE, FALLBACK_ENTRY);
  check("a single-entry export keeps the page it came from", entryExport.sessionId === "session-1" && entryExport.title === PAGE.title);
  check("...and the model that produced it", entryExport.method === "fine_tuned_seq2seq" && entryExport.modelId === PAGE.modelId);
  check(
    "...but no whole-run statistics, which would describe rows it doesn't hold",
    entryExport.stats === undefined && entryExport.totalTimeMs === undefined && entryExport.status === undefined,
    Object.keys(entryExport).join(",")
  );
  check("...and holds exactly the one entry", entryExport.entries.length === 1 && entryExport.entries[0] === FALLBACK_ENTRY);

  console.log(failures === 0 ? "\nall history view checks passed" : `\n${failures} check(s) failed`);
  process.exit(failures === 0 ? 0 : 1);
})();
