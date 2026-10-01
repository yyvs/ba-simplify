// Reports page (reports.html + shared/reports.js): a kept run's block, its progress while
// still being logged, the stored evaluation, and deletion.
//
//   cd extension/tests && npm install && node reports-view.test.js
//
// Same harness as history-view.test.js, built from reports.html's markup.
//
// Covers: a report saved mid-run follows the log while the run is in it and keeps its
// last snapshot after; "complete" vs. "stopped" are distinct; the evaluation is stored,
// not recomputed.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { JSDOM } = require("jsdom");

const EXT = path.join(__dirname, "..");
const SHARED = path.join(EXT, "shared");
const SOURCES = ["model-labels.js", "model-selection.js", "analysis.js", "report-store.js", "reports.js"].map(
  (f) => fs.readFileSync(path.join(SHARED, f), "utf8")
);
const PAGE_HTML = fs
  .readFileSync(path.join(EXT, "reports.html"), "utf8")
  .replace(/<script[^>]*>[\s\S]*?<\/script>/g, "");

let failures = 0;
function check(label, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${ok || !detail ? "" : ` — ${detail}`}`);
  if (!ok) failures += 1;
}

function entry(input, output, changed = true) {
  return {
    input,
    modelResult: output,
    output,
    changed,
    cached: false,
    fallbackReason: changed ? null : "no_meaningful_change",
    requestTimeMs: 1200,
  };
}

// Run in progress: 40 items found, 24 accounted for. Progress is read from `stats`, not
// from the rows on hand.
const RUNNING_PAGE = {
  sessionId: "session-live",
  url: "https://example.com/recycling",
  title: "Recycling collections",
  timestamp: 1_700_000_000_000,
  method: "fine_tuned_seq2seq",
  modelKey: "finetuned",
  modelId: "yunvs/bart-base-wikilarge-simplification",
  granularity: "sentence_by_sentence",
  status: "completed",
  totalTimeMs: 12_000,
  stats: {
    totalItems: 40,
    skippedItems: 10,
    processedItems: 14,
    simplifiedItems: 13,
    unchangedItems: 1,
    cachedResponses: 0,
  },
  entries: [
    entry(
      "Residents are requested to position their receptacles at the kerbside.",
      "Residents should put their bins at the kerb."
    ),
    entry(
      "Collections will commence at approximately 6 a.m. on Tuesdays.",
      "Collections start at about 6 a.m. on Tuesdays."
    ),
  ],
};

function load({ reports = [], pages = [] } = {}) {
  const dom = new JSDOM(PAGE_HTML, { pretendToBeVisual: true, url: "chrome-extension://test/reports.html" });
  const { window } = dom;
  const store = { simplifyReports: reports, simplifyHistory: pages };
  const listeners = [];

  const context = {
    window,
    document: window.document,
    // read by requestedReportId to open the page at one report
    location: window.location,
    Blob: window.Blob,
    URL: window.URL,
    crypto: { randomUUID: () => "0000-test" },
    navigator: { clipboard: { writeText: () => Promise.resolve() } },
    confirm: () => true,
    setTimeout,
    clearTimeout,
    console,
    chrome: {
      runtime: { sendMessage: () => Promise.resolve({ ok: false }) },
      storage: {
        local: {
          get: (key) => Promise.resolve(key in store ? { [key]: store[key] } : {}),
          set: (values) => {
            Object.assign(store, values);
            listeners.forEach((fn) =>
              fn(Object.fromEntries(Object.keys(values).map((k) => [k, { newValue: values[k] }])), "local")
            );
            return Promise.resolve();
          },
          remove: (key) => {
            delete store[key];
            listeners.forEach((fn) => fn({ [key]: {} }, "local"));
            return Promise.resolve();
          },
        },
        onChanged: { addListener: (fn) => listeners.push(fn) },
      },
    },
  };
  context.globalThis = context;
  vm.createContext(context);
  SOURCES.forEach((source) => vm.runInContext(source, context));
  return { context, window, store };
}

const tick = (ms = 0) => new Promise((resolve) => setTimeout(resolve, ms));
const blocks = (window) => Array.from(window.document.querySelectorAll(".report-block"));
const lines = (block) => Array.from(block.querySelectorAll(".page-aggregate")).map((el) => el.textContent);

(async () => {
  // --- a report made from a run
  {
    const { context, store } = load({ pages: [RUNNING_PAGE] });
    const saved = await vm.runInContext("saveReportForPage", context)(RUNNING_PAGE);
    check("saving a run writes a report", store.simplifyReports.length === 1);
    check(
      "...named for the page, the model and the scope",
      saved.report.name === "Recycling collections · Fine-tuned model · sentence by sentence",
      saved.report.name
    );
    check("...pointing back at the run it came from", saved.report.sourceSessionId === "session-live");
    check("...with its own copy of the rows", saved.report.entries.length === 2 && saved.report.entries !== RUNNING_PAGE.entries);
    check("...and no evaluation until one is asked for", saved.report.evaluation === null);

    const again = await vm.runInContext("saveReportForPage", context)(RUNNING_PAGE);
    check("saving the same run again updates its report rather than adding a second", store.simplifyReports.length === 1 && again.already === true);
  }

  // --- progress
  {
    const { context } = load();
    const progress = vm.runInContext("reportProgress", context);
    const running = progress({ stats: RUNNING_PAGE.stats }, true);
    check("a run still in the log reports how far it has got", running.state === "running" && running.done === 24 && running.total === 40, JSON.stringify(running));
    const gone = progress({ stats: RUNNING_PAGE.stats }, false);
    check("...and the same counts with the run gone from the log say it stopped", gone.state === "stopped", gone.state);
    const done = progress({ stats: { ...RUNNING_PAGE.stats, processedItems: 30 } }, false);
    check("every item accounted for is complete, whatever the log holds", done.state === "complete" && done.done === 40, JSON.stringify(done));
    const old = progress({ stats: { simplifiedItems: 2, unchangedItems: 0, cachedResponses: 0 } }, false);
    check(
      "a run that recorded no item total is not given an invented one",
      old.total === null && old.state === "unknown",
      JSON.stringify(old)
    );
  }

  // --- the rendered block
  {
    const { context, window, store } = load({ pages: [RUNNING_PAGE] });
    await vm.runInContext("saveReportForPage", context)(RUNNING_PAGE);
    await tick();
    const block = blocks(window)[0];
    check("the page renders one block per report", blocks(window).length === 1);
    check("...opening the newest one", block.open);
    check(
      "...with a progress line reading against the run's own counts",
      lines(block).some((l) => l === "Running — 24 of 40 items"),
      lines(block).join(" | ")
    );
    check(
      "...a bar filled to that fraction",
      block.querySelector(".report-bar-fill").style.width === "60%",
      block.querySelector(".report-bar-fill").style.width
    );
    check(
      "...the same selection line the History page shows",
      lines(block).some((l) => /Fine-tuned seq2seq · yunvs\/bart-base-wikilarge-simplification · Sentence by sentence/.test(l)),
      lines(block).join(" | ")
    );
    check(
      "...the run's outcome",
      lines(block).some((l) => /13 simplified · 1 unchanged · 10 items skipped/.test(l)),
      lines(block).join(" | ")
    );
    check("...and an evaluation that hasn't been run yet", lines(block).includes("Not evaluated yet."));

    // --- the evaluation
    const runBtn = block.querySelector(".report-evaluation-head button");
    check("the evaluation is offered as an action, not run on load", runBtn.textContent === "Run evaluation");
    runBtn.dispatchEvent(new window.Event("click"));
    await tick(10);
    const stored = store.simplifyReports[0].evaluation;
    check("running it stores the result on the report", !!stored && Array.isArray(stored.sections), JSON.stringify(stored && Object.keys(stored)));
    check("...saying how many rows it covered", stored.entryCount === 2, String(stored && stored.entryCount));
    check(
      "...as the three metric sections",
      stored.sections.map((s) => s.title).join(" | ") === "Readability | Structure | Rewriting",
      stored.sections.map((s) => s.title).join(" | ")
    );
    check(
      "...with a Flesch-Kincaid row that has real numbers in it",
      /^\d+\.\d$/.test(stored.sections[0].rows[0].cells[1]) && /^\d+\.\d$/.test(stored.sections[0].rows[0].cells[2]),
      stored.sections[0].rows[0].cells.join(" | ")
    );

    await tick(10);
    const rerendered = blocks(window)[0];
    check(
      "the stored result is read back rather than recomputed",
      rerendered.querySelector(".report-evaluation-head button").textContent === "Re-run evaluation",
      rerendered.querySelector(".report-evaluation-head button").textContent
    );
    check(
      "...and rendered as tables",
      rerendered.querySelectorAll(".analyze-table").length === 3,
      String(rerendered.querySelectorAll(".analyze-table").length)
    );
  }

  // --- a report whose run has left the log
  {
    const { context, window, store } = load({ pages: [RUNNING_PAGE] });
    await vm.runInContext("saveReportForPage", context)(RUNNING_PAGE);
    await tick();
    // run progresses, then falls out of the rolling log
    store.simplifyHistory = [{ ...RUNNING_PAGE, stats: { ...RUNNING_PAGE.stats, processedItems: 30, simplifiedItems: 28 } }];
    await vm.runInContext("render", context)();
    await tick(10);
    check(
      "a report following a live run picks up the rows the run added",
      lines(blocks(window)[0]).some((l) => l === "Complete — 40 items"),
      lines(blocks(window)[0]).join(" | ")
    );

    store.simplifyHistory = [];
    await vm.runInContext("render", context)();
    await tick(10);
    check(
      "...and keeps that snapshot once the run is gone from the log",
      lines(blocks(window)[0]).some((l) => l === "Complete — 40 items"),
      lines(blocks(window)[0]).join(" | ")
    );
    check("the report itself is untouched by the log emptying", store.simplifyReports.length === 1);

    // --- deletion
    await vm.runInContext("deleteReport", context)(store.simplifyReports[0].reportId);
    await tick(10);
    check("deleting a report removes it", store.simplifyReports.length === 0);
    check(
      "...and the page says how to make one instead of showing nothing",
      /Save as report/.test(window.document.querySelector(".empty").textContent),
      window.document.querySelector(".empty") && window.document.querySelector(".empty").textContent
    );
  }

  // --- a comparison, as one report
  // The log holds a comparison's arm-runs as unrelated pages. The report holds entries
  // once, folds in a column as each run finishes, and scores every column against the one
  // original.
  //
  // Two cuts: three sections through `document` whole, and through `finetuned` both whole
  // and one leaf element at a time, so one arm answers twice.
  {
    const { context, window, store } = load();
    const save = vm.runInContext("saveComparisonRun", context);

    const inputs = [
      "Residents are requested to position their receptacles at the kerbside before six in the morning.",
      "Collections will commence at approximately six in the morning on every Tuesday throughout the year.",
      "Materials which are contaminated with foodstuffs cannot be reprocessed and must be presented separately.",
    ];
    const result = (output, extra = {}) => ({
      output,
      modelResult: output,
      changed: true,
      cached: false,
      fallbackReason: null,
      requestTimeMs: 900,
      ...extra,
    });
    const PLAIN = [
      "Put your bins at the kerb before six.",
      "Trucks come at six in the morning each Tuesday.",
      "Dirty food waste cannot be reused. Put it out on its own.",
    ];
    const STIFF = [
      "Residents must position their receptacles kerbside before six.",
      "Collections commence approximately six in the morning every Tuesday.",
      "Contaminated materials cannot be reprocessed and require separate presentation.",
    ];
    const page = { url: "https://example.com/recycling", title: "Recycling collections", timestamp: 1_700_000_000_000 };
    const CUTS = ["whole_sections", "sentence_by_sentence"];
    const arm = (label, index, modelKey, modelId, granularity) => ({
      label,
      index,
      modelKey,
      modelId,
      method: "fine_tuned_seq2seq",
      granularity,
    });
    const run = (totalTimeMs, requests) => ({
      status: "completed",
      totalTimeMs,
      stats: { totalItems: 3, requests, processedItems: 3, simplifiedItems: 3, unchangedItems: 0, cachedResponses: 0 },
    });
    const A = arm("A", 0, "finetuned", "yunvs/bart-base-wikilarge-simplification", "sentence_by_sentence");
    const B = arm("B", 1, "document", "yunvs/bart-base-dwikipedia-simplification-full", "whole_sections");
    const write = (cut, armMeta, runMeta, results, runStatus) =>
      save({ group: "page-1", page, unitGranularity: "whole_sections", cutOrder: CUTS, models: 3, inputs, cut, arm: armMeta, run: runMeta, results, runStatus });

    // sections cut first: the entries are its units
    await write("whole_sections", A, run(9_000, 3), PLAIN.map((out) => result(out)), "running");
    await tick(10);
    check("an arm-run of a comparison writes a report", store.simplifyReports.length === 1);
    let report = store.simplifyReports[0];
    check("...marked as a comparison", report.kind === "comparison" && !!report.arms);
    check("...keyed by the run it belongs to", report.comparisonGroup === "page-1");
    check("...holding the entries once", JSON.stringify(report.entries.map((e) => e.input)) === JSON.stringify(inputs));
    // shape: cut, then arm
    check("...with that run's answers filed under its cut and its arm", report.entries.every((e) => !!e.results.whole_sections.A));
    check("...naming the cut the entries are the units of", report.unitGranularity === "whole_sections");
    check("...and both cuts the run will use", JSON.stringify(report.cutOrder) === JSON.stringify(CUTS));
    check("...and how many arm-runs it set out to make", report.models === 3);
    check("...saying the run is still going", report.status === "running", report.status);

    await write("whole_sections", B, run(21_000, 3), STIFF.map((out) => result(out)), "running");
    // Same arm, other cut, third entry lost. Answers differ slightly from the sections
    // cut, as they would for real input split differently.
    const SPLIT = [
      "Put your bins out at the kerb before six in the morning.",
      "The trucks come at about six in the morning on each Tuesday.",
    ];
    await write("sentence_by_sentence", A, run(15_000, 7), [result(SPLIT[0]), result(SPLIT[1]), null], "completed");
    await tick(10);
    check("every arm-run folds into the same report", store.simplifyReports.length === 1);
    report = store.simplifyReports[0];
    check("...as two arms, not three", JSON.stringify(Object.keys(report.arms).sort()) === '["A","B"]');
    // one arm, two runs, separated by cut
    check(
      "...one of which ran in both cuts",
      JSON.stringify(Object.keys(report.arms.A.cuts).sort()) === '["sentence_by_sentence","whole_sections"]',
      JSON.stringify(Object.keys(report.arms.A.cuts))
    );
    check("...and one in only the cut it can read", JSON.stringify(Object.keys(report.arms.B.cuts)) === '["whole_sections"]');
    check("...with that arm's per-cut timings kept apart", report.arms.A.cuts.whole_sections.totalTimeMs === 9_000 && report.arms.A.cuts.sentence_by_sentence.totalTimeMs === 15_000);
    check(
      "an entry holds every column's answer, by cut then arm",
      !!report.entries[0].results.whole_sections.A &&
        !!report.entries[0].results.whole_sections.B &&
        !!report.entries[0].results.sentence_by_sentence.A,
      JSON.stringify(Object.keys(report.entries[0].results))
    );
    // a lost request stays a gap: rows align by position, so filling it would shift answers
    check(
      "...leaving a gap where a run lost one",
      !(report.entries[2].results.sentence_by_sentence || {}).A && !!report.entries[2].results.whole_sections.A,
      JSON.stringify(Object.keys(report.entries[2].results))
    );
    check("...and the run now complete", report.status === "completed", report.status);
    check(
      "...named for the models and both cuts",
      report.name === "Recycling collections · 2 models compared · sections and sentences",
      report.name
    );

    // --- the rendered block
    const block = blocks(window)[0];
    const text = lines(block).join(" | ");
    check("the block names each column", /A · sections/.test(text) && /B · sections/.test(text) && /A · sentences/.test(text), text);
    // A's two columns are the same model; the id and input description tell them apart
    check(
      "...with the id of the artifact that answered",
      /yunvs\/bart-base-wikilarge-simplification/.test(text) && /yunvs\/bart-base-dwikipedia-simplification-full/.test(text),
      text
    );
    check(
      "...and what each run was given",
      /each section whole/.test(text) && /each section split into sentences/.test(text) && /one leaf element at a time/.test(text),
      text
    );
    check(
      "...with progress counted in models/methods as well as entries",
      lines(block).some((l) => l === "Complete — 3 models/methods × 3 entries"),
      text
    );

    // --- the evaluation
    const runBtn = block.querySelector(".report-evaluation-head button");
    runBtn.dispatchEvent(new window.Event("click"));
    await tick(10);
    const stored = store.simplifyReports[0].evaluation;
    check("evaluating a comparison stores the result on it", !!stored && Array.isArray(stored.sections));
    check(
      "...as the sections a comparison has rather than a single run's three",
      stored.sections.map((s) => s.title).join(" | ") === "Readability | Structure | Rewriting | What each model/method did",
      stored.sections.map((s) => s.title).join(" | ")
    );
    // one column per (cut, arm), all against the one original
    check(
      "...with one column per arm-run, against one original",
      JSON.stringify(stored.sections[0].columns) === '["Metric","Original","A · sections","B · sections","A · sentences"]',
      JSON.stringify(stored.sections[0].columns)
    );
    // Scored only over entries every column answered; per-column subsets would compare different text.
    check("...over the entries every model/method answered", stored.entryCount === 2, String(stored.entryCount));
    check("...saying so", /2 entries every model\/method answered/.test(stored.sections[0].note), stored.sections[0].note);
    check("...and how many of them that was", stored.armCount === 3, String(stored.armCount));
    // model ids appear once, above the tables; no legend table
    check(
      "...naming each model id above the tables rather than in a legend table",
      /yunvs\/bart-base-wikilarge-simplification/.test(text) &&
        /yunvs\/bart-base-dwikipedia-simplification-full/.test(text) &&
        !stored.sections.some((s) => s.title === "Columns"),
      stored.sections.map((s) => s.title).join(" | ")
    );

    const fkgl = stored.sections[0].rows[0];
    check("...a Flesch-Kincaid row with a number per column", fkgl.cells.slice(1).every((c) => /^\d+\.\d$/.test(c)), fkgl.cells.join(" | "));
    check(
      "...and the better column of the row marked, once",
      fkgl.verdicts.filter((v) => v === "best").length === 1,
      JSON.stringify(fkgl.verdicts)
    );
    check(
      "...which is the plainest of the three, not the stiffest",
      fkgl.verdicts[2] === "best" &&
        Number(fkgl.cells[2]) < Number(fkgl.cells[3]) &&
        Number(fkgl.cells[2]) < Number(fkgl.cells[4]),
      fkgl.cells.join(" | ")
    );
    // A shared best is marked on every column holding it (e.g. one model over both cuts on
    // single-element sections answers identically; ties once left the table unmarked).
    const bestArms = vm.runInContext("bestComparisonArms", context);
    check(
      "a best two model/methods share is marked on both",
      JSON.stringify(bestArms([10.7, 11.9, 10.7], "lower")) === "[0,2]",
      JSON.stringify(bestArms([10.7, 11.9, 10.7], "lower"))
    );
    check(
      "...and on the one that holds it alone",
      JSON.stringify(bestArms([11.9, 10.7, 11.6], "lower")) === "[1]",
      JSON.stringify(bestArms([11.9, 10.7, 11.6], "lower"))
    );
    check(
      "...and a row where all of them tie marks all of them",
      JSON.stringify(bestArms([10.7, 10.7, 10.7], "lower")) === "[0,1,2]",
      JSON.stringify(bestArms([10.7, 10.7, 10.7], "lower"))
    );

    // count rows have no better direction: fewer words is not a win
    const words = stored.sections[1].rows[1];
    check("a row with no better direction marks no column", words.verdicts.every((v) => v === null), JSON.stringify(words.verdicts));

    await tick(10);
    const rerendered = blocks(window)[0];
    check(
      "the stored comparison is read back rather than recomputed",
      rerendered.querySelector(".report-evaluation-head button").textContent === "Re-run evaluation"
    );
    check("...and rendered as one table per section", rerendered.querySelectorAll(".analyze-table").length === 4, String(rerendered.querySelectorAll(".analyze-table").length));
    // three arm-runs overflow the block: each table scrolls with the metric column pinned
    check(
      "...each in a scroller, so the metric column stays readable however many ran",
      rerendered.querySelectorAll(".analyze-table-scroll > .analyze-table").length === 4,
      String(rerendered.querySelectorAll(".analyze-table-scroll > .analyze-table").length)
    );
    check(
      "...with the winning cell marked in the table too",
      rerendered.querySelectorAll(".analyze-table .metric-best").length > 0,
      String(rerendered.querySelectorAll(".analyze-table .metric-best").length)
    );
  }

  // --- opening the page at one report
  // A run leaves its page unchanged, so it opens this page at its report via the fragment.
  {
    const { context, window, store } = load({ pages: [RUNNING_PAGE] });
    await vm.runInContext("saveReportForPage", context)(RUNNING_PAGE);
    await vm.runInContext("saveComparisonRun", context)({
      group: "page-9",
      page: { url: "https://example.com/x", title: "Another page", timestamp: 1 },
      unitGranularity: "sentence_by_sentence",
      cutOrder: ["sentence_by_sentence"],
      models: 1,
      inputs: ["One sentence that was simplified."],
      cut: "sentence_by_sentence",
      arm: { label: "A", index: 0, modelKey: "finetuned", modelId: "yunvs/bart-base-wikilarge-simplification", method: "fine_tuned_seq2seq", granularity: "sentence_by_sentence" },
      run: { status: "completed", totalTimeMs: 10, stats: { totalItems: 1, requests: 1, processedItems: 1, simplifiedItems: 1, unchangedItems: 0, cachedResponses: 0 } },
      results: [{ output: "One short sentence.", modelResult: "One short sentence.", changed: true, cached: false, fallbackReason: null, requestTimeMs: 10 }],
      runStatus: "completed",
    });
    await tick(10);

    check("with no fragment, the newest report is the one open", blocks(window)[0].open === true);
    const kept = store.simplifyReports.find((r) => r.sourceSessionId === "session-live");
    check("...and every block carries an anchor of its own", !!window.document.getElementById(`report-${kept.reportId}`));

    window.location.hash = `#report-${kept.reportId}`;
    await vm.runInContext("render", context)();
    await tick(10);
    const opened = window.document.getElementById(`report-${kept.reportId}`);
    check("a fragment naming a report opens that one", opened.open === true);
    check("...and marks it as the one the page was opened for", opened.classList.contains("report-block-focused"));
    check("...leaving the newest one closed", blocks(window)[0].open === false || blocks(window)[0] === opened);

    // comparisons are addressed by group id: the report id is assigned in the background
    // page, which the run never sees
    window.location.hash = "#comparison-page-9";
    await vm.runInContext("render", context)();
    await tick(10);
    const comparison = store.simplifyReports.find((r) => r.comparisonGroup === "page-9");
    check(
      "a fragment naming a comparison's group opens that comparison",
      window.document.getElementById(`report-${comparison.reportId}`).classList.contains("report-block-focused")
    );
  }

  console.log(failures === 0 ? "\nall reports view checks passed" : `\n${failures} check(s) failed`);
  process.exit(failures === 0 ? 0 : 1);
})();
