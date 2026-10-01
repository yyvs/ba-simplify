// Comparing several models on one page: the panel that sets one up, and the run itself.
//
//   cd extension/tests && npm install && node compare-models.test.js
//
// Inputs are constant across models: the page is read once per granularity and every
// model gets the same strings. The panel picks granularity before models, and the run
// collects once and reuses.
//
// It is an evaluation, not a simplification: the page is never written to (that would
// destroy the text the next model reads). Also covered: the one-run-at-a-time lock,
// models in sequence, stopping, and a finish notification outside the tab.
//
// content.js runs in a vm context after the shared files manifest.json lists before it.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { JSDOM } = require("jsdom");

const EXT = path.join(__dirname, "..");
// exactly what manifest.json lists, so the harness runs what the browser runs
const CONTENT_SCRIPTS = JSON.parse(fs.readFileSync(path.join(EXT, "manifest.json"), "utf8")).content_scripts[0].js;

const results = [];
const check = (name, condition, detail = "") => results.push([name, !!condition, detail]);
const settle = (ms = 30) => new Promise((resolve) => setTimeout(resolve, ms));

const PAGE_BODY = `
  <h1>Recycling</h1>
  <p>Residents are requested to position their receptacles at the kerbside before six.</p>
  <p>Collections will commence at approximately six in the morning on every Tuesday.</p>
`;

const HEALTH = {
  status: "ok",
  model_loaded: true,
  models_loaded: ["online", "finetuned", "document", "llm_7b", "llm_doc_7b"],
  model_names: {
    online: "eilamc14/bart-large-text-simplification",
    finetuned: "yunvs/bart-base-wikilarge-simplification",
    document: "yunvs/bart-base-dwikipedia-simplification-full",
    llm_7b: "qwen2.5:7b-instruct-q4_K_M",
    llm_doc_7b: "qwen2.5:7b-instruct-q4_K_M",
  },
  audience_models: ["llm_7b", "llm_doc_7b"],
  granularities: {
    online: "sentence_by_sentence",
    finetuned: "sentence_by_sentence",
    llm_7b: "sentence_by_sentence",
    document: "whole_sections",
    llm_doc_7b: "whole_sections",
  },
  methods: {
    online: "fine_tuned_seq2seq",
    finetuned: "fine_tuned_seq2seq",
    document: "fine_tuned_seq2seq",
    llm_7b: "prompted_llm",
    llm_doc_7b: "prompted_llm",
  },
  audiences: [{ value: "non_native_speakers", label: "Non-native speakers" }],
  default_audience: "non_native_speakers",
};

function loadPage({ stored = { simplifierGranularity: "sentence_by_sentence" }, hold = false } = {}) {
  const dom = new JSDOM(`<!doctype html><html><head><title>Recycling</title></head><body>${PAGE_BODY}</body></html>`, {
    pretendToBeVisual: true,
    url: "https://en.wikipedia.org/wiki/Recycling",
  });
  const { window } = dom;
  const sent = [];
  // by-model requests with their resolvers; held open with `hold` to observe a run mid-flight
  const pending = [];
  const store = { ...stored };

  const answer = (message, callback) =>
    callback({
      ok: true,
      model: message.model,
      audience: "non_native_speakers",
      data: { simplified: `${message.text} [${message.model}]`, cached: false, model_result: `${message.text} [${message.model}]` },
    });

  const context = {
    window,
    document: window.document,
    location: window.location,
    history: window.history,
    Node: window.Node,
    MutationObserver: window.MutationObserver,
    getComputedStyle: window.getComputedStyle.bind(window),
    performance: window.performance,
    setTimeout: window.setTimeout.bind(window),
    clearTimeout: window.clearTimeout.bind(window),
    setInterval: window.setInterval.bind(window),
    clearInterval: window.clearInterval.bind(window),
    console: { ...console, error() {}, log() {}, warn() {} },
    crypto: { randomUUID: () => "test-page-session" },
    fetch: () => Promise.reject(new Error("the content script must not reach the network directly")),
    chrome: {
      runtime: {
        onMessage: { addListener: (fn) => (context.onMessage = fn) },
        lastError: null,
        sendMessage(message, callback) {
          sent.push(message);
          if (message.cmd === "health") {
            if (callback) callback({ ok: true, health: HEALTH });
            return;
          }
          if (message.cmd === "claimRun") {
            if (callback) callback({ ok: true, holder: null });
            return;
          }
          if (message.cmd === "fetchSimplifyWithModel") {
            if (hold) pending.push({ message, callback });
            else answer(message, callback);
            return;
          }
          if (callback) callback({ ok: true });
        },
      },
      storage: {
        local: {
          get: (keys) => {
            const names = Array.isArray(keys) ? keys : [keys];
            const out = {};
            names.forEach((k) => {
              if (k in store) out[k] = store[k];
            });
            return Promise.resolve(out);
          },
          set: (values) => {
            Object.assign(store, values);
            return Promise.resolve();
          },
        },
        onChanged: { addListener() {} },
      },
    },
  };
  context.globalThis = context;
  vm.createContext(context);
  CONTENT_SCRIPTS.forEach((f) => vm.runInContext(fs.readFileSync(path.join(EXT, f), "utf8"), context));

  const shadow = () => {
    const host = window.document.getElementById("simplify-compare-picker");
    return host ? host.shadowRoot : null;
  };

  return {
    window,
    sent,
    pending,
    run: (expr) => vm.runInContext(expr, context),
    dispatch: (message, respond = () => {}) => context.onMessage(message, {}, respond),
    shadow,
    rows: () => Array.from(shadow().querySelectorAll(".option[data-compare-model]")),
    segments: () => Array.from(shadow().querySelectorAll(".segment[data-compare-granularity]")),
    runButton: () => shadow().getElementById("simplify-compare-run"),
    footer: () => shadow().querySelector(".footer-note").textContent,
    // answers everything currently held
    answerHeld: () => pending.splice(0).forEach(({ message, callback }) => answer(message, callback)),
    modelRequests: () => sent.filter((m) => m.cmd === "fetchSimplifyWithModel"),
    flushes: () => sent.filter((m) => m.cmd === "recordHistory"),
    comparisons: () => sent.filter((m) => m.cmd === "recordComparison"),
    notice: () => {
      const box = window.document.getElementById("simplify-notice");
      return box ? box.textContent : "";
    },
    simplifiedCount: () => window.document.querySelectorAll("[data-original-html]").length,
  };
}

async function openPanel(ctx) {
  ctx.dispatch({ cmd: "showCompareModels" });
  await settle();
}

async function granularityComesFirstAndFiltersTheModels() {
  const ctx = loadPage();
  await openPanel(ctx);
  check("the menu opens the comparison panel", !!ctx.shadow());
  check("...on a granularity control", ctx.segments().length === 2);
  check("...saying what it decides", /what every model is given/i.test(ctx.shadow().textContent), ctx.shadow().textContent.slice(0, 200));

  const sentenceModels = ctx.rows().map((r) => r.dataset.compareModel);
  check(
    "sentence scope lists the models that read a sentence",
    JSON.stringify(sentenceModels) === JSON.stringify(["finetuned", "llm_7b", "llm_3b", "online"]),
    sentenceModels.join()
  );
  // shown with a reason rather than hidden, so it doesn't look nonexistent
  const unloaded = ctx.rows().find((r) => r.dataset.compareModel === "llm_3b");
  check("...with one the backend hasn't loaded shown but not selectable", unloaded.disabled === true);
  check("...saying why", /not loaded/i.test(unloaded.textContent), unloaded.textContent);

  // tick sections, untick sentences: sections cut alone
  ctx.segments().find((s) => s.dataset.compareGranularity === "whole_sections").click();
  await settle();
  ctx.segments().find((s) => s.dataset.compareGranularity === "sentence_by_sentence").click();
  await settle();
  const sectionModels = ctx.rows().map((r) => r.dataset.compareModel);
  // Every key: the backend splits a section into sentences for a sentence-scope model and
  // rejoins them, so "sections, read sentence by sentence" is a valid arm. Each family's
  // own reading of the chosen unit comes first.
  check(
    "switching to whole sections lists every model that can read a section",
    JSON.stringify(sectionModels) ===
      JSON.stringify(["document", "finetuned", "llm_doc_7b", "llm_7b", "llm_doc_3b", "llm_3b", "online"]),
    sectionModels.join()
  );
  // the comparison checkpoint has no document-scope counterpart, so it reads sections sentence by sentence
  check("...including the family that has no whole-sections model", sectionModels.includes("online"));

  const readsWhole = ctx.rows().find((r) => r.dataset.compareModel === "document");
  const readsSplit = ctx.rows().find((r) => r.dataset.compareModel === "finetuned");
  check("...saying which of them reads a section whole", /each section whole/i.test(readsWhole.textContent), readsWhole.textContent);
  check(
    "...and which splits it into sentences",
    /each section split into sentences/i.test(readsSplit.textContent),
    readsSplit.textContent
  );
  // both rows share a family name; the reading is what tells them apart
  check("...which is what tells two arms of one family apart", readsWhole.textContent !== readsSplit.textContent);
}

async function severalModelsCanBeChosen() {
  const ctx = loadPage();
  await openPanel(ctx);
  check("nothing is selected to begin with", ctx.rows().every((r) => r.getAttribute("aria-checked") === "false"));
  check("...so the run button is disabled", ctx.runButton().disabled === true);
  check("...and says what is missing", /at least one model/i.test(ctx.footer()));

  ctx.rows()[0].click();
  await settle();
  ctx.rows()[1].click();
  await settle();
  const checked = ctx.rows().filter((r) => r.getAttribute("aria-checked") === "true");
  check("two models can be selected at once", checked.length === 2, `${checked.length}`);
  check("...which the button counts", /Compare 2 models$/.test(ctx.runButton().textContent), ctx.runButton().textContent);
  check("...and the footer explains the run", /prepared once/i.test(ctx.footer()), ctx.footer());

  ctx.rows()[0].click();
  await settle();
  check("clicking a selected model unselects it", ctx.rows().filter((r) => r.getAttribute("aria-checked") === "true").length === 1);

  // a key means a different model at the other granularity
  ctx.rows()[1].click();
  await settle();
  ctx.segments().find((s) => s.dataset.compareGranularity === "whole_sections").click();
  await settle();
  check("changing granularity clears the selection", ctx.rows().every((r) => r.getAttribute("aria-checked") === "false"));
}

async function eachModelGetsTheSameInputs() {
  const ctx = loadPage();
  await openPanel(ctx);
  ctx.rows()
    .filter((r) => ["finetuned", "llm_7b"].includes(r.dataset.compareModel))
    .forEach((r) => r.click());
  await settle();
  ctx.runButton().click();
  await settle(120);

  const requests = ctx.modelRequests();
  check("the panel closes on running", !ctx.shadow());
  check("both models were asked", new Set(requests.map((r) => r.model)).size === 2, JSON.stringify([...new Set(requests.map((r) => r.model))]));

  const byModel = {};
  requests.forEach((r) => (byModel[r.model] = (byModel[r.model] || []).concat(r.text)));
  const [first, second] = Object.keys(byModel);
  check("each model got the same inputs, in the same order", JSON.stringify(byModel[first]) === JSON.stringify(byModel[second]));
  check("...and there were inputs to send", byModel[first].length > 0, `${byModel[first].length}`);
  // headings are never sent (as on the simplify path)
  check("...with the heading left out", !byModel[first].some((t) => t === "Recycling"));

  check("the page itself is never rewritten", ctx.simplifiedCount() === 0);
  check("...and no simplify request was made against the selected model", !ctx.sent.some((m) => m.cmd === "fetchSimplify"));
}

async function eachModelIsRecordedAsItsOwnKeptRun() {
  const ctx = loadPage();
  await openPanel(ctx);
  ctx.rows()
    .filter((r) => ["finetuned", "llm_7b"].includes(r.dataset.compareModel))
    .forEach((r) => r.click());
  await settle();
  ctx.runButton().click();
  await settle(120);

  const flushes = ctx.flushes();
  check("each model's run is logged", flushes.length === 2, `${flushes.length}`);
  check("...under its own session id", new Set(flushes.map((f) => f.page.sessionId)).size === 2);
  check("...naming the model that produced it", flushes.every((f) => !!f.session.modelKey));
  // repo id / Ollama tag from /health at run start (was once null, leaving History rows
  // and JSON exports with only the extension's key)
  check(
    "...and the id of the artifact that answered",
    JSON.stringify(flushes.map((f) => f.session.modelId).sort()) ===
      JSON.stringify(["qwen2.5:7b-instruct-q4_K_M", "yunvs/bart-base-wikilarge-simplification"]),
    JSON.stringify(flushes.map((f) => f.session.modelId))
  );
  check("...marked as one arm of a comparison", flushes.every((f) => f.session.comparison && f.session.comparison.models === 2));
  check("...numbered within it", JSON.stringify(flushes.map((f) => f.session.comparison.modelIndex)) === "[1,2]");
  check("...and lettered", JSON.stringify(flushes.map((f) => f.session.comparison.label)) === '["A","B"]');
  // the run is kept as the comparison's own report (below), not N one-model reports
  check("...and not kept as a report of its own", flushes.every((f) => !f.report));
}

// Both cuts at once: the page read as sections and as leaf elements in one comparison,
// so both readings are columns of one table.
async function bothCutsRunInOneComparison() {
  const ctx = loadPage();
  await openPanel(ctx);
  // sentence is ticked from the stored selection; add sections
  ctx.segments().find((s) => s.dataset.compareGranularity === "whole_sections").click();
  await settle();
  const ticked = ctx.segments().filter((s) => s.getAttribute("aria-checked") === "true");
  check("both cuts can be ticked at once", ticked.length === 2, `${ticked.length}`);
  check(
    "...and the panel says what that means",
    /both answers/i.test(ctx.shadow().textContent),
    ctx.shadow().textContent.slice(0, 400)
  );

  ctx.rows().find((r) => r.dataset.compareModel === "finetuned").click();
  await settle();
  check(
    "a model that reads both is offered as reading both",
    /each section split into sentences, then one leaf element at a time/.test(
      ctx.rows().find((r) => r.dataset.compareModel === "finetuned").textContent
    ),
    ctx.rows().find((r) => r.dataset.compareModel === "finetuned").textContent
  );
  check("...and the button names the models, not the runs", /Compare 1 model, both ways/.test(ctx.runButton().textContent), ctx.runButton().textContent);

  ctx.runButton().click();
  await settle(150);

  const writes = ctx.comparisons();
  check("one model over two cuts is two arm-runs", writes.length === 2, `${writes.length}`);
  check("...the coarsest cut first", JSON.stringify(writes.map((w) => w.run.cut)) === '["whole_sections","sentence_by_sentence"]', JSON.stringify(writes.map((w) => w.run.cut)));
  // one arm, two runs: the letter follows the model
  check("...both under the one arm letter", writes.every((w) => w.run.arm.label === "A"));
  check("...both naming the cuts the run was given", writes.every((w) => JSON.stringify(w.run.cutOrder) === '["whole_sections","sentence_by_sentence"]'));

  // entries are the coarsest cut's units, so both runs answer the same rows
  check("...over one set of entries", JSON.stringify(writes[0].run.inputs) === JSON.stringify(writes[1].run.inputs));
  check("...which are the sections", writes[0].run.unitGranularity === "whole_sections");
  check("...and there was a section to compare", writes[0].run.inputs.length > 0, `${writes[0].run.inputs.length}`);

  // but the requests differ: whole section vs. one leaf element at a time
  const sectionRequests = ctx.modelRequests().filter((r) => /-sections-/.test(r.page));
  const sentenceRequests = ctx.modelRequests().filter((r) => /-sentences-/.test(r.page));
  check("the sections cut sends one request per section", sectionRequests.length === writes[0].run.inputs.length, `${sectionRequests.length}`);
  check("...and the sentence cut one per leaf element in it", sentenceRequests.length > sectionRequests.length, `${sentenceRequests.length} vs ${sectionRequests.length}`);
  check(
    "...which together are the same words the section was sent as",
    sentenceRequests.map((r) => r.text).join(" ") === sectionRequests.map((r) => r.text).join(" "),
    `${sentenceRequests.map((r) => r.text).join(" ")} || ${sectionRequests.map((r) => r.text).join(" ")}`
  );
  // separate session ids: own rows, counts and timings
  check("...logged as two runs of their own", new Set(ctx.flushes().map((f) => f.page.sessionId)).size === 2);

  // finer cut's answers are joined into the entry's answer and also kept in `parts`
  const split = writes[1].run.results[0];
  check("the sentence cut's answers are joined into the entry's answer", !!split && typeof split.output === "string" && split.output.includes(" "), JSON.stringify(split));
  check("...and kept individually beside it", Array.isArray(split.parts) && split.parts.length === sentenceRequests.length, JSON.stringify(split && split.parts && split.parts.length));
  check("...with the whole-section answer under the other cut", typeof writes[0].run.results[0].output === "string");
  check("...which was one request, so it was not taken apart", !writes[0].run.results[0].parts);

  check("the page itself is never rewritten", ctx.simplifiedCount() === 0);
}

// The History log has no shape for a comparison: arms appear there as unrelated pages.
async function theArmsAreCombinedIntoOneReport() {
  const ctx = loadPage();
  await openPanel(ctx);
  ctx.rows()
    .filter((r) => ["finetuned", "llm_7b"].includes(r.dataset.compareModel))
    .forEach((r) => r.click());
  await settle();
  ctx.runButton().click();
  await settle(120);

  const writes = ctx.comparisons();
  check("the comparison is written once per arm", writes.length === 2, `${writes.length}`);
  check("...all under one group id", new Set(writes.map((w) => w.run.group)).size === 1);
  check("...each naming its arm", JSON.stringify(writes.map((w) => w.run.arm.label)) === '["A","B"]');
  check("...with the id of the model that arm ran", writes.every((w) => !!w.run.arm.modelId), JSON.stringify(writes.map((w) => w.run.arm.modelId)));
  check("...its method", writes.every((w) => !!w.run.arm.method));
  // not necessarily the unit the page was cut into
  check("...and the unit it read", writes.every((w) => !!w.run.arm.granularity));
  check("...beside the unit the page was cut into", writes.every((w) => w.run.unitGranularity === "sentence_by_sentence"));
  check("...and how many arms the run set out to have", writes.every((w) => w.run.models === 2));

  // inputs held once, in page order; each arm's answers aligned by position
  const [a, b] = writes;
  check("the inputs are recorded once for the run", a.run.inputs.length > 0 && JSON.stringify(a.run.inputs) === JSON.stringify(b.run.inputs));
  check("...with one result slot per input", writes.every((w) => w.run.results.length === w.run.inputs.length));
  check("...holding what that arm answered", a.run.results.every((r) => r && typeof r.output === "string"));
  check("...and what the model itself produced", a.run.results.every((r) => typeof r.modelResult === "string"));
  check("...and how long it took", a.run.results.every((r) => typeof r.requestTimeMs === "number"));

  // run state, not arm state
  check("the run is still running while an arm remains", a.run.runStatus === "running", a.run.runStatus);
  check("...and complete once the last one lands", b.run.runStatus === "completed", b.run.runStatus);

  const done = ctx.sent.find((m) => m.cmd === "comparisonFinished");
  check("finishing is announced outside the tab", !!done);
  check("...saying how many models ran", done && done.models === 2);
  check("...and that it wasn't stopped", done && done.stopped === false);
  check("the lock is given back", ctx.sent.some((m) => m.cmd === "releaseRun"));
}

async function modelsRunOneAfterAnother() {
  const ctx = loadPage({ hold: true });
  await openPanel(ctx);
  ctx.rows()
    .filter((r) => ["finetuned", "llm_7b"].includes(r.dataset.compareModel))
    .forEach((r) => r.click());
  await settle();
  ctx.runButton().click();
  await settle();

  // Invariant: one model at a time, not one request. A model's units fan out concurrently
  // (serialising them defeated the backend's batching); two models must never have
  // requests open together.
  check("a model's units are opened concurrently", ctx.pending.length > 1, `${ctx.pending.length}`);
  check("...and all belong to one model", new Set(ctx.pending.map((p) => p.message.model)).size === 1);
  const firstModel = ctx.pending[0].message.model;
  check("...and the notice says which run is going and whose", ctx.notice().includes("Run 1 of 2: Fine-tuned model"), ctx.notice());
  check("...offering to stop it", !!ctx.window.document.getElementById("simplify-stop"));
  check("...and saying the page is not being changed", /not being changed/i.test(ctx.notice()));

  // draining model 1's requests finishes it, so what opens next is model 2 alone
  ctx.answerHeld();
  await settle();
  check("the second model starts only once the first has finished", ctx.pending.every((p) => p.message.model !== firstModel));
  check("...and it too runs alone", new Set(ctx.pending.map((p) => p.message.model)).size === 1);
}

async function aComparisonCanBeStopped() {
  const ctx = loadPage({ hold: true });
  await openPanel(ctx);
  ctx.rows()
    .filter((r) => ["finetuned", "llm_7b"].includes(r.dataset.compareModel))
    .forEach((r) => r.click());
  await settle();
  ctx.runButton().click();
  await settle();

  check("a running comparison is something to stop", ctx.run("runIsActive()") === true);
  check("...and stopping it reports that it did", ctx.run("stopCurrentRun()") === true);
  // stop takes effect between units; in-flight requests are answered and recorded
  ctx.answerHeld();
  await settle(60);
  check("no further requests are made", ctx.pending.length === 0, `${ctx.pending.length}`);
  check("the notice says it was stopped", /Stopped comparing/.test(ctx.notice()), ctx.notice());
  const done = ctx.sent.find((m) => m.cmd === "comparisonFinished");
  check("...and so does the notification", done && done.stopped === true);
  check("what the models did answer is still recorded", ctx.flushes().length >= 1, `${ctx.flushes().length}`);
}

async function aComparisonHoldsTheOneRunLock() {
  const ctx = loadPage({ hold: true });
  await openPanel(ctx);
  ctx.rows()[0].click();
  await settle();
  ctx.runButton().click();
  await settle();

  const claim = ctx.sent.find((m) => m.cmd === "claimRun" && m.run.kind === "compare");
  check("a comparison claims the one-run-at-a-time lock", !!claim);
  check("...as a comparison, not as a page run", claim && claim.run.kind === "compare");
  check("...saying how many models it will work through", claim && claim.run.models === 1);
  check("...and reports its progress against that", ctx.sent.some((m) => m.cmd === "runProgress" && m.progress.models === 1));
}

async function main() {
  await granularityComesFirstAndFiltersTheModels();
  await severalModelsCanBeChosen();
  await eachModelGetsTheSameInputs();
  await eachModelIsRecordedAsItsOwnKeptRun();
  await theArmsAreCombinedIntoOneReport();
  await bothCutsRunInOneComparison();
  await modelsRunOneAfterAnother();
  await aComparisonCanBeStopped();
  await aComparisonHoldsTheOneRunLock();

  let failed = 0;
  results.forEach(([name, ok, detail]) => {
    if (!ok) failed += 1;
    console.log(`${ok ? "PASS" : "FAIL"}  ${name}${ok || !detail ? "" : ` — ${detail}`}`);
  });
  console.log(`\n${results.length - failed}/${results.length} checks passed`);
  process.exit(failed === 0 ? 0 : 1);
}

main();
