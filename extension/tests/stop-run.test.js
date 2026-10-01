// Stopping a run part-way, from the notice box that reports it.
//
//   cd extension/tests && npm install && node stop-run.test.js
//
// A run can take minutes (a prompted LLM at document scope takes seconds per section);
// toggling off reverts everything. Stopping keeps written text, sends nothing further,
// and logs the run as stopped, not completed or failed.
//
// content.js is loaded as in failure-notice.test.js, with the real showNotice() (the
// button is part of the box). The backend stub holds requests open so there is something
// to stop.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { JSDOM } = require("jsdom");

const EXT = path.join(__dirname, "..");
const source = fs.readFileSync(path.join(EXT, "content.js"), "utf8");
const LABELS_JS = fs.readFileSync(path.join(EXT, "shared", "model-labels.js"), "utf8");
// loaded between model-selection.js and content.js in manifest.json: lock key, run kinds, wording
const ACTIVE_RUN_JS = fs.readFileSync(path.join(EXT, "shared", "active-run.js"), "utf8");

const results = [];
const check = (name, condition, detail = "") => results.push([name, !!condition, detail]);
const tick = (ms = 10) => new Promise((resolve) => setTimeout(resolve, ms));

// More paragraphs than MAX_CONCURRENT_SIMPLIFY, so a first wave can be answered while
// the rest is still queued.
const PARAGRAPHS = Array.from(
  { length: 20 },
  (_, i) => `<p>Residents are requested to position their receptacles at the kerbside on day ${i + 1}.</p>`
).join("");

function loadContentScript(html = PARAGRAPHS) {
  const dom = new JSDOM(`<!doctype html><html><head><title>Bin collections</title></head><body>${html}</body></html>`, {
    pretendToBeVisual: true,
    url: "https://example.com/bins",
  });
  const { window } = dom;
  const messages = [];
  // simplify requests with their resolvers, held open until a test answers
  const pending = [];

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
          // one-run-at-a-time lock (shared/active-run.js): uncontended here, always granted
          if (message.cmd === "claimRun") {
            if (callback) callback({ ok: true, holder: null });
            return;
          }
          if (message.cmd === "runProgress" || message.cmd === "releaseRun") {
            if (callback) callback({ ok: true });
            return;
          }
          messages.push(message);
          if (message.cmd === "healthCheck") {
            callback({
              ok: true,
              model: "finetuned",
              modelLabel: "Fine-tuned sentence model",
              modelId: "yunvs/bart-base-wikilarge-simplification",
              method: "fine_tuned_seq2seq",
              granularity: "sentence_by_sentence",
              methodLabel: "Fine-tuned seq2seq · Sentence by sentence",
              audience: "non_native_speakers",
              audienceLabel: "Non-native speakers",
              audienceApplies: false,
              documentMaxTokens: 512,
            });
            return;
          }
          if (message.cmd === "fetchSimplify") {
            pending.push({ text: message.text, callback });
            return;
          }
        },
      },
      storage: { local: { get: (_keys, cb) => cb && cb({}) } },
    },
  };
  context.globalThis = context;
  vm.createContext(context);
  vm.runInContext(LABELS_JS, context);
  vm.runInContext(ACTIVE_RUN_JS, context);
  vm.runInContext(source, context);

  const run = (expression) => vm.runInContext(expression, context);
  return {
    run,
    window,
    messages,
    pending,
    // answers everything currently in flight
    answerAll: () =>
      pending.splice(0).forEach(({ text, callback }) =>
        callback({ ok: true, model: "finetuned", audience: "non_native_speakers", data: { simplified: `${text} SIMPLE`, cached: false } })
      ),
    box: () => window.document.getElementById("simplify-notice"),
    text: () => {
      const box = window.document.getElementById("simplify-notice");
      return box ? box.textContent : "";
    },
    stopButton: () => window.document.getElementById("simplify-stop"),
    // content.js's single onMessage listener: the service worker's side of the stop
    dispatch: (message, respond) => context.onMessage(message, {}, respond),
    historyFlushes: () => messages.filter((m) => m.cmd === "recordHistory"),
    simplifiedCount: () => window.document.querySelectorAll("[data-original-html]").length,
  };
}

async function theRunningNoticeOffersAWayOut() {
  const ctx = loadContentScript();
  ctx.run("toggleSimplification()");
  await tick();
  check("the preflight's box carries a stop button", !!ctx.stopButton());
  check("...saying what it does", /stop/i.test(ctx.stopButton().title));

  // preflight passes, run starts with requests held open
  await tick(20);
  check("the running box still carries it", !!ctx.stopButton());
  check("...and requests are in flight to stop", ctx.pending.length > 0);

  // MAX_CONCURRENT_SIMPLIFY caps what is in flight; the rest is queued behind it
  const inFlight = ctx.pending.length;
  ctx.run("stopCurrentRun()");
  await tick();
  check("stopping sends nothing further", ctx.pending.length === inFlight, `${ctx.pending.length} vs ${inFlight}`);
  check("the box says the run was stopped", /Stopped simplifying/.test(ctx.text()), ctx.text().slice(0, 80));
  check("...and it is not dressed as a failure", !/couldn't|troubleshooting/i.test(ctx.text()));
  check("...no troubleshooting tab is opened", !ctx.messages.some((m) => m.cmd === "openErrorPage"));
  check("...and the stop button is gone from the report", !ctx.stopButton());
}

async function whatWasAlreadySimplifiedStays() {
  const ctx = loadContentScript();
  ctx.run("toggleSimplification()");
  await tick(20);
  // answer the first wave so some text is rewritten before the stop
  ctx.answerAll();
  await tick();
  const written = ctx.simplifiedCount();
  check("some of the page is rewritten before the stop", written > 0, `${written} elements`);

  ctx.run("stopCurrentRun()");
  await tick();
  check("stopping leaves that text in place", ctx.simplifiedCount() === written, `${ctx.simplifiedCount()}`);
  check("...and the badge still reports the page as simplified", ctx.messages.some((m) => m.cmd === "updateBadge" && m.isSimplified === true));

  ctx.run("toggleSimplification()");
  await tick();
  check("toggling off after a stop reverts the page", ctx.simplifiedCount() === 0);
}

async function answersThatArriveAfterTheStopAreNotWritten() {
  const ctx = loadContentScript();
  ctx.run("toggleSimplification()");
  await tick(20);
  const held = ctx.pending.length;
  check("requests are in flight when the stop lands", held > 0);

  ctx.run("stopCurrentRun()");
  await tick();
  const written = ctx.simplifiedCount();
  ctx.answerAll(); // the backend answers a run nobody is waiting for any more
  await tick();
  check(
    "an answer arriving after the stop changes no text",
    ctx.simplifiedCount() === written,
    `${written} -> ${ctx.simplifiedCount()}`
  );
  check("...and does not redraw progress over the stopped report", /Stopped simplifying/.test(ctx.text()));
}

async function theLogRecordsAStoppedRunAsStopped() {
  const ctx = loadContentScript();
  ctx.run("toggleSimplification()");
  await tick(20);
  // one wave answered, the rest still queued
  ctx.answerAll();
  await tick();
  check("the run is still going when the stop lands", ctx.pending.length > 0, `${ctx.pending.length} in flight`);
  check("...with entries collected but not yet flushed", ctx.historyFlushes().length === 0);
  ctx.run("stopCurrentRun()");
  await tick();

  const flushes = ctx.historyFlushes();
  check("stopping flushes what the run had collected", flushes.length > 0);
  const last = flushes[flushes.length - 1];
  check("...recorded as stopped, not completed", last.session.status === "stopped", last.session && last.session.status);
  // both leave the page partly covered, but "failed" means the service broke
  check("...and not as failed", last.session.status !== "failed");
}

async function stoppingWhatIsNotRunningDoesNothing() {
  const ctx = loadContentScript();
  check("there is nothing to stop before a run starts", ctx.run("stopCurrentRun()") === false);
  check("...and no notice is invented for it", !ctx.box());

  ctx.run("toggleSimplification()");
  await tick(20);
  for (let i = 0; i < 10 && ctx.pending.length > 0; i += 1) {
    ctx.answerAll();
    await tick();
  }
  check("a settled run has nothing to stop either", ctx.run("stopCurrentRun()") === false);
}

async function theWorkerCanStopARunToo() {
  // same stop, from the toolbar icon's menu
  const ctx = loadContentScript();
  ctx.run("toggleSimplification()");
  await tick(20);

  let answer = null;
  ctx.dispatch({ cmd: "stopRun" }, (resp) => (answer = resp));
  await tick();
  check("a stopRun message stops the run", answer && answer.stopped === true, JSON.stringify(answer));
  check("...and the page says so", /Stopped simplifying/.test(ctx.text()));

  let second = null;
  ctx.dispatch({ cmd: "stopRun" }, (resp) => (second = resp));
  await tick();
  check("asking again reports there was nothing to stop", second && second.stopped === false, JSON.stringify(second));
}

async function main() {
  await theRunningNoticeOffersAWayOut();
  await whatWasAlreadySimplifiedStays();
  await answersThatArriveAfterTheStopAreNotWritten();
  await theLogRecordsAStoppedRunAsStopped();
  await stoppingWhatIsNotRunningDoesNothing();
  await theWorkerCanStopARunToo();

  let failed = 0;
  results.forEach(([name, ok, detail]) => {
    if (!ok) failed += 1;
    console.log(`${ok ? "PASS" : "FAIL"}  ${name}${ok || !detail ? "" : ` — ${detail}`}`);
  });
  console.log(`\n${results.length - failed}/${results.length} checks passed`);
  process.exit(failed === 0 ? 0 : 1);
}

main();
