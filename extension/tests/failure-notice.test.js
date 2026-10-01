// DOM-level checks for the notice box's failure state.
//
//   cd extension/tests && npm install && node failure-notice.test.js
//
// content.js is loaded as in document-watcher.test.js, but keeps the real showNotice():
// the rows, their order and classes are what's under test.
//
// Covers: the failure state reuses the progress box's classes (sn-title / sn-outcome /
// sn-muted / sn-status), and no technical detail (URL, "TypeError: Failed to fetch", HTTP
// status) reaches the page while the troubleshooting tab still receives it.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { JSDOM } = require("jsdom");

const CONTENT_JS = path.join(__dirname, "..", "content.js");
const source = fs.readFileSync(CONTENT_JS, "utf8");
// manifest.json loads shared/model-labels.js before content.js, which depends on it
// (e.g. sessionSummary).
const LABELS_JS = fs.readFileSync(path.join(__dirname, "..", "shared", "model-labels.js"), "utf8");
// loaded between model-selection.js and content.js in manifest.json: lock key and wording
const ACTIVE_RUN_JS = fs.readFileSync(path.join(__dirname, "..", "shared", "active-run.js"), "utf8");

let failures = 0;
function check(label, ok) {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}`);
  if (!ok) failures += 1;
}

function loadContentScript() {
  const dom = new JSDOM(`<!doctype html><html><head><title>Example page</title></head><body></body></html>`, {
    pretendToBeVisual: true,
    url: "https://example.com/article",
  });
  const { window } = dom;
  const messages = [];

  const context = {
    window,
    document: window.document,
    location: window.location,
    Node: window.Node,
    MutationObserver: window.MutationObserver,
    getComputedStyle: window.getComputedStyle.bind(window),
    performance: window.performance,
    setTimeout: window.setTimeout.bind(window),
    clearTimeout: window.clearTimeout.bind(window),
    setInterval: window.setInterval.bind(window),
    clearInterval: window.clearInterval.bind(window),
    // silence the expected raw-error logging
    console: { ...console, error() {}, log() {}, warn() {} },
    crypto: { randomUUID: () => "test-page-session" },
    fetch: () => Promise.reject(new Error("the failure notice must not reach the network")),
    chrome: {
      runtime: {
        onMessage: { addListener() {} },
        lastError: null,
        sendMessage(message, callback) {
          messages.push(message);
          // lock uncontended: every claim granted
          if (callback) callback(message.cmd === "claimRun" ? { ok: true, holder: null } : { ok: true });
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
    // one row per line, with its class
    rows() {
      const box = window.document.getElementById("simplify-notice");
      if (!box) return [];
      return Array.from(box.children).map((el) => ({ className: el.className, text: el.textContent }));
    },
    text() {
      const box = window.document.getElementById("simplify-notice");
      return box ? box.textContent : "";
    },
    // by cmd, not position: messages[] also holds the on-load badge reset
    errorPageRequests() {
      return messages.filter((m) => m.cmd === "openErrorPage");
    },
    fail(stage, error) {
      run(`showFailureNotice(${JSON.stringify(stage)}, ${JSON.stringify(error)})`);
    },
  };
}

// Four states, worded by what the reader can do: start the service, or retry / read the tab.
function eachFailureGetsItsOwnWording() {
  const cases = [
    {
      label: "a backend that isn't running",
      stage: "reachable",
      error: "backend not reachable at http://127.0.0.1:8000 [TypeError: Failed to fetch]",
      body: "The simplification service isn't available.",
      hint: "Please make sure the local service is running, then try again.",
    },
    {
      label: "a request the backend answered with an error",
      stage: null,
      error: "Error: HTTP 500",
      body: "The simplification service returned an error while processing this page.",
      hint: "Please try again or check the troubleshooting information.",
    },
    {
      label: "a request that timed out",
      stage: null,
      error: "AbortError: The operation was aborted.",
      body: "The simplification service took too long to respond.",
      hint: "Please try again or check the troubleshooting information.",
    },
    {
      label: "a response that wasn't usable",
      stage: "model_response",
      error: "test request returned no usable output",
      body: "The simplification service returned an unexpected response.",
      hint: "Please check the troubleshooting information.",
    },
  ];

  for (const c of cases) {
    const ctx = loadContentScript();
    ctx.fail(c.stage, c.error);
    const rows = ctx.rows();
    check(`${c.label}: says what happened`, rows.some((r) => r.text === c.body));
    check(`${c.label}: says what to do about it`, rows.some((r) => r.text === c.hint));
    ctx.run("stopFailureCountdown()");
  }
}

// Loading and never-loaded models both read as unavailable; the troubleshooting page
// distinguishes them.
function aServiceThatIsntReadyReadsAsUnavailable() {
  for (const stage of ["model_loaded", "model_unavailable"]) {
    const ctx = loadContentScript();
    ctx.fail(stage, `selected model isn't loaded on the backend`);
    check(
      `stage "${stage}" reads as the service being unavailable`,
      ctx.text().includes("The simplification service isn't available.")
    );
    check(`...and the stage still reaches troubleshooting as "${stage}"`, ctx.errorPageRequests()[0].stage === stage);
    ctx.run("stopFailureCountdown()");
  }
}

// The failure box is a state of the progress box: same rows, same sizes.
function theBoxIsTheProgressBoxInAnotherState() {
  const ctx = loadContentScript();
  ctx.fail("reachable", "backend not reachable");
  const rows = ctx.rows();

  check("the box is the same element the progress notice renders into", ctx.text() !== "");
  check(
    "the rows are the progress box's own title/body/secondary/status classes, in that order",
    rows.map((r) => r.className).join(" | ") === "sn-title | sn-outcome | sn-muted | sn-status"
  );
  check(
    "the title names the page, like the progress box's does",
    rows[0].text === "Couldn't simplify Example page"
  );
  check("the transient line is the box's italic status row", rows[3].className === "sn-status");
  ctx.run("stopFailureCountdown()");
}

function nothingTechnicalReachesThePage() {
  const ctx = loadContentScript();
  const raw = "backend not reachable at http://127.0.0.1:8000 [TypeError: Failed to fetch]\n    at fetchSimplify";
  ctx.fail("reachable", raw);
  const shown = ctx.text();

  check("no URL in the box", !shown.includes("127.0.0.1") && !shown.includes("http"));
  check("no exception text in the box", !/TypeError|Failed to fetch/.test(shown));
  check("no stack frame in the box", !shown.includes("at fetchSimplify"));
  check("no HTTP status in the box", !/HTTP \d/.test(shown));
  check("the raw error still goes to troubleshooting", ctx.errorPageRequests()[0].error === raw);
  check("...at a stage that page has a section for", ctx.errorPageRequests()[0].stage === "reachable");
  ctx.run("stopFailureCountdown()");
}

// The countdown counts down to background.js's INFO_TAB_OPEN_DELAY_MS, not its own timer.
function theCountdownMatchesWhenTheTabOpens() {
  const backgroundSource = fs.readFileSync(path.join(__dirname, "..", "background.js"), "utf8");
  const tabDelay = Number(/INFO_TAB_OPEN_DELAY_MS = (\d+)/.exec(backgroundSource)[1]);

  const ctx = loadContentScript();
  ctx.fail("reachable", "backend not reachable");
  const announced = /in (\d+) seconds?…/.exec(ctx.text());

  check("the box announces a countdown", Boolean(announced));
  check(
    `the announced ${announced && announced[1]}s matches the ${tabDelay}ms the tab actually waits`,
    Number(announced[1]) === Math.round(tabDelay / 1000)
  );
  check("troubleshooting is asked for once", ctx.errorPageRequests().length === 1);
  ctx.run("stopFailureCountdown()");
}

async function theStatusLineSwitchesWhenTheTabOpens() {
  const ctx = loadContentScript();
  ctx.fail("reachable", "backend not reachable");
  check("it starts on the countdown", /will open automatically/.test(ctx.text()));

  // move the clock past the delay (a constant paired with background.js's) instead of waiting 3s
  ctx.run(`
    const realNow = performance.now.bind(performance);
    performance = { now: () => realNow() + TROUBLESHOOT_OPEN_DELAY_MS };
  `);
  await new Promise((resolve) => setTimeout(resolve, 800));
  check("it ends on 'Opening troubleshooting…'", ctx.text().includes("Opening troubleshooting…"));
  check("no expired countdown is left on screen", !/will open automatically/.test(ctx.text()));
  check("the countdown stopped once it was done", ctx.run("countdownTimer") === null);
}

// A click while a failure box is still counting down belongs to the new run.
function anotherClickSupersedesTheFailure() {
  const ctx = loadContentScript();
  ctx.fail("reachable", "backend not reachable");
  ctx.run("toggleSimplification()");

  check("the failure countdown was stopped", ctx.run("countdownTimer") === null);
  check("the box now belongs to the new run", ctx.text().includes("Simplifying Example page"));
}

(async () => {
  eachFailureGetsItsOwnWording();
  aServiceThatIsntReadyReadsAsUnavailable();
  theBoxIsTheProgressBoxInAnotherState();
  nothingTechnicalReachesThePage();
  theCountdownMatchesWhenTheTabOpens();
  await theStatusLineSwitchesWhenTheTabOpens();
  anotherClickSupersedesTheFailure();
  console.log(failures === 0 ? "\nall checks passed" : `\n${failures} check(s) failed`);
  process.exit(failures === 0 ? 0 : 1);
})();
