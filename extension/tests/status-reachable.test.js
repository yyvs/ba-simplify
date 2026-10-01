// "Backend reachable" box on the status dashboard (shared/status.js's pollHealth /
// setLiveStatus).
//
//   cd extension/tests && npm install && node status-reachable.test.js
//
// Regression: after a failed check, updateDashboardFacts (no health object) wrote
// "Checking…" back into the box, so it never left "Checking…" while the backend was down.
// The box now has one writer, and a check with no answer is abandoned after HEALTH_TIMEOUT_MS.
//
// status.js is a non-module script that polls at load time; it runs in a vm context with jsdom.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { JSDOM } = require("jsdom");

const SHARED = path.join(__dirname, "..", "shared");
const SOURCES = ["model-labels.js", "model-selection.js", "status.js"].map((f) =>
  fs.readFileSync(path.join(SHARED, f), "utf8")
);

const DASHBOARD = `<!doctype html><html><body>
  <span id="dash-reachable" class="status-box status-checking">Checking…</span>
  <div id="dash-models">Checking…</div>
  <div id="dash-selected">Checking…</div>
  <span id="dash-method">Checking…</span>
</body></html>`;

// fetchImpl stands in for /health; the load-time poll hits it immediately.
function load(fetchImpl) {
  const dom = new JSDOM(DASHBOARD, { pretendToBeVisual: true });
  const { window } = dom;
  const timers = [];
  const cleared = [];
  const context = {
    window,
    document: window.document,
    Node: window.Node,
    URL: window.URL,
    console,
    fetch: (url, options) => fetchImpl(options),
    AbortController: window.AbortController,
    // Recorded, never fired: the cadence cases assert on what was armed and at what delay.
    setInterval: (fn, delay) => {
      timers.push({ fn, delay });
      return timers.length; // never 0 -- a real setInterval id is never 0 either
    },
    clearInterval: (id) => {
      cleared.push(id);
    },
    setTimeout: window.setTimeout.bind(window),
    performance: window.performance,
    chrome: {
      storage: {
        local: { get: () => Promise.resolve({}), set: () => Promise.resolve() },
        onChanged: { addListener() {} },
      },
      runtime: { onMessage: { addListener() {} }, sendMessage() {} },
    },
  };
  context.globalThis = context;
  vm.createContext(context);
  SOURCES.forEach((src) => vm.runInContext(src, context));

  const box = window.document.getElementById("dash-reachable");
  return {
    window,
    context,
    box,
    timers,
    cleared,
    // delay of the most recent setInterval
    armedDelay: () => (timers.length ? timers[timers.length - 1].delay : null),
    tick: () => timers[timers.length - 1].fn(),
    // jsdom can't set visibilityState: stub the getter and fire the event
    setVisibility: (state) => {
      Object.defineProperty(window.document, "visibilityState", {
        value: state,
        configurable: true,
      });
      window.document.dispatchEvent(new window.Event("visibilitychange"));
    },
    // enough for the async load-time poll (fetch, json, storage) to settle
    settle: () => new Promise((resolve) => window.setTimeout(resolve, 0)),
  };
}

const results = [];
const check = (name, condition) => results.push([name, !!condition]);

// --- backend down

async function refusedConnectionReportsFailure() {
  const { box, settle } = load(() => Promise.reject(new TypeError("Failed to fetch")));
  await settle();
  check("a refused connection stops saying Checking…", box.textContent !== "Checking…");
  check("...and says the backend is not reachable", box.textContent === "Not reachable");
  check("...in the failure colour", box.classList.contains("status-down"));
  check("...and not still in the checking colour", !box.classList.contains("status-checking"));
}

// A fetch that neither resolves nor rejects; without the abort the box stays on "Checking…".
async function aHangingCheckTimesOut() {
  // honours the abort signal, which status.js must supply and fire itself
  const { box, window, context } = load(
    ({ signal } = {}) =>
      new Promise((_, reject) => {
        if (!signal) return;
        signal.addEventListener("abort", () => {
          const err = new Error("aborted");
          err.name = "AbortError";
          reject(err);
        });
      })
  );
  const timeout = vm.runInContext("HEALTH_TIMEOUT_MS", context);
  check("the health check has a deadline", typeof timeout === "number" && timeout > 0);
  check("...shorter than the poll interval, so checks can't pile up", timeout < vm.runInContext("POLL_INTERVAL_MS", context));

  await new Promise((resolve) => window.setTimeout(resolve, timeout + 50));
  check("a check that never answers gives up", box.textContent !== "Checking…");
  check("...and says so", box.textContent === "No answer");
  check("...in the failure colour", box.classList.contains("status-down"));
}

// --- backend up

function healthResponse(body) {
  return () => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) });
}

async function readyBackendReportsReady() {
  const { box, settle } = load(
    // "online" is model-selection.js's DEFAULT_MODEL, i.e. the selected model
    healthResponse({
      model_loaded: true,
      models_loaded: ["online"],
      model_names: {},
      granularities: {},
    })
  );
  await settle();
  check("a ready backend reads as ready", box.textContent === "Backend ready");
  check("...in the healthy colour", box.classList.contains("status-up"));
}

async function loadingModelSaysSo() {
  const { box, settle } = load(healthResponse({ model_loaded: false }));
  await settle();
  check("a backend still loading its model says so", box.textContent === "Loading model…");
  check("...rather than reading as reachable", !box.classList.contains("status-up"));
}

async function httpErrorNamesTheStatus() {
  const { box, settle } = load(() => Promise.resolve({ ok: false, status: 503 }));
  await settle();
  check("an HTTP error names the status code", box.textContent === "HTTP 503");
  check("...in the failure colour", box.classList.contains("status-down"));
}

// --- polling cadence
// Backs off on a steady reading and stops while hidden. A constant 3s poll once filled the
// access log with /health requests.

async function aSteadyReadingBacksOff() {
  const t = load(healthResponse({ model_loaded: true, models_loaded: ["online"], model_names: {}, granularities: {} }));
  await t.settle();
  const fast = vm.runInContext("POLL_INTERVAL_MS", t.context);
  const ceiling = vm.runInContext("POLL_INTERVAL_MAX_MS", t.context);
  check("the first check is armed at the fast cadence", t.armedDelay() === fast);
  check("...and the ceiling is longer than it", ceiling > fast);

  const before = t.armedDelay();
  t.tick();
  await t.settle();
  check("the same reading twice widens the interval", t.armedDelay() > before);

  // well past the ceiling
  for (let i = 0; i < 20; i++) {
    t.tick();
    await t.settle();
  }
  check("...but never past the ceiling", t.armedDelay() === ceiling);
}

async function aChangedReadingSnapsBack() {
  let up = true;
  const t = load(() =>
    up
      ? Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ model_loaded: true, models_loaded: ["online"], model_names: {}, granularities: {} }),
        })
      : Promise.reject(new TypeError("Failed to fetch"))
  );
  await t.settle();
  const fast = vm.runInContext("POLL_INTERVAL_MS", t.context);

  // back off first so the snap back is observable
  for (let i = 0; i < 4; i++) {
    t.tick();
    await t.settle();
  }
  check("a steady reading has backed off before the backend drops", t.armedDelay() > fast);

  up = false;
  t.tick();
  await t.settle();
  check("the backend going down is noticed", t.box.textContent === "Not reachable");
  check("...and returns to the fast cadence", t.armedDelay() === fast);
}

async function aHiddenTabStopsPolling() {
  const t = load(healthResponse({ model_loaded: true, models_loaded: ["online"], model_names: {}, granularities: {} }));
  await t.settle();
  const armedWhileVisible = t.timers.length;
  const clearedWhileVisible = t.cleared.length;

  t.setVisibility("hidden");
  await t.settle();
  check("a tab going away clears its timer", t.cleared.length > clearedWhileVisible);
  check("...and arms no replacement", t.timers.length === armedWhileVisible);

  t.setVisibility("visible");
  await t.settle();
  check("coming back arms polling again", t.timers.length > armedWhileVisible);
  check("...at the fast cadence, not a backed-off one", t.armedDelay() === vm.runInContext("POLL_INTERVAL_MS", t.context));
}

(async () => {
  await refusedConnectionReportsFailure();
  await aHangingCheckTimesOut();
  await readyBackendReportsReady();
  await loadingModelSaysSo();
  await httpErrorNamesTheStatus();
  await aSteadyReadingBacksOff();
  await aChangedReadingSnapsBack();
  await aHiddenTabStopsPolling();

  results.forEach(([name, ok]) => console.log(`${ok ? "PASS" : "FAIL"}  ${name}`));
  const passed = results.filter(([, ok]) => ok).length;
  console.log(`\n${passed}/${results.length} checks passed`);
  if (passed !== results.length) process.exit(1);
})();
