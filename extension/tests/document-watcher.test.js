// DOM-level checks for document mode's dynamic-content watcher.
//
//   cd extension/tests && npm install && node document-watcher.test.js
//
// content.js is a non-module content script expecting `chrome`, a document and
// MutationObserver, so it runs in a vm context with jsdom and a minimal chrome.*. This
// also exposes top-level state (unsimplifiedLateCount) for the stopObserver() check.
//
// Contract: notice late-arriving prose, coalesce a burst into one message, and stay
// silent for anything else, including the extension's own notice element.
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

// Long enough to clear the watcher's 12-word prose threshold.
const LATE_PARAGRAPH =
  "A lazily loaded paragraph with clearly more than twelve words in it, for sure.";

function loadContentScript() {
  const dom = new JSDOM(
    `<!doctype html><html><body>
       <main id="scope">
         <h2>Heading</h2>
         <p>Some existing prose that is long enough to be a real paragraph of text.</p>
       </main>
     </body></html>`,
    { pretendToBeVisual: true }
  );
  const { window } = dom;
  const notices = [];

  const context = {
    window,
    document: window.document,
    // content.js watches the URL for SPA navigation
    location: window.location,
    Node: window.Node,
    MutationObserver: window.MutationObserver,
    getComputedStyle: window.getComputedStyle.bind(window),
    performance: window.performance,
    setTimeout: window.setTimeout.bind(window),
    clearTimeout: window.clearTimeout.bind(window),
    setInterval: window.setInterval.bind(window),
    clearInterval: window.clearInterval.bind(window),
    console,
    crypto: { randomUUID: () => "test-page-session" },
    fetch: () => Promise.reject(new Error("the watcher must not reach the network")),
    chrome: {
      runtime: {
        onMessage: { addListener() {} },
        lastError: null,
        // lock uncontended: every claim granted
        sendMessage(message, callback) {
          if (callback) callback(message.cmd === "claimRun" ? { ok: true, holder: null } : { ok: true });
        },
      },
      storage: { local: { get: (_keys, cb) => cb && cb({}) } },
    },
    notices,
  };
  context.globalThis = context;
  vm.createContext(context);
  vm.runInContext(LABELS_JS, context);
  vm.runInContext(ACTIVE_RUN_JS, context);
  vm.runInContext(source, context);

  // Notices as text: the real notice is a mutation (the loop OWN_UI_IDS prevents), and
  // the test shouldn't rely on that check.
  vm.runInContext(
    `showNotice = function (content) {
       notices.push(typeof content === "string" ? content : content.map((n) => n.textContent).join(" "));
     };`,
    context
  );

  const run = (expression) => vm.runInContext(expression, context);
  const scope = window.document.getElementById("scope");
  const watch = () => run(`startDocumentWatcher(document.getElementById("scope"))`);
  const append = (parent, text, tag = "p") => {
    const el = window.document.createElement(tag);
    el.textContent = text;
    parent.appendChild(el);
    return el;
  };
  return { run, window, scope, notices, watch, append };
}

// past the watcher's 1200ms debounce, so silence is observable
const settle = () => new Promise((resolve) => setTimeout(resolve, 1600));

const results = [];
const check = (name, condition) => results.push([name, !!condition]);

async function detectsLateProseOnce() {
  const { scope, notices, watch, append } = loadContentScript();
  watch();
  for (let i = 0; i < 3; i++) append(scope, LATE_PARAGRAPH);
  await settle();
  check("a burst of late prose produces exactly one notice", notices.length === 1);
  check("the notice reports the cumulative count", /^3 new text items/.test(notices[0] || ""));
  check("the notice states the toggle workaround", /Toggle off and on/.test(notices[0] || ""));
  check("the notice explains why, not just what", /whole sections/.test(notices[0] || ""));
}

async function ignoresShortChrome() {
  const { scope, notices, watch, append } = loadContentScript();
  watch();
  append(scope, "Accept cookies");
  await settle();
  check("a short inserted label raises no notice", notices.length === 0);
}

async function ignoresOwnNotice() {
  const { window, notices, watch } = loadContentScript();
  watch();
  const div = window.document.createElement("div");
  div.id = "simplify-notice";
  div.textContent = "Simplified this page in 12s. Found 9 text items in total, sent 9 to the backend.";
  window.document.body.appendChild(div);
  await settle();
  check("the extension's own notice does not re-trigger the watcher", notices.length === 0);
}

async function ignoresOutOfScope() {
  const { window, notices, watch, append } = loadContentScript();
  watch();
  append(window.document.body, "A footer paragraph outside the content scope with more than twelve words in it.");
  await settle();
  check("prose outside the content scope raises no notice", notices.length === 0);
}

async function ignoresAlreadySimplifiedSections() {
  const { window, scope, notices, watch, append } = loadContentScript();
  const simplified = window.document.createElement("div");
  simplified.dataset.originalHtml = "<p>old</p>";
  scope.appendChild(simplified);
  watch();
  append(simplified, "Text inserted inside a section that has already been simplified, twelve words plus.");
  await settle();
  check("insertions inside an already-simplified section raise no notice", notices.length === 0);
}

async function stopObserverResets() {
  const { run, scope, notices, watch, append } = loadContentScript();
  watch();
  append(scope, LATE_PARAGRAPH);
  run(`stopObserver()`);
  await settle();
  check("stopObserver cancels a pending notice", notices.length === 0);
  check("stopObserver clears the late-content counter", run(`unsimplifiedLateCount`) === 0);
  append(scope, LATE_PARAGRAPH);
  await settle();
  check("stopObserver detaches the watcher", notices.length === 0);
}

async function sentencePathUntouched() {
  const { run } = loadContentScript();
  check(
    "the sentence path's simplifying observer is still separate",
    run(`typeof startObserver === "function" && startObserver !== startDocumentWatcher`)
  );
}

(async () => {
  await detectsLateProseOnce();
  await ignoresShortChrome();
  await ignoresOwnNotice();
  await ignoresOutOfScope();
  await ignoresAlreadySimplifiedSections();
  await stopObserverResets();
  await sentencePathUntouched();

  const failed = results.filter(([, ok]) => !ok);
  for (const [name, ok] of results) console.log(`${ok ? "PASS" : "FAIL"}  ${name}`);
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  process.exit(failed.length ? 1 : 0);
})();
