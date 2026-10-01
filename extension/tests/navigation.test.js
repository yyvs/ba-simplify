// The page changing under a running simplification.
//
//   cd extension/tests && npm install && node navigation.test.js
//
// content.js is loaded as in document-watcher.test.js. jsdom's history.pushState() is the
// case at issue: URL changes, no reload, observer still attached.
//
// Content from a route to a new page must not be simplified or announced; content added
// to the original page still must be.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { JSDOM } = require("jsdom");

const CONTENT_JS = path.join(__dirname, "..", "content.js");
const source = fs.readFileSync(CONTENT_JS, "utf8");
// manifest.json loads shared/model-labels.js before content.js, which depends on it
// (e.g. sessionSummary).
const LABELS_JS = fs.readFileSync(path.join(__dirname, "..", "shared", "model-labels.js"), "utf8");
// loaded between model-selection.js and content.js in manifest.json: lock key, run kinds, wording
const ACTIVE_RUN_JS = fs.readFileSync(path.join(__dirname, "..", "shared", "active-run.js"), "utf8");

let failures = 0;
function check(label, ok) {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}`);
  if (!ok) failures += 1;
}

const PROSE = "A paragraph of ordinary prose with comfortably more than four words in it.";

function loadContentScript() {
  const dom = new JSDOM(
    `<!doctype html><html><body><main id="scope"><p id="first">${PROSE}</p></main></body></html>`,
    { pretendToBeVisual: true, url: "https://example.com/one" }
  );
  const { window } = dom;
  const sent = [];
  const badges = [];
  const notices = [];
  const pendingHealth = [];

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
    console,
    crypto: { randomUUID: () => `session-${badges.length}-${sent.length}` },
    chrome: {
      runtime: {
        onMessage: { addListener() {} },
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
          if (message.cmd === "updateBadge") badges.push(message.isSimplified);
          // held so a test can act during the preflight and control when it ends
          if (message.cmd === "healthCheck") {
            pendingHealth.push(callback);
            return;
          }
          if (message.cmd !== "fetchSimplify") return;
          sent.push(message.text);
          callback({
            ok: true,
            model: "test-model",
            audience: "non_native_speakers",
            data: { simplified: `simplified: ${message.text}`, cached: false },
          });
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
  vm.runInContext(
    `showNotice = function (content) {
       notices.push(typeof content === "string" ? content : content.map((n) => n.textContent).join(" "));
     };`,
    context
  );

  const run = (expression) => vm.runInContext(expression, context);
  return {
    window,
    sent,
    badges,
    notices,
    run,
    finishPreflight: () =>
      pendingHealth.splice(0).forEach((cb) =>
        cb({ ok: true, granularity: "sentence", documentMaxTokens: 512, model: "test-model" })
      ),
    // two turns: the observer fires on a microtask, then its simplification settles
    settle: () => new Promise((resolve) => window.setTimeout(resolve, 0)).then(() => new Promise((r) => window.setTimeout(r, 0))),
    addProse: (id) => run(`
      const p = document.createElement("p");
      p.id = ${JSON.stringify(id)};
      p.textContent = ${JSON.stringify(PROSE)};
      document.getElementById("scope").appendChild(p);
    `),
    navigate: (url) => run(`history.pushState({}, "", ${JSON.stringify(url)});`),
  };
}

// Baseline: late content on the same page is still simplified.
async function lateContentOnTheSamePageIsStillSimplified() {
  const ctx = loadContentScript();
  ctx.run("startObserver(); isSimplified = true;");
  ctx.addProse("late");
  await ctx.settle();

  check("content added to the same page is still sent", ctx.sent.length === 1);
  check("...and written into the page", ctx.window.document.getElementById("late").textContent.startsWith("simplified:"));
}

async function aRoutedNavigationEndsTheRun() {
  const ctx = loadContentScript();
  ctx.run("startObserver(); isSimplified = true;");
  ctx.run(`simplifyElement(document.getElementById("first"))`);
  await ctx.settle();
  check("the first page was simplified", ctx.window.document.getElementById("first").classList.contains("simplified"));

  ctx.navigate("/two");
  ctx.addProse("second-page");
  await ctx.settle();

  check("the new page's content is not sent to the backend", ctx.sent.length === 1);
  check("...and is left exactly as the page wrote it", ctx.window.document.getElementById("second-page").textContent === PROSE);
  check("the run is no longer marked on", ctx.run("isSimplified") === false);
  check("the badge was cleared", ctx.badges[ctx.badges.length - 1] === false);
  check("the observer was detached", ctx.run("observer") === null);
  check(
    "the reader is told why the extension went quiet",
    ctx.notices.some((n) => n.includes("new page") && n.includes("simplification is off"))
  );
  check(
    "text the router left standing is back to the original",
    !ctx.window.document.getElementById("first").classList.contains("simplified") &&
      ctx.window.document.getElementById("first").textContent === PROSE
  );
}

// Document mode's watcher reports late prose; a new page's content is not late prose.
async function aRoutedNavigationRaisesNoLateContentNotice() {
  const ctx = loadContentScript();
  ctx.run(`startDocumentWatcher(document.getElementById("scope")); isSimplified = true;`);
  ctx.navigate("/two");
  ctx.addProse("second-page");
  await ctx.settle();
  // past the watcher's 1200ms coalescing delay
  await new Promise((resolve) => setTimeout(resolve, 1400));

  check(
    "no 'new text loaded' notice is raised for a page that was navigated to",
    !ctx.notices.some((n) => n.includes("loaded after"))
  );
  check("the late-content counter stays at zero", ctx.run("unsimplifiedLateCount") === 0);
}

// A click during the preflight (up to a minute for prompted LLMs) must not simplify
// whatever page is showing when it returns.
async function aNavigationDuringThePreflightCancelsTheRun() {
  const ctx = loadContentScript();
  ctx.run("toggleSimplification()"); // the click; the preflight is now in flight
  ctx.navigate("/two");
  ctx.run("checkForNavigation()");
  ctx.finishPreflight();
  await ctx.settle();

  check("a preflight that returns after a navigation starts no run", ctx.sent.length === 0);
  check("...and simplifies nothing", !ctx.window.document.getElementById("first").classList.contains("simplified"));
  check("the session was ended", ctx.run(`session.phase`) === "idle");
}

(async () => {
  await lateContentOnTheSamePageIsStillSimplified();
  await aRoutedNavigationEndsTheRun();
  await aRoutedNavigationRaisesNoLateContentNotice();
  await aNavigationDuringThePreflightCancelsTheRun();
  console.log(failures === 0 ? "\nall checks passed" : `\n${failures} check(s) failed`);
  process.exit(failures === 0 ? 0 : 1);
})();
