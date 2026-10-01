// One run at a time: the lock the service worker owns, and what a page that couldn't get
// it shows instead.
//
//   cd extension/tests && npm install && node one-run-at-a-time.test.js
//
// The models are one local service with one batch queue per model, so two concurrent runs
// each go at about half speed and record timings that describe neither (unusable on the
// prompted-LLM path, where run cost is measured).
//
// Covers the worker's lock (one holder, races, abandoned records) and the refused page's
// notice (which page runs, its progress, stop or wait).
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { JSDOM } = require("jsdom");

const EXT = path.join(__dirname, "..");

const results = [];
const check = (name, condition, detail = "") => results.push([name, !!condition, detail]);
const settle = (ms = 20) => new Promise((resolve) => setTimeout(resolve, ms));

// --- worker (owns the lock)

function loadWorker({ stored = {}, liveTabs = [1, 2, 3] } = {}) {
  const store = { ...stored };
  const listeners = {};
  const messagesToTab = [];
  const open = new Set(liveTabs);

  const chrome = {
    runtime: {
      lastError: null,
      getURL: (p) => `chrome-extension://test/${p}`,
      onInstalled: { addListener: (fn) => (listeners.installed = fn) },
      onMessage: { addListener: (fn) => (listeners.message = fn) },
      sendMessage() {},
    },
    contextMenus: {
      removeAll: (cb) => cb && cb(),
      create() {},
      update: (_id, _props, cb) => cb && cb(),
      onClicked: { addListener: (fn) => (listeners.menuClick = fn) },
      ACTION_MENU_TOP_LEVEL_LIMIT: 6,
    },
    storage: {
      local: {
        get(keys) {
          const names = Array.isArray(keys) ? keys : [keys];
          const out = {};
          names.forEach((k) => {
            if (k in store) out[k] = store[k];
          });
          return Promise.resolve(out);
        },
        set(values) {
          Object.assign(store, values);
          return Promise.resolve();
        },
        remove(keys) {
          (Array.isArray(keys) ? keys : [keys]).forEach((k) => delete store[k]);
          return Promise.resolve();
        },
      },
      onChanged: { addListener: (fn) => (listeners.storageChange = fn) },
    },
    action: {
      onClicked: { addListener: (fn) => (listeners.actionClick = fn) },
      setBadgeText() {},
      setBadgeBackgroundColor() {},
    },
    tabs: {
      create: () => Promise.resolve({ id: 99 }),
      query: () => Promise.resolve([]),
      update: () => Promise.resolve(),
      // closed tabs reject: how the worker tells a live holder from an abandoned record
      get: (tabId) => (open.has(tabId) ? Promise.resolve({ id: tabId }) : Promise.reject(new Error("No tab"))),
      onRemoved: { addListener: (fn) => (listeners.tabRemoved = fn) },
      sendMessage: (tabId, message, cb) => {
        messagesToTab.push({ tabId, message });
        if (cb) cb({ ok: true, stopped: true });
      },
    },
    windows: { update: () => Promise.resolve() },
    notifications: { create() {} },
  };

  const context = {
    chrome,
    console: { log() {}, warn() {}, error() {} },
    fetch: () => Promise.reject(new Error("the lock must not reach the network")),
    setTimeout: (fn, ms) => setTimeout(fn, ms),
    clearTimeout: (t) => clearTimeout(t),
    AbortController,
    importScripts: (...files) =>
      files.forEach((f) => vm.runInContext(fs.readFileSync(path.join(EXT, f), "utf8"), context)),
  };
  context.globalThis = context;
  context.self = context;
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(EXT, "background.js"), "utf8"), context);

  const ask = (tabId, message) =>
    new Promise((resolve) => {
      const kept = listeners.message(message, { tab: { id: tabId } }, resolve);
      if (!kept) resolve(undefined);
    });

  return {
    store,
    messagesToTab,
    closeTab: (tabId) => {
      open.delete(tabId);
      listeners.tabRemoved(tabId);
    },
    // content-script messages; the worker reads the tab id off the sender
    claim: (tabId, run) => ask(tabId, { cmd: "claimRun", run }),
    release: (tabId) => ask(tabId, { cmd: "releaseRun" }),
    progress: (tabId, progress) => ask(tabId, { cmd: "runProgress", progress }),
    active: () => ask(null, { cmd: "activeRun" }),
    stop: () => ask(null, { cmd: "stopActiveRun" }),
    record: () => store.simplifierActiveRun,
    run: (expr) => vm.runInContext(expr, context),
  };
}

const pageRun = (n) => ({ kind: "page", sessionId: `s${n}`, url: `https://example.org/${n}`, title: `Page ${n}`, host: "example.org" });

async function oneTabAtATime() {
  const ctx = loadWorker();
  const first = await ctx.claim(1, pageRun(1));
  check("the first page to ask gets the lock", first.ok === true);
  check("...and the record names the tab holding it, not what the page claimed", ctx.record().tabId === 1);

  const second = await ctx.claim(2, pageRun(2));
  check("a second tab is refused", second.ok === false);
  check("...and told which page is running, so it can say so", second.holder && second.holder.title === "Page 1");
  check("...without the record changing hands", ctx.record().tabId === 1);

  const again = await ctx.claim(1, pageRun(1));
  check("the holder re-claiming its own lock succeeds", again.ok === true, JSON.stringify(again));

  await ctx.release(1);
  check("releasing frees it", ctx.record() === undefined);
  const third = await ctx.claim(2, pageRun(2));
  check("...and the next tab to ask gets it", third.ok === true);
}

async function onlyTheHolderCanReleaseOrReport() {
  const ctx = loadWorker();
  await ctx.claim(1, pageRun(1));

  await ctx.release(2);
  check("a tab that isn't holding the lock can't release it", ctx.record() && ctx.record().tabId === 1);

  await ctx.progress(2, { done: 99, total: 99 });
  check("...nor report progress into it", (ctx.record().done || 0) !== 99);

  await ctx.progress(1, { done: 7, total: 23 });
  check("the holder's own progress is recorded", ctx.record().done === 7 && ctx.record().total === 23);
  check("...along with the heartbeat that keeps the record alive", typeof ctx.record().updatedAt === "number");
}

async function twoClaimsInTheSameMomentDoNotBothWin() {
  // Naive race: both claims read "free" at their first await and both write themselves in.
  const ctx = loadWorker();
  const [a, b] = await Promise.all([ctx.claim(1, pageRun(1)), ctx.claim(2, pageRun(2))]);
  const winners = [a, b].filter((r) => r.ok).length;
  check("exactly one of two simultaneous claims wins", winners === 1, `${winners} winners`);
  check("...and the record belongs to that one", ctx.record().tabId === (a.ok ? 1 : 2));
}

async function anAbandonedRecordDoesNotWedgeTheLock() {
  const ctx = loadWorker();
  await ctx.claim(1, pageRun(1));
  ctx.closeTab(1);
  await settle();
  check("closing the running tab frees the lock straight away", ctx.record() === undefined);

  // tab still open but no longer reporting (reloaded, or content script threw)
  const stale = loadWorker({
    stored: {
      simplifierActiveRun: { ...pageRun(9), tabId: 1, startedAt: 1, updatedAt: 1 },
    },
  });
  const claimed = await stale.claim(2, pageRun(2));
  check("a record that stopped reporting is treated as nobody's", claimed.ok === true);
  check("...and the asking tab takes it over", stale.record().tabId === 2);

  // tab gone entirely, e.g. after a browser restart
  const orphaned = loadWorker({
    stored: { simplifierActiveRun: { ...pageRun(9), tabId: 42, startedAt: Date.now(), updatedAt: Date.now() } },
    liveTabs: [1, 2],
  });
  const takeover = await orphaned.claim(1, pageRun(1));
  check("a record whose tab no longer exists is treated the same way", takeover.ok === true);
}

async function stoppingTheRunningPageFreesTheLock() {
  const ctx = loadWorker();
  await ctx.claim(1, pageRun(1));
  const result = await ctx.stop();
  check("the worker asks the running tab to stop", ctx.messagesToTab.some((m) => m.tabId === 1 && m.message.cmd === "stopRun"));
  check("...reports that it did", result.stopped === true);
  check("...and names the run it stopped", result.run && result.run.title === "Page 1");
  check("...leaving the lock free", ctx.record() === undefined);

  const idle = loadWorker();
  const nothing = await idle.stop();
  check("stopping with nothing running says so rather than failing", nothing.stopped === false);
}

async function theLockIsReadableWithoutHoldingIt() {
  const ctx = loadWorker();
  const empty = await ctx.active();
  check("nothing running reads as nothing", empty.run === null);
  await ctx.claim(1, pageRun(1));
  const busy = await ctx.active();
  check("a run in progress is readable by anything that asks", busy.run && busy.run.sessionId === "s1");
}

// --- refused page

function loadPage({ holder }) {
  const dom = new JSDOM(
    `<!doctype html><html><head><title>This page</title></head><body><p>Residents are requested to position their receptacles at the kerbside.</p></body></html>`,
    { pretendToBeVisual: true, url: "https://example.com/mine" }
  );
  const { window } = dom;
  const sent = [];
  const storageListeners = [];

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
        onMessage: { addListener() {} },
        lastError: null,
        sendMessage(message, callback) {
          sent.push(message);
          if (message.cmd === "claimRun") {
            // refused, with the holder, as the worker answers
            if (callback) callback({ ok: false, holder });
            return;
          }
          if (message.cmd === "stopActiveRun") {
            if (callback) callback({ ok: true, stopped: true });
            return;
          }
          if (callback) callback({ ok: true });
        },
      },
      storage: {
        local: { get: (_keys, cb) => cb && cb({}) },
        onChanged: { addListener: (fn) => storageListeners.push(fn) },
      },
    },
  };
  context.globalThis = context;
  vm.createContext(context);
  ["shared/model-labels.js", "shared/active-run.js", "content.js"].forEach((f) =>
    vm.runInContext(fs.readFileSync(path.join(EXT, f), "utf8"), context)
  );

  return {
    sent,
    window,
    run: (expr) => vm.runInContext(expr, context),
    text: () => {
      const box = window.document.getElementById("simplify-notice");
      return box ? box.textContent : "";
    },
    box: () => window.document.getElementById("simplify-notice"),
    bar: () => window.document.querySelector("#simplify-notice .sn-bar-fill"),
    // simulates the worker writing a new holder record; the bar follows it without polling
    publishHolder: (next) =>
      storageListeners.forEach((fn) => fn({ simplifierActiveRun: { newValue: next } }, "local")),
  };
}

const HOLDER = {
  kind: "page",
  tabId: 1,
  sessionId: "other",
  url: "https://en.wikipedia.org/wiki/Recycling",
  title: "Recycling — Wikipedia",
  host: "en.wikipedia.org",
  done: 4,
  total: 20,
  startedAt: Date.now(),
  updatedAt: Date.now(),
};

async function theRefusedPageSaysWhatIsRunning() {
  const ctx = loadPage({ holder: HOLDER });
  ctx.run("toggleSimplification()");
  await settle();

  const text = ctx.text();
  check("the refused page says another page is already running", /Already simplifying another page/.test(text), text.slice(0, 60));
  check("...naming it, since it is in a tab the reader can't see", /Recycling — Wikipedia/.test(text));
  check("...and saying why one at a time", /one local service/i.test(text));
  check("it shows that run's real progress", ctx.bar() && ctx.bar().style.width === "20%", ctx.bar() && ctx.bar().style.width);
  check("it offers to stop that run", !!ctx.window.document.getElementById("simplify-stop-other"));
  check("...and to leave it alone", !!ctx.window.document.getElementById("simplify-wait"));
  check("this page starts nothing", !ctx.sent.some((m) => m.cmd === "fetchSimplify"));
  check("...and holds no session of its own", ctx.run("session.phase") === "idle");
}

async function theBarFollowsTheOtherTab() {
  const ctx = loadPage({ holder: HOLDER });
  ctx.run("toggleSimplification()");
  await settle();
  ctx.publishHolder({ ...HOLDER, done: 15, total: 20 });
  await settle();
  check("the other run's progress moves the bar", ctx.bar() && ctx.bar().style.width === "75%", ctx.bar() && ctx.bar().style.width);

  ctx.publishHolder(undefined);
  await settle();
  check("when that run ends, the notice says this page can go", /Click the icon to simplify this one/.test(ctx.text()), ctx.text());
}

async function stopThatPageStopsTheOtherOne() {
  const ctx = loadPage({ holder: HOLDER });
  ctx.run("toggleSimplification()");
  await settle();
  ctx.window.document.getElementById("simplify-stop-other").click();
  await settle();
  check("'Stop that page' asks the worker to stop the run", ctx.sent.some((m) => m.cmd === "stopActiveRun"));
  check("...and says what happened", /Stopped the other page/.test(ctx.text()), ctx.text());
}

async function waitJustDismisses() {
  const ctx = loadPage({ holder: HOLDER });
  ctx.run("toggleSimplification()");
  await settle();
  ctx.window.document.getElementById("simplify-wait").click();
  await settle();
  check("'Wait' takes the notice away", !ctx.box());
  check("...and queues nothing", !ctx.sent.some((m) => m.cmd === "fetchSimplify"));
  check("...and stops nothing", !ctx.sent.some((m) => m.cmd === "stopActiveRun"));
}

async function main() {
  await oneTabAtATime();
  await onlyTheHolderCanReleaseOrReport();
  await twoClaimsInTheSameMomentDoNotBothWin();
  await anAbandonedRecordDoesNotWedgeTheLock();
  await stoppingTheRunningPageFreesTheLock();
  await theLockIsReadableWithoutHoldingIt();
  await theRefusedPageSaysWhatIsRunning();
  await theBarFollowsTheOtherTab();
  await stopThatPageStopsTheOtherOne();
  await waitJustDismisses();

  let failed = 0;
  results.forEach(([name, ok, detail]) => {
    if (!ok) failed += 1;
    console.log(`${ok ? "PASS" : "FAIL"}  ${name}${ok || !detail ? "" : ` — ${detail}`}`);
  });
  console.log(`\n${results.length - failed}/${results.length} checks passed`);
  process.exit(failed === 0 ? 0 : 1);
}

main();
