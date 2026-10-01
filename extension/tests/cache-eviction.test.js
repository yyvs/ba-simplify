// A cached simplification must not outlive its History log record. Checked in the service
// worker, the one place that caps the log and can reach the backend.
//
//   cd extension/tests && npm install && node cache-eviction.test.js
//
// Same vm + recording chrome mock harness as background-menu.test.js.
//
// The log caps at 20 pages; the backend cache is bounded only by CACHE_MAX. The cap drives
// an eviction request, each /simplify carries its page so there is a key to evict by, and a
// stopped backend (in-memory cache already empty) stays silent.
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const EXT = path.join(__dirname, "..");

function load({ stored = {}, fetchImpl = null } = {}) {
  const store = { ...stored };
  const fetches = [];
  const notifications = [];
  const listeners = {};

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
      sendMessage: (_tabId, _message, cb) => cb && cb(),
      // lock release on tab close, and lock-holder existence check (shared/active-run.js)
      onRemoved: { addListener() {} },
      get: (tabId) => Promise.resolve({ id: tabId }),
    },
    windows: { update: () => Promise.resolve() },
    notifications: { create: (n) => notifications.push(n) },
  };

  const context = {
    chrome,
    console: { log() {}, warn() {}, error() {} },
    fetch: (url, options) => {
      fetches.push({ url, options, body: options && options.body ? JSON.parse(options.body) : null });
      if (fetchImpl) return fetchImpl(url, options);
      return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ dropped: 0, pages: 0 }) });
    },
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

  return {
    store,
    fetches,
    notifications,
    // called for every flushed batch from a content script
    record: (page, entries, session) =>
      vm.runInContext(
        `recordHistory(${JSON.stringify(page)}, ${JSON.stringify(entries)}, ${JSON.stringify(session || null)})`,
        context
      ),
    message: (msg, respond = () => {}) => listeners.message(msg, {}, respond),
    run: (expr) => vm.runInContext(expr, context),
  };
}

const results = [];
const check = (name, condition) => results.push([name, !!condition]);
const settle = () => new Promise((resolve) => setTimeout(resolve, 20));

const pageMeta = (n) => ({ sessionId: `page-${n}`, url: `https://example.org/${n}`, title: `Page ${n}`, timestamp: n });
const entries = [{ input: "a sentence", output: "a simpler sentence" }];

// read from the worker so the two can't drift
const maxPages = (ctx) => ctx.run("MAX_HISTORY_PAGES");

async function fillLogToItsCap(ctx) {
  for (let n = 1; n <= maxPages(ctx); n += 1) {
    await ctx.record(pageMeta(n), entries);
  }
}

async function theLogsCapDrivesTheCacheEviction() {
  const ctx = load();
  await fillLogToItsCap(ctx);
  await settle();
  check("filling the log to its cap evicts nothing", ctx.fetches.length === 0);
  check("...and the log holds exactly its cap", ctx.store.simplifyHistory.length === maxPages(ctx));

  await ctx.record(pageMeta(maxPages(ctx) + 1), entries);
  await settle();
  const call = ctx.fetches.find((f) => f.url.includes("/cache/pages/delete"));
  check("one page past the cap asks the backend to drop the page that fell out", !!call);
  check("...by POST, as that endpoint requires", call && call.options.method === "POST");
  check("...at the API's origin, not the demo server's", call && call.url.startsWith("http://127.0.0.1:8000"));
  check("...naming the evicted page and only it", call && JSON.stringify(call.body) === JSON.stringify({ pages: ["page-1"] }));
  check("...and the log still holds its cap", ctx.store.simplifyHistory.length === maxPages(ctx));
  check("...with the oldest page gone from it", !ctx.store.simplifyHistory.some((p) => p.sessionId === "page-1"));
  // automatic bookkeeping; a notification would be noise on every 21st page
  check("...silently, since nobody asked for it", ctx.notifications.length === 0);
}

async function aRevisitedPageDoesNotEvictAnything() {
  // Counted per page load, not per site; a later batch extends its page's record.
  const ctx = load();
  await fillLogToItsCap(ctx);
  await ctx.record(pageMeta(1), entries);
  await settle();
  check("a second batch from a page already logged evicts nothing", ctx.fetches.length === 0);
  check("...and doesn't grow the log", ctx.store.simplifyHistory.length === maxPages(ctx));
  check("...it extends that page's entries", ctx.store.simplifyHistory[0].entries.length === 2);
}

async function everyRequestCarriesThePageItBelongsTo() {
  const ctx = load({
    stored: { simplifierModel: "finetuned" },
    fetchImpl: () => Promise.resolve({ ok: true, json: () => Promise.resolve({ simplified: "x" }) }),
  });
  ctx.message({ cmd: "fetchSimplify", text: "a sentence", page: "page-7" }, () => {});
  await settle();
  const call = ctx.fetches.find((f) => f.url.includes("/simplify"));
  check("a simplify request names the page it belongs to", call && call.body.page === "page-7");
  check("...without that changing which model runs", call && call.body.model === "finetuned");

  // History's comparison runs and the toggle preflight have no page; the backend keeps
  // those entries unowned
  const noPage = load({ stored: { simplifierModel: "finetuned" }, fetchImpl: () => Promise.resolve({ ok: true, json: () => Promise.resolve({}) }) });
  noPage.message({ cmd: "fetchSimplify", text: "a sentence" }, () => {});
  await settle();
  const anonymous = noPage.fetches.find((f) => f.url.includes("/simplify"));
  check("a request with no page says so explicitly rather than omitting the field", anonymous && anonymous.body.page === null);
}

async function anUnreachableBackendIsNotAFailure() {
  // in-memory, per-process cache: a stopped backend has nothing to drop
  const ctx = load({ fetchImpl: () => Promise.reject(new Error("connection refused")) });
  await fillLogToItsCap(ctx);
  await ctx.record(pageMeta(99), entries);
  await settle();
  check("a stopped backend raises nothing at the user", ctx.notifications.length === 0);
  check("...and the page still leaves the log", ctx.store.simplifyHistory.length === maxPages(ctx));

  const refused = load({ fetchImpl: () => Promise.resolve({ ok: false, status: 500, json: () => Promise.resolve({}) }) });
  await fillLogToItsCap(refused);
  await refused.record(pageMeta(99), entries);
  await settle();
  check("a backend that answers with an error is just as quiet", refused.notifications.length === 0);
}

async function main() {
  await theLogsCapDrivesTheCacheEviction();
  await aRevisitedPageDoesNotEvictAnything();
  await everyRequestCarriesThePageItBelongsTo();
  await anUnreachableBackendIsNotAFailure();

  let failed = 0;
  results.forEach(([name, ok]) => {
    if (!ok) failed += 1;
    console.log(`${ok ? "PASS" : "FAIL"}  ${name}`);
  });
  console.log(`\n${results.length - failed}/${results.length} checks passed`);
  process.exit(failed === 0 ? 0 : 1);
}

main();
