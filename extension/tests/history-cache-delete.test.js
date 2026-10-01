// Deleting a run's record deletes the cached simplifications that run produced.
//
//   cd extension/tests && npm install && node history-cache-delete.test.js
//
// Otherwise a deleted page would be served again from the cache, unlogged, on its next
// simplification. Both delete paths ("this page", "all") have separate call sites.
//
// Same harness as history-view.test.js, plus a callback-form chrome.runtime.sendMessage
// (how the page asks the service worker to drop a page's cache).
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

const results = [];
const check = (name, condition, detail = "") => results.push([name, !!condition, detail]);

const entry = { input: "a sentence", modelResult: "a simpler one", output: "a simpler one", changed: true, cached: false };
const page = (n) => ({
  sessionId: `session-${n}`,
  timestamp: 1700000000000 + n,
  title: `Page ${n}`,
  url: `https://example.com/${n}`,
  modelKey: "finetuned",
  entries: [entry],
});

function load(pages, { confirmed = true, dropped = 7, workerReachable = true } = {}) {
  const dom = new JSDOM(PAGE_HTML, { pretendToBeVisual: true, url: "chrome-extension://test/history.html" });
  const { window } = dom;
  const store = { simplifyHistory: pages };
  const sent = [];

  const chrome = {
    runtime: {
      lastError: null,
      sendMessage: (message, callback) => {
        sent.push(message);
        if (typeof callback !== "function") return Promise.resolve({ ok: false });
        if (!workerReachable) {
          chrome.runtime.lastError = { message: "no receiving end" };
          callback(undefined);
          chrome.runtime.lastError = null;
          return undefined;
        }
        callback({ dropped, pages: message.pages ? message.pages.length : 0 });
        return undefined;
      },
    },
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
  };

  const confirms = [];
  const context = {
    window,
    document: window.document,
    Blob: window.Blob,
    URL: window.URL,
    navigator: { clipboard: { writeText: () => Promise.resolve() } },
    confirm: (text) => {
      confirms.push(text);
      return confirmed;
    },
    setTimeout,
    clearTimeout,
    console,
    chrome,
  };
  context.globalThis = context;
  vm.createContext(context);
  SOURCES.forEach((source) => vm.runInContext(source, context));

  return {
    store,
    sent,
    confirms,
    notice: () => window.document.getElementById("page-notice").textContent,
    run: (expr) => vm.runInContext(expr, context),
  };
}

const tick = (ms = 10) => new Promise((resolve) => setTimeout(resolve, ms));

async function deletingOnePageDropsItsCache() {
  const ctx = load([page(1), page(2)]);
  await ctx.run(`deletePage("session-1")`);
  await tick();

  const ask = ctx.sent.find((m) => m.cmd === "dropCachedPages");
  check("deleting one page asks the worker to drop its cached simplifications", !!ask);
  check("...naming that page and only it", ask && JSON.stringify(ask.pages) === JSON.stringify(["session-1"]));
  check("...and the log loses that page", !ctx.store.simplifyHistory.some((p) => p.sessionId === "session-1"));
  check("...while the other page stays", ctx.store.simplifyHistory.length === 1);
  check("the confirmation says the cache goes too", /cached/i.test(ctx.confirms[0] || ""));
  check("...and the notice reports how much did", /7 cached simplifications dropped/.test(ctx.notice()), ctx.notice());
}

async function deletingEverythingDropsEveryPagesCache() {
  const ctx = load([page(1), page(2), page(3)]);
  await ctx.run(`deleteAllHistory()`);
  await tick();

  const ask = ctx.sent.find((m) => m.cmd === "dropCachedPages");
  check("deleting the whole log drops every logged page's cache", !!ask);
  check(
    "...naming all of them, read before the log was emptied",
    ask && JSON.stringify(ask.pages) === JSON.stringify(["session-1", "session-2", "session-3"])
  );
  check("...and the log is gone", ctx.store.simplifyHistory === undefined);
  check("the confirmation says the cache goes too", /cached/i.test(ctx.confirms[0] || ""));
}

async function decliningTheConfirmationChangesNothing() {
  const ctx = load([page(1)], { confirmed: false });
  await ctx.run(`deletePage("session-1")`);
  await ctx.run(`deleteAllHistory()`);
  await tick();
  check("a declined confirmation deletes nothing", ctx.store.simplifyHistory.length === 1);
  check("...and drops no cache either", !ctx.sent.some((m) => m.cmd === "dropCachedPages"));
}

async function nothingToDropIsReportedHonestly() {
  // Zero has several causes (restarted backend with in-memory cache, every unit skipped as
  // too short, entries still held by a remaining page), so the notice names none.
  const ctx = load([page(1)], { dropped: 0 });
  await ctx.run(`deletePage("session-1")`);
  await tick();
  check("nothing dropped says so without inventing a reason", /belonged only to it/.test(ctx.notice()), ctx.notice());

  // unreachable worker: same outcome, the deletion has already happened
  const unreachable = load([page(1)], { workerReachable: false });
  await unreachable.run(`deletePage("session-1")`);
  await tick();
  check("an unreachable worker still deletes the log entry", unreachable.store.simplifyHistory.length === 0);
  check("...and says nothing was dropped rather than failing", /belonged only to it/.test(unreachable.notice()), unreachable.notice());
}

async function anEmptyLogAsksNothing() {
  const ctx = load([]);
  await ctx.run(`deleteAllHistory()`);
  await tick();
  check("deleting an empty log sends no eviction request", !ctx.sent.some((m) => m.cmd === "dropCachedPages"));
}

async function main() {
  await deletingOnePageDropsItsCache();
  await deletingEverythingDropsEveryPagesCache();
  await decliningTheConfirmationChangesNothing();
  await nothingToDropIsReportedHonestly();
  await anEmptyLogAsksNothing();

  let failed = 0;
  results.forEach(([name, ok, detail]) => {
    if (!ok) failed += 1;
    console.log(`${ok ? "PASS" : "FAIL"}  ${name}${ok || !detail ? "" : ` — ${detail}`}`);
  });
  console.log(`\n${results.length - failed}/${results.length} checks passed`);
  process.exit(failed === 0 ? 0 : 1);
}

main();
