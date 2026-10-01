// Service worker toolbar-icon menu: layout, what each entry reports, what each does.
//
//   cd extension/tests && npm install && node background-menu.test.js
//
// background.js runs in a vm context with a recording mock of every chrome API it uses
// (including importScripts, which loads the shared files). fetch is stubbed.
//
// Contract:
// - Six top-level entries, no separators. Chrome silently drops items past the sixth, and
//   separators count as items (this dropped the last entry twice before).
// - Home shows what is running, otherwise whether the backend was last seen up.
// - Selection submenu: titled with the active selection, every model as a disabled read-out
//   with a tick on the active one, and one entry that opens the panel in the page (falling
//   back to an extension page where no content script can run).
// - Tools holds actions rather than pages; Stop is disabled when nothing is running.

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const EXT = path.join(__dirname, "..");

function load({
  stored = {},
  tab = { id: 7, url: "https://en.wikipedia.org/wiki/Nikola_Tesla" },
  // for the "refocus the homepage" path
  openTabs = [],
  // only clearing the backend cache uses the network; rejecting by default catches any other fetch
  fetchImpl = () => Promise.reject(new Error("the menu must not reach the network")),
} = {}) {
  const store = { ...stored };
  const updatedTabs = [];
  const menus = new Map();
  const created = [];
  const messagesToTab = [];
  const notifications = [];
  const openedTabs = [];
  const fetches = [];
  const listeners = {};

  const chrome = {
    runtime: {
      lastError: null,
      getURL: (p) => `chrome-extension://test/${p}`,
      onInstalled: { addListener: (fn) => (listeners.installed = fn) },
      onMessage: { addListener: (fn) => (listeners.message = fn) },
      // unused: the worker messages tabs, not itself
      sendMessage() {},
    },
    contextMenus: {
      removeAll: (cb) => cb && cb(),
      create: (props) => {
        menus.set(props.id, { ...props });
        created.push(props);
      },
      update: (id, props, cb) => {
        if (menus.has(id)) Object.assign(menus.get(id), props);
        if (cb) cb();
      },
      remove: (id, cb) => {
        menus.delete(id);
        if (cb) cb();
      },
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
          if (listeners.storageChange) {
            const changes = {};
            Object.keys(values).forEach((k) => (changes[k] = { newValue: values[k] }));
            listeners.storageChange(changes, "local");
          }
          return Promise.resolve();
        },
        remove(keys) {
          const names = Array.isArray(keys) ? keys : [keys];
          names.forEach((k) => delete store[k]);
          if (listeners.storageChange) {
            const changes = {};
            names.forEach((k) => (changes[k] = { newValue: undefined }));
            listeners.storageChange(changes, "local");
          }
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
    // promise form, as MV3 serves them
    tabs: {
      create: (props) => {
        openedTabs.push(props);
        return Promise.resolve({ id: 99, ...props });
      },
      query: (query) => Promise.resolve(openTabs.filter((t) => t.url === query.url)),
      update: (id, props) => {
        updatedTabs.push({ id, ...props });
        return Promise.resolve();
      },
      // lock release on tab close, and lock-holder existence check (shared/active-run.js)
      onRemoved: { addListener() {} },
      get: (tabId) => Promise.resolve({ id: tabId }),
      sendMessage: (tabId, message, cb) => {
        messagesToTab.push({ tabId, message });
        // only stopRun expects an answer
        if (cb) cb(message.cmd === "stopRun" ? { ok: true, stopped: true } : undefined);
      },
    },
    windows: { update: () => Promise.resolve() },
    notifications: { create: (n) => notifications.push(n) },
  };

  const context = {
    chrome,
    console: { log() {}, warn() {}, error() {} },
    fetch: (url, options) => {
      fetches.push({ url, options });
      return fetchImpl(url, options);
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
    menus,
    created,
    messagesToTab,
    notifications,
    openedTabs,
    updatedTabs,
    fetches,
    tab,
    install: () => listeners.installed({ reason: "install" }),
    clickMenu: (id, withTab = tab) => listeners.menuClick({ menuItemId: id }, withTab),
    run: (expr) => vm.runInContext(expr, context),
  };
}

// read from the mock so the two can't drift apart
const chrome_limit = (ctx) => ctx.run("chrome.contextMenus.ACTION_MENU_TOP_LEVEL_LIMIT");

const results = [];
const check = (name, condition, detail = "") => results.push([name, !!condition, detail]);
const settle = () => new Promise((resolve) => setTimeout(resolve, 20));

// /health answer as the picker remembers it; narrows the model list to loaded keys and names them
const HEALTH = {
  status: "ok",
  model_loaded: true,
  models_loaded: ["online", "finetuned", "llm_7b"],
  model_names: {
    online: "eilamc14/bart-large-text-simplification",
    finetuned: "yunvs/bart-base-wikilarge-simplification",
    llm_7b: "qwen2.5:7b-instruct-q4_K_M",
  },
  granularities: { online: "sentence_by_sentence", finetuned: "sentence_by_sentence", llm_7b: "sentence_by_sentence" },
  methods: { online: "fine_tuned_seq2seq", finetuned: "fine_tuned_seq2seq", llm_7b: "prompted_llm" },
  audience_models: ["llm_7b"],
  audiences: [{ value: "children", label: "Children · 8–12" }],
};

const runningPage = (extra = {}) => ({
  kind: "page",
  tabId: 7,
  sessionId: "s1",
  url: "https://en.wikipedia.org/wiki/Recycling",
  title: "Recycling — Wikipedia",
  host: "en.wikipedia.org",
  done: 4,
  total: 20,
  startedAt: Date.now(),
  updatedAt: Date.now(),
  ...extra,
});

async function theLayoutFitsAndIsGrouped() {
  const ctx = load({ stored: { simplifierModel: "finetuned" } });
  ctx.install();
  await settle();

  const topLevel = ctx.created.filter((m) => !m.parentId);
  check("the menu fits inside Chrome's top-level limit", topLevel.length <= chrome_limit(ctx), `${topLevel.length}`);
  // separators count toward the limit
  check("...with every slot spent on an entry rather than a separator", !topLevel.some((m) => m.type === "separator"));
  check(
    "the six are: where to look, what is selected, the log, the tools, the guide, the fixtures",
    topLevel.map((m) => m.id).join() ===
      [
        "simplifier-home",
        "simplifier-selection",
        "simplifier-history",
        "simplifier-tools",
        "simplifier-instructions",
        "simplifier-demo",
      ].join(),
    topLevel.map((m) => m.id).join()
  );

  const nested = (parent) => ctx.created.filter((m) => m.parentId === parent).map((m) => m.id);
  check("the selection submenu ends with the one entry that changes anything", nested("simplifier-selection").pop() === "simplifier-picker");
  check(
    "...separated from the read-outs above it",
    nested("simplifier-selection").includes("simplifier-selection-separator")
  );
  check(
    "Tools is Stop, then the two evaluation entries, then the cache",
    nested("simplifier-tools").join() ===
      [
        "simplifier-stop",
        "simplifier-tools-separator",
        "simplifier-report",
        "simplifier-compare",
        "simplifier-tools-cache-separator",
        "simplifier-clear-cache",
      ].join(),
    nested("simplifier-tools").join()
  );
  check("the nested entries are only affordable because they are nested", ctx.created.length > topLevel.length);
  check("the top-level entries name themselves plainly", ["History", "Instructions", "Demo", "Tools"].every((title) =>
    ctx.created.some((m) => m.title === title)
  ));
}

async function homeReportsTheCurrentStatus() {
  const fresh = load({ stored: { simplifierModel: "finetuned" } });
  fresh.install();
  await settle();
  // a fresh profile has never checked; "Backend unknown" would read as a claim about the backend
  check("with nothing known, Home is just Home", fresh.menus.get("simplifier-home").title === "Home", fresh.menus.get("simplifier-home").title);

  const up = load({ stored: { simplifierModel: "finetuned", simplifierBackendStatus: { ok: true, at: Date.now() } } });
  up.install();
  await settle();
  check("a backend last seen up says so", up.menus.get("simplifier-home").title === "Home · Backend ready", up.menus.get("simplifier-home").title);

  const down = load({ stored: { simplifierModel: "finetuned", simplifierBackendStatus: { ok: false, at: Date.now() } } });
  down.install();
  await settle();
  check("...as does one that wasn't", down.menus.get("simplifier-home").title === "Home · Backend not reachable", down.menus.get("simplifier-home").title);

  const busy = load({
    stored: { simplifierModel: "finetuned", simplifierBackendStatus: { ok: true, at: Date.now() }, simplifierActiveRun: runningPage() },
  });
  busy.install();
  await settle();
  check(
    "a run in progress takes precedence over the backend line",
    busy.menus.get("simplifier-home").title === "Home · Simplifying Recycling — Wikipedia",
    busy.menus.get("simplifier-home").title
  );

  const comparing = load({
    stored: {
      simplifierModel: "finetuned",
      simplifierActiveRun: runningPage({ kind: "compare", models: 3, modelIndex: 2 }),
    },
  });
  comparing.install();
  await settle();
  check(
    "...and a comparison says how far through its models it is",
    /Comparing 3 models on Recycling — Wikipedia \(2 of 3\)/.test(comparing.menus.get("simplifier-home").title),
    comparing.menus.get("simplifier-home").title
  );
}

async function theSelectionSubmenuReportsAndLists() {
  const ctx = load({ stored: { simplifierModel: "finetuned" } });
  ctx.install();
  await settle();
  check(
    "the submenu is titled with what is active",
    ctx.menus.get("simplifier-selection").title === "Fine-tuned model · Sentence by sentence",
    ctx.menus.get("simplifier-selection").title
  );

  const models = ctx.created.filter((m) => m.id.startsWith("simplifier-model-"));
  check("every selectable model gets a line", models.length === ctx.run("MODEL_KEYS.length"));
  // not pressable: a menu item can't describe a model, show the granularity control, or say
  // why an entry is unavailable
  check("...all of them read-outs rather than controls", models.every((m) => m.enabled === false));
  check("...none of them a checkbox that couldn't be unchecked", !models.some((m) => m.type === "checkbox"));

  const active = ctx.menus.get("simplifier-model-finetuned");
  check("the active model is ticked", active.title.includes("✓"), active.title);
  check("...and named with the unit of text it reads", /· sentence by sentence/.test(active.title), active.title);
  check("nothing else is ticked", ctx.created.filter((m) => (ctx.menus.get(m.id).title || "").includes("✓")).length === 1);
  check("the entry that changes the selection says what it changes", ctx.menus.get("simplifier-picker").title === "Change simplification method");
}

async function theModelListNarrowsOnceHealthIsKnown() {
  const ctx = load({ stored: { simplifierModel: "finetuned" } });
  ctx.install();
  await settle();
  // cached labels deliberately omit models_loaded (it goes stale), so every key is listed
  check("with no health answer, every key is listed", ctx.created
    .filter((m) => m.id.startsWith("simplifier-model-"))
    .every((m) => ctx.menus.get(m.id).visible !== false));

  await ctx.run(`rememberHealth(${JSON.stringify(HEALTH)}); refreshMenu();`);
  await settle();
  const shown = ctx.created
    .filter((m) => m.id.startsWith("simplifier-model-"))
    .filter((m) => ctx.menus.get(m.id).visible !== false)
    .map((m) => m.id.replace("simplifier-model-", ""));
  check("once health answers, only the loaded keys are listed", JSON.stringify(shown) === JSON.stringify(HEALTH.models_loaded), shown.join());
  check(
    "...and a served tag names its own model",
    ctx.menus.get("simplifier-model-llm_7b").title.startsWith("LLM (Qwen2.5 7B)"),
    ctx.menus.get("simplifier-model-llm_7b").title
  );
}

async function theSelectionTitleFollowsTheSelection() {
  const ctx = load({ stored: { simplifierModel: "finetuned" } });
  ctx.install();
  await settle();

  // selection changes from anywhere (panel, page toolbar, Error page) arrive as storage changes
  await ctx.run(`writeSelection({ family: "llm_3b", granularity: "whole_sections", audience: "children" }, null)`);
  await settle();
  check(
    "changing the selection updates the title, audience included where it applies",
    ctx.menus.get("simplifier-selection").title === "Ollama Open LLM · 3B · Whole sections · Children · 8–12",
    ctx.menus.get("simplifier-selection").title
  );
  check("...and moves the tick", ctx.menus.get("simplifier-model-llm_doc_3b").title.includes("✓"));

  await ctx.run(`writeSelection({ family: "finetuned" }, null)`);
  await settle();
  check(
    "switching to a model that ignores the audience drops it from the title",
    ctx.menus.get("simplifier-selection").title === "Fine-tuned model · Whole sections",
    ctx.menus.get("simplifier-selection").title
  );
}

async function cachedLabelsSurviveAWorkerRestart() {
  // Written by rememberHealth(). An MV3 worker restarts after ~30s idle and must still find
  // these rather than fall back to the generic model name.
  const ctx = load({
    stored: {
      simplifierModel: "llm_7b",
      simplifierHealthLabels: {
        model_names: { llm_7b: "qwen2.5:7b-instruct-q4_K_M" },
        audience_models: ["llm_7b", "llm_3b"],
        audiences: [{ value: "children", label: "Children · 8–12" }],
      },
    },
  });
  ctx.install();
  await settle();
  check(
    "a worker with no live health still names the served model",
    ctx.menus.get("simplifier-selection").title.startsWith("Ollama Qwen2.5 · 7B"),
    ctx.menus.get("simplifier-selection").title
  );
}

async function stopIsOnlyLiveWhenSomethingIsRunning() {
  const idle = load({ stored: { simplifierModel: "finetuned" } });
  idle.install();
  await settle();
  check("with nothing running, Stop is disabled", idle.menus.get("simplifier-stop").enabled === false);
  check("...and named for the thing it would stop", idle.menus.get("simplifier-stop").title === "Stop current process");

  idle.clickMenu("simplifier-stop");
  await settle();
  // the menu can be stale by the time a click lands
  check("pressing it anyway says there was nothing to stop", idle.notifications.some((n) => /Nothing to stop/.test(n.title)));

  const busy = load({ stored: { simplifierModel: "finetuned", simplifierActiveRun: runningPage() } });
  busy.install();
  await settle();
  check("with a run in progress, Stop is live", busy.menus.get("simplifier-stop").enabled === true);
  check(
    "...and names the run it would stop, with the page's own title untouched",
    busy.menus.get("simplifier-stop").title === "Stop: Simplifying Recycling — Wikipedia",
    busy.menus.get("simplifier-stop").title
  );

  busy.clickMenu("simplifier-stop");
  await settle();
  check("pressing it asks the running tab to stop", busy.messagesToTab.some((m) => m.tabId === 7 && m.message.cmd === "stopRun"));
  check("...and says so", busy.notifications.some((n) => n.title === "Stopped"), JSON.stringify(busy.notifications));
  check("...leaving the lock free", busy.store.simplifierActiveRun === undefined);
}

async function toolsOpensTheComparisonPanelInThePage() {
  const ctx = load({ stored: { simplifierModel: "finetuned" } });
  ctx.install();
  await settle();
  ctx.clickMenu("simplifier-compare");
  await settle();
  const ask = ctx.messagesToTab.find((m) => m.message.cmd === "showCompareModels");
  check("Compare asks the page to open the comparison panel", !!ask);
  check("...in the tab it was clicked from", ask && ask.tabId === 7);
  check("...and opens no tab of its own", ctx.openedTabs.length === 0);

  // no extension-page fallback (unlike the picker): a comparison needs the page's own text
  const unsupported = load({ stored: { simplifierModel: "finetuned" }, tab: { id: 9, url: "chrome://settings" } });
  unsupported.install();
  await settle();
  unsupported.clickMenu("simplifier-compare");
  await settle();
  check("a page with no text to read says so instead", unsupported.notifications.some((n) => /Can't compare models here/.test(n.title)));
  check("...and opens nothing", unsupported.openedTabs.length === 0);
}

async function generateReportKeepsThisPagesRun() {
  const page = {
    sessionId: "s1",
    url: "https://en.wikipedia.org/wiki/Nikola_Tesla",
    title: "Nikola Tesla",
    modelKey: "finetuned",
    granularity: "sentence_by_sentence",
    entries: [{ input: "a", output: "b" }],
  };
  const ctx = load({ stored: { simplifierModel: "finetuned", simplifyHistory: [page] } });
  ctx.install();
  await settle();
  ctx.clickMenu("simplifier-report");
  await settle();

  check("the run logged for this page is kept as a report", (ctx.store.simplifyReports || []).length === 1);
  check("...naming the run it was made from", ctx.store.simplifyReports[0].sourceSessionId === "s1");
  check("...and the log is left alone", ctx.store.simplifyHistory.length === 1);
  // opened at the new report, not just the list
  const reportId = ctx.store.simplifyReports[0].reportId;
  check(
    "the Reports page is opened at the report it just made",
    ctx.openedTabs.some((t) => t.url.endsWith(`reports.html#report-${reportId}`)),
    JSON.stringify(ctx.openedTabs.map((t) => t.url))
  );
  check("...saying which run was kept", ctx.notifications.some((n) => /Report saved/.test(n.title)));

  ctx.clickMenu("simplifier-report");
  await settle();
  check("pressing it again updates that report rather than adding another", ctx.store.simplifyReports.length === 1);
  check("...and says which of the two happened", ctx.notifications.some((n) => /Report updated/.test(n.title)));

  const nothing = load({ stored: { simplifierModel: "finetuned" } });
  nothing.install();
  await settle();
  nothing.clickMenu("simplifier-report");
  await settle();
  check("a page never simplified adds no report", nothing.store.simplifyReports === undefined);
  check("...but still opens the reports there are", nothing.openedTabs.some((t) => t.url.endsWith("reports.html")));
  check("...explaining why this page added none", nothing.notifications.some((n) => /Nothing to report on this page/.test(n.title)));
}

async function clickingOpensThePanelInThePage() {
  const ctx = load({ stored: { simplifierModel: "finetuned" } });
  ctx.install();
  await settle();
  ctx.clickMenu("simplifier-picker");
  check("clicking the change entry asks the page to open the panel", ctx.messagesToTab.length === 1);
  check("...in the tab it was clicked from", ctx.messagesToTab[0].tabId === 7);
  check("...with the picker command", ctx.messagesToTab[0].message.cmd === "showModelPicker");
  check("...and opens no tab of its own", ctx.openedTabs.length === 0);
}

async function unsupportedPagesStillReachThePicker() {
  // chrome://, about:, file://, and the extension's own pages (e.g. the homepage itself)
  const ctx = load({ stored: { simplifierModel: "finetuned" }, tab: { id: 9, url: "chrome://settings" } });
  ctx.install();
  await settle();
  ctx.clickMenu("simplifier-picker");
  await settle();
  check("a page no content script can run on gets no message", ctx.messagesToTab.length === 0);
  check("...but the click still ends at the picker", ctx.openedTabs.length === 1);
  check(
    "...by opening the homepage with the hash its toolbar acts on",
    ctx.openedTabs[0].url.endsWith("home.html#picker")
  );
  check("...with no notification, since nothing failed", ctx.notifications.length === 0);
}

async function anAlreadyOpenHomepageIsReused() {
  const base = "chrome-extension://test/home.html";
  const ctx = load({
    stored: { simplifierModel: "finetuned" },
    tab: { id: 9, url: "chrome://settings" },
    openTabs: [{ id: 4, windowId: 1, url: base }],
  });
  ctx.install();
  await settle();
  ctx.clickMenu("simplifier-picker");
  await settle();
  check("an open homepage is reused rather than duplicated", ctx.openedTabs.length === 0);
  check("...refocused", ctx.updatedTabs.some((t) => t.id === 4 && t.active === true));
  check("...and navigated to the hash so the panel opens there", ctx.updatedTabs.some((t) => t.url === `${base}#picker`));
}

async function navEntriesOpenTheirPages() {
  const ctx = load({ stored: { simplifierModel: "finetuned" } });
  ctx.install();
  await settle();

  ctx.clickMenu("simplifier-instructions");
  await settle();
  check("the setup guide opens from the menu", ctx.openedTabs.some((t) => t.url.endsWith("instructions.html")));
  // {reason} triggers the first-install greeting
  check("...without the first-install greeting", ctx.store.instructions && ctx.store.instructions.reason === undefined);

  ctx.clickMenu("simplifier-history");
  await settle();
  check("the history opens from the menu", ctx.openedTabs.some((t) => t.url.endsWith("history.html")));

  ctx.clickMenu("simplifier-home");
  await settle();
  check("Home opens from the menu", ctx.openedTabs.some((t) => t.url.endsWith("home.html")));
  // a "home" key would sit in the storage the other pages watch
  check("...without writing a payload it has no use for", !("home" in ctx.store));
}

// Must empty the backend's cache, not the History log beside it (the History page's
// button once deleted the log).
async function theCacheButtonClearsTheBackendCache() {
  const history = [{ sessionId: "s1", url: "https://example.org", entries: [{ input: "a", output: "b" }] }];
  const ctx = load({
    stored: { simplifierModel: "finetuned", simplifyHistory: history },
    fetchImpl: () => Promise.resolve({ ok: true, json: () => Promise.resolve({ cleared: 3 }) }),
  });
  ctx.install();
  await settle();

  ctx.clickMenu("simplifier-clear-cache");
  await settle();
  const call = ctx.fetches.find((f) => f.url.includes("/cache/clear"));
  check("the cache entry asks the backend to clear its cache", !!call);
  check("...by POST, as that endpoint requires", call && call.options.method === "POST");
  check("...at the API's origin, not the demo server's", call && call.url.startsWith("http://127.0.0.1:8000"));
  check("...and leaves the History log alone", ctx.store.simplifyHistory === history);
  check("...reporting what happened, since a menu item has no page to write a notice into", ctx.notifications.length === 1);
  check("...including how much was dropped", ctx.notifications[0].message.startsWith("3 entries dropped"));
  // menu items have no tooltip, so the notification carries this
  check("...and what it did not clear", ctx.notifications[0].message.includes("History log is untouched"));
  check("...and opens nothing", ctx.openedTabs.length === 0);

  // singular: not "1 entries"
  const one = load({
    stored: { simplifierModel: "finetuned" },
    fetchImpl: () => Promise.resolve({ ok: true, json: () => Promise.resolve({ cleared: 1 }) }),
  });
  one.install();
  await settle();
  one.clickMenu("simplifier-clear-cache");
  await settle();
  check("one dropped entry is one entry", one.notifications[0].message.startsWith("1 entry dropped"));
}

async function aFailedCacheClearSaysSo() {
  // the default fetch rejects, as a stopped backend would
  const ctx = load({ stored: { simplifierModel: "finetuned" } });
  ctx.install();
  await settle();
  ctx.clickMenu("simplifier-clear-cache");
  await settle();
  check("a stopped backend is reported rather than passed over in silence", ctx.notifications.length === 1);
  check("...saying which script starts it", ctx.notifications[0].message.includes("run_dev.sh"));
  check("...without redirecting anywhere", ctx.openedTabs.length === 0);

  const refused = load({
    stored: { simplifierModel: "finetuned" },
    fetchImpl: () => Promise.resolve({ ok: false, status: 500, json: () => Promise.resolve({}) }),
  });
  refused.install();
  await settle();
  refused.clickMenu("simplifier-clear-cache");
  await settle();
  check("a backend that answers with an error says which", refused.notifications[0].message.includes("500"));
}

async function main() {
  await theLayoutFitsAndIsGrouped();
  await homeReportsTheCurrentStatus();
  await theSelectionSubmenuReportsAndLists();
  await theModelListNarrowsOnceHealthIsKnown();
  await theSelectionTitleFollowsTheSelection();
  await cachedLabelsSurviveAWorkerRestart();
  await stopIsOnlyLiveWhenSomethingIsRunning();
  await toolsOpensTheComparisonPanelInThePage();
  await generateReportKeepsThisPagesRun();
  await navEntriesOpenTheirPages();
  await theCacheButtonClearsTheBackendCache();
  await aFailedCacheClearSaysSo();
  await clickingOpensThePanelInThePage();
  await unsupportedPagesStillReachThePicker();
  await anAlreadyOpenHomepageIsReused();

  let failed = 0;
  results.forEach(([name, ok, detail]) => {
    if (!ok) failed += 1;
    console.log(`${ok ? "PASS" : "FAIL"}  ${name}${ok || !detail ? "" : ` — ${detail}`}`);
  });
  console.log(`\n${results.length - failed}/${results.length} checks passed`);
  process.exit(failed === 0 ? 0 : 1);
}

main();
