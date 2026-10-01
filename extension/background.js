// background.js - extension service worker
//
// Shared modules, in dependency order (model-selection.js needs model-labels.js).
importScripts(
  "shared/model-labels.js",
  "shared/model-selection.js",
  "shared/demo-pages.js",
  "shared/active-run.js",
  // saved reports, so comparison runs don't age out of the History log; only the writers
  // (saveReportForPage, saveComparisonRun) are used here
  "shared/report-store.js",
);

const BACKEND_ORIGIN = "http://127.0.0.1:8000";
// Must match backend/run_dev.sh's default DEMO_PORT (serves demo/ over HTTP, so no
// "Allow access to file URLs" needed).
const DEMO_ORIGIN = "http://127.0.0.1:8001";

// Toolbar icon's right-click menu. Not a popup: Chrome only fires action.onClicked
// without a default_popup, and left-click toggles simplification.
//
// Model picking lives in the in-page panel (shared/picker.js), not here: a menu can't
// explain models, show the granularity control or say why an option is unavailable.
//
// Chrome silently ignores action-menu top-level items past the sixth, and separators
// count. So exactly six top-level entries, no separators:
//
//   Home · <status>
//   <model> · <scope>            ▸   selected model, plus the other models
//   History
//   Tools                        ▸   actions
//   Instructions
//   Demo
//
// New entries go inside a submenu.
const MENU_HOME_ID = "simplifier-home";
const MENU_SELECTION_ID = "simplifier-selection";
const MENU_HISTORY_ID = "simplifier-history";
const MENU_TOOLS_ID = "simplifier-tools";
const MENU_INSTRUCTIONS_ID = "simplifier-instructions";
const MENU_DEMO_ID = "simplifier-demo";

// Selection submenu: every selectable model as a disabled line (tick on the active one),
// then the button that opens the panel. Disabled on purpose, so choosing happens only in
// the panel.
const MENU_MODEL_PREFIX = "simplifier-model-";
const MENU_SELECTION_SEPARATOR_ID = "simplifier-selection-separator";
const MENU_PICKER_ID = "simplifier-picker";

// Tools submenu. Stop comes first, disabled when nothing is running.
const MENU_STOP_ID = "simplifier-stop";
const MENU_TOOLS_SEPARATOR_ID = "simplifier-tools-separator";
const MENU_REPORT_ID = "simplifier-report";
const MENU_COMPARE_ID = "simplifier-compare";
const MENU_TOOLS_CACHE_SEPARATOR_ID = "simplifier-tools-cache-separator";
const MENU_CLEAR_CACHE_ID = "simplifier-clear-cache";

// Not "history": that key is openInfoTab's payload for the history tab.
const HISTORY_STORAGE_KEY = "simplifyHistory";
const MAX_HISTORY_PAGES = 20;
// guards against long-open pages with infinite scroll feeding the MutationObserver
const MAX_ENTRIES_PER_PAGE = 200;

// Appends entries to the page's history record (most recent page first), capping pages
// and entries per page to stay within chrome.storage.local's quota.
//
// `sessionMeta` is per-run metadata (content.js's sessionSummary), stored on the page, not
// per entry. Each flush carries a fresh copy that replaces the previous one (later
// observer batches extend the same run).
// `keepAsReport` saves the run as a report so it doesn't age out (comparison arms). Saved
// from the stored page, not the message, so report and log entry can't disagree.
async function recordHistory(
  pageMeta,
  newEntries,
  sessionMeta = null,
  keepAsReport = false,
) {
  if (!pageMeta || !newEntries || newEntries.length === 0) return;
  const result = await chrome.storage.local.get(HISTORY_STORAGE_KEY);
  const pages = result[HISTORY_STORAGE_KEY] || [];
  const existingIndex = pages.findIndex(
    (p) => p.sessionId === pageMeta.sessionId,
  );
  let page;
  if (existingIndex >= 0) {
    [page] = pages.splice(existingIndex, 1);
  } else {
    page = {
      sessionId: pageMeta.sessionId,
      url: pageMeta.url,
      title: pageMeta.title,
      timestamp: pageMeta.timestamp,
      entries: [],
    };
  }
  // merged, not replaced, so `entries` and identity fields survive
  if (sessionMeta) Object.assign(page, sessionMeta);
  page.entries = page.entries.concat(newEntries);
  if (page.entries.length > MAX_ENTRIES_PER_PAGE) {
    page.entries = page.entries.slice(
      page.entries.length - MAX_ENTRIES_PER_PAGE,
    );
  }
  const kept = [page, ...pages];
  const updatedPages = kept.slice(0, MAX_HISTORY_PAGES);
  await chrome.storage.local.set({ [HISTORY_STORAGE_KEY]: updatedPages });
  // Pages that fell out of the log also leave the backend cache, which otherwise only
  // caps at CACHE_MAX (1024 entries, LRU) and could evict still-listed pages instead.
  // Counted per page load (session), not per site.
  await dropCachedPages(kept.slice(MAX_HISTORY_PAGES).map((p) => p.sessionId));

  if (keepAsReport) {
    try {
      await saveReportForPage(page);
    } catch (e) {
      // the log entry is already written
      console.warn("couldn't keep this run as a report:", e);
    }
  }
}

// Asks the backend to drop the given pages' cached simplifications. Best-effort and
// silent: a stopped backend's in-memory cache is already empty. The caller reports.
async function dropCachedPages(sessionIds) {
  const pages = (sessionIds || []).filter(Boolean);
  if (pages.length === 0) return { dropped: 0, pages: 0 };
  try {
    const res = await fetchWithTimeout(
      `${BACKEND_ORIGIN}/cache/pages/delete`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pages }),
      },
      5000,
    );
    if (!res.ok) {
      console.warn(
        `dropping ${pages.length} page(s) from the backend cache: HTTP ${res.status}`,
      );
      return { dropped: 0, pages: 0 };
    }
    return await res.json();
  } catch (e) {
    console.warn("couldn't drop cached pages:", e);
    return { dropped: 0, pages: 0 };
  }
}

// /health.document_max_tokens: per model, null for sentence models. Fallback 512 is the
// seq2seq checkpoints' trained ceiling, the smallest document budget here; returning
// undefined would make the caller chunk against NaN and send whole pages.
function resolveDocumentMaxTokens(model, health) {
  const perModel = health && health.document_max_tokens;
  const budget = perModel && perModel[model];
  return typeof budget === "number" ? budget : 512;
}

// Ready-made selection description for the content script's notice (preflight result
// and "selectionInfo"), which can't read storage or /health itself. Without `health`,
// falls back to the static labels in model-labels.js.
function describeSelection(model, audience, health) {
  const names = health && health.model_names;
  const audienceCapable =
    health && Array.isArray(health.audience_models)
      ? health.audience_models
      : FALLBACK_AUDIENCE_MODELS;
  return {
    model,
    modelLabel: modelDisplayName(model, names),
    // raw id/path/tag for the notice's "Model:" line; modelLabel only if /health has none
    modelId: (names && names[model]) || null,
    // canonical values, from /health or the fallback maps
    method: modelMethod(model, health && health.methods),
    granularity: modelGranularity(model, health && health.granularities),
    methodLabel: modelMethodLabel(
      model,
      health && health.methods,
      health && health.granularities,
    ),
    audience,
    audienceLabel: audienceDisplayName(audience, health && health.audiences),
    // sent for every model, but only some act on it
    audienceApplies: audienceCapable.includes(model),
  };
}

// Delay before opening an info tab (error/instructions), so the triggering notice or
// browser UI can be read first. Must stay well under MV3's ~30 s idle kill (else use
// chrome.alarms). Keep in sync with content.js's TROUBLESHOOT_OPEN_DELAY_MS (its notice
// counts this down).
const INFO_TAB_OPEN_DELAY_MS = 3000;
// page name -> pending timeout, so a repeat call reschedules instead of opening twice
const pendingInfoTabOpens = new Map();

// Stores `info` under `page`'s key and opens/refocuses <page>.html (error, instructions,
// history). Storage instead of query params keeps URLs short and lets an open tab pick up
// new info via storage.onChanged. Extension pages are flat files at the root.
async function openInfoTab(page, info) {
  await chrome.storage.local.set({ [page]: info });
  await openExtensionTab(`${page}.html`);
}

// Opens an extension page, refocusing an existing tab instead of opening a copy.
// `hash` selects a report (generateReportForTab, notifyComparisonFinished) and is applied
// to an existing tab too, so it doesn't keep showing the previous report. Tab matching
// ignores the fragment; a fragment-only change fires `hashchange` (handled in
// shared/reports.js) instead of reloading.
async function openExtensionTab(file, hash) {
  const base = chrome.runtime.getURL(file);
  const url = hash ? `${base}#${hash}` : base;
  const existing = await chrome.tabs.query({ url: base });
  if (existing.length > 0) {
    await chrome.tabs.update(existing[0].id, {
      active: true,
      ...(hash ? { url } : {}),
    });
    await chrome.windows.update(existing[0].windowId, { focused: true });
  } else {
    await chrome.tabs.create({ url });
  }
}

// Immediate feedback where no content script can show a notice (unsupported page, tab
// predating the extension (re)load), before the delayed info tab.
function notifyImmediately(title, message) {
  chrome.notifications.create({
    type: "basic",
    iconUrl: chrome.runtime.getURL("icons/icon128.png"),
    title,
    message,
  });
}

function openInfoTabDelayed(page, info) {
  if (pendingInfoTabOpens.has(page)) {
    clearTimeout(pendingInfoTabOpens.get(page));
  }
  const timer = setTimeout(() => {
    pendingInfoTabOpens.delete(page);
    openInfoTab(page, info);
  }, INFO_TAB_OPEN_DELAY_MS);
  pendingInfoTabOpens.set(page, timer);
}

chrome.runtime.onInstalled.addListener((details) => {
  // chrome_update / shared_module_update are unrelated to this extension
  if (details.reason === "install" || details.reason === "update") {
    openInfoTabDelayed("instructions", { reason: details.reason });
  }

  // removeAll first: onInstalled also fires on update, and re-creating an id throws
  chrome.contextMenus.removeAll(() => {
    // Exactly six top-level entries, no separators (see header). Computed titles start as
    // placeholders; refreshMenu() fills them once storage and the backend are read.
    chrome.contextMenus.create({
      id: MENU_HOME_ID,
      title: "Home",
      contexts: ["action"],
    });

    chrome.contextMenus.create({
      id: MENU_SELECTION_ID,
      title: "Reading current model…",
      contexts: ["action"],
    });
    // One disabled child per key, hidden until loaded. Created once and updated in place:
    // rebuilding a submenu isn't atomic, so it flickers and races with the menu opening.
    MODEL_KEYS.forEach((key) => {
      chrome.contextMenus.create({
        id: `${MENU_MODEL_PREFIX}${key}`,
        parentId: MENU_SELECTION_ID,
        title: key,
        enabled: false,
        contexts: ["action"],
      });
    });
    chrome.contextMenus.create({
      id: MENU_SELECTION_SEPARATOR_ID,
      parentId: MENU_SELECTION_ID,
      type: "separator",
      contexts: ["action"],
    });
    // "method", not "model": the panel sets model, granularity and audience together
    chrome.contextMenus.create({
      id: MENU_PICKER_ID,
      parentId: MENU_SELECTION_ID,
      title: "Change simplification method",
      contexts: ["action"],
    });

    chrome.contextMenus.create({
      id: MENU_HISTORY_ID,
      title: "History",
      contexts: ["action"],
    });

    chrome.contextMenus.create({
      id: MENU_TOOLS_ID,
      title: "Tools",
      contexts: ["action"],
    });
    chrome.contextMenus.create({
      id: MENU_STOP_ID,
      parentId: MENU_TOOLS_ID,
      title: "Stop current process",
      enabled: false,
      contexts: ["action"],
    });
    chrome.contextMenus.create({
      id: MENU_TOOLS_SEPARATOR_ID,
      parentId: MENU_TOOLS_ID,
      type: "separator",
      contexts: ["action"],
    });
    chrome.contextMenus.create({
      id: MENU_REPORT_ID,
      parentId: MENU_TOOLS_ID,
      title: "Generate report (Evaluate)",
      contexts: ["action"],
    });
    chrome.contextMenus.create({
      id: MENU_COMPARE_ID,
      parentId: MENU_TOOLS_ID,
      title: "Compare multiple models",
      contexts: ["action"],
    });
    chrome.contextMenus.create({
      id: MENU_TOOLS_CACHE_SEPARATOR_ID,
      parentId: MENU_TOOLS_ID,
      type: "separator",
      contexts: ["action"],
    });
    // "simplifications" so it isn't read as the browser cache. Menu items have no tooltip
    // (unlike shared/toolbar.js's button), so the notification says what isn't cleared.
    chrome.contextMenus.create({
      id: MENU_CLEAR_CACHE_ID,
      parentId: MENU_TOOLS_ID,
      title: "Clear cached simplifications",
      contexts: ["action"],
    });

    chrome.contextMenus.create({
      id: MENU_INSTRUCTIONS_ID,
      title: "Instructions",
      contexts: ["action"],
    });
    chrome.contextMenus.create({
      id: MENU_DEMO_ID,
      title: "Demo",
      contexts: ["action"],
    });

    refreshMenu();
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === MENU_PICKER_ID) {
    openModelPicker(tab);
  } else if (info.menuItemId === MENU_HOME_ID) {
    // no payload, so no stray "home" storage key
    openExtensionTab("home.html");
  } else if (info.menuItemId === MENU_INSTRUCTIONS_ID) {
    // no {reason}: opened on purpose, not by install/update
    openInfoTab("instructions", {});
  } else if (info.menuItemId === MENU_HISTORY_ID) {
    openInfoTab("history", {});
  } else if (info.menuItemId === MENU_DEMO_ID) {
    // the overview page, which links the other fixtures
    openDemoPage(
      DEMO_PAGES.find((page) => page.file === "index.html") || DEMO_PAGES[0],
    );
  } else if (info.menuItemId === MENU_STOP_ID) {
    stopEverythingRunning();
  } else if (info.menuItemId === MENU_REPORT_ID) {
    generateReportForTab(tab);
  } else if (info.menuItemId === MENU_COMPARE_ID) {
    openComparePicker(tab);
  } else if (info.menuItemId === MENU_CLEAR_CACHE_ID) {
    clearBackendCache();
  }
});

// Tools ▸ Stop. The run can end between the menu being drawn and the click, hence the
// "nothing to stop" case.
async function stopEverythingRunning() {
  const result = await stopActiveRun();
  if (!result.stopped) {
    notifyImmediately(
      "Nothing to stop",
      "No page is being simplified and no comparison is running.",
    );
    return;
  }
  notifyImmediately(
    "Stopped",
    `${activeRunSummary(result.run)} — stopped. Anything already simplified is left as it is.`,
  );
}

// Tools ▸ Generate report (Evaluate). Saves this page's most recent run as a report and
// opens the Reports page, where its evaluation is computed.
//
// Looked up by URL, not the content script's session id (which is the current run, not the
// finished one). The log is newest-first, so the newest match wins.
async function generateReportForTab(tab) {
  const url = (tab && tab.url) || "";
  const stored = await chrome.storage.local.get(HISTORY_STORAGE_KEY);
  const pages = stored[HISTORY_STORAGE_KEY] || [];
  const page = pages.find((p) => p.url === url);
  if (!page) {
    // open the existing reports anyway
    notifyImmediately(
      "Nothing to report on this page",
      "This page hasn't been simplified yet, so there is no run to keep. Opening the reports you have.",
    );
    openExtensionTab("reports.html");
    return;
  }
  try {
    const { report, already } = await saveReportForPage(page);
    notifyImmediately(
      already ? "Report updated" : "Report saved",
      `${page.title || page.url} — kept out of the rolling History log. Its evaluation is on the Reports page.`,
    );
    // open at this report, not the top of the list
    await openExtensionTab("reports.html", `report-${report.reportId}`);
    return;
  } catch (e) {
    console.warn("couldn't save the report:", e);
    notifyImmediately(
      "Couldn't save the report",
      "Opening the reports you already have.",
    );
  }
  openExtensionTab("reports.html");
}

// Tools ▸ Compare multiple models. The panel is in-page DOM (shared/compare-picker.js)
// and needs page text, so unlike the model picker there is no extension-page fallback.
function openComparePicker(tab) {
  const tabId = tab && tab.id;
  const url = (tab && tab.url) || "";
  if (
    tabId == null ||
    !(url.startsWith("http://") || url.startsWith("https://"))
  ) {
    notifyImmediately(
      "Can't compare models here",
      "A comparison runs over a page's own text, and browser pages like chrome:// can't be read. Open a web page and try again.",
    );
    return;
  }
  chrome.tabs.sendMessage(tabId, { cmd: "showCompareModels" }, () => {
    if (chrome.runtime.lastError) {
      // reading lastError marks it handled (else Chrome logs an uncaught error)
      console.warn(
        "no content script to open the comparison panel in:",
        chrome.runtime.lastError.message,
      );
      notifyImmediately(
        "Refresh this tab",
        "This tab needs to be refreshed before the extension can read it.",
      );
    }
  });
}

// Empties the backend's result cache (what makes repeat pages come back "Cached"). The
// History log is untouched. One-way on purpose: deleting a log entry drops its cached
// simplifications, but clearing the cache deletes no log entries.
//
// No confirmation (unlike shared/toolbar.js): a stray press only means re-running the
// model, same as a backend restart. Feedback via notification only; failure opens no
// Error tab.
async function clearBackendCache() {
  try {
    const res = await fetchWithTimeout(
      `${BACKEND_ORIGIN}/cache/clear`,
      { method: "POST" },
      5000,
    );
    if (!res.ok) {
      notifyImmediately(
        "Couldn't clear the cache",
        `The backend answered HTTP ${res.status}.`,
      );
      return;
    }
    const data = await res.json();
    const cleared =
      data && typeof data.cleared === "number" ? data.cleared : null;
    const consequence =
      "Text simplified before will be sent to the model again. The History log is untouched.";
    notifyImmediately(
      "Backend cache cleared",
      cleared === null
        ? consequence
        : `${cleared} ${cleared === 1 ? "entry" : "entries"} dropped. ${consequence}`,
    );
  } catch (e) {
    console.warn("cache clear failed:", e);
    notifyImmediately(
      "Couldn't clear the cache",
      "The backend isn't reachable — start backend/run_dev.sh, then try again.",
    );
  }
}

// Native notification because a comparison takes minutes and changes nothing on screen,
// so the user is likely in another tab.
function notifyComparisonFinished(message) {
  const models = message.models || 0;
  const modelWord = `${models} ${models === 1 ? "model" : "models"}`;
  if (message.nothingToDo) {
    notifyImmediately(
      "Nothing to compare",
      `No text on ${message.page} matched the units you chose.`,
    );
    return;
  }
  if (message.stopped) {
    notifyImmediately(
      "Comparison stopped",
      `Stopped part-way through ${modelWord} on ${message.page}. What each of them answered before that is in the report.`,
    );
  } else {
    notifyImmediately(
      "Comparison finished",
      `${modelWord} ran over ${message.units} ${message.units === 1 ? "entry" : "entries"} of ${message.page}. ` +
        `Their answers are combined in one report.`,
    );
  }
  // The page is left unchanged, so open the report -- for stopped comparisons too, since
  // partial answers are in it.
  if (message.group)
    openExtensionTab("reports.html", `comparison-${message.group}`);
}

// The picker panel is in-page DOM (shared/picker.js), opened by the content script. Where
// there is none (chrome://, about:, file://, the extension's own pages, a tab predating
// the extension (re)load, no active tab), fall back to the homepage with the hash its
// toolbar treats as "open the panel", so the panel always appears.
function openModelPicker(tab) {
  const tabId = tab && tab.id;
  const url = (tab && tab.url) || "";
  if (
    tabId == null ||
    !(url.startsWith("http://") || url.startsWith("https://"))
  ) {
    openPickerOnHomepage();
    return;
  }
  chrome.tabs.sendMessage(tabId, { cmd: "showModelPicker" }, () => {
    if (chrome.runtime.lastError) {
      // reading lastError marks it handled (else Chrome logs an uncaught error)
      console.warn(
        "no content script to open the picker in:",
        chrome.runtime.lastError.message,
      );
      openPickerOnHomepage();
    }
  });
}

// Must match TOOLBAR_PICKER_HASH in shared/toolbar.js.
const PICKER_FALLBACK_HASH = "#picker";

// Immediate, unlike openInfoTabDelayed: no notice to read first. An open homepage gets the
// hash via a same-document navigation (toolbar listens for hashchange; no load event fires).
async function openPickerOnHomepage() {
  const base = chrome.runtime.getURL("home.html");
  const existing = await chrome.tabs.query({ url: base });
  if (existing.length > 0) {
    await chrome.tabs.update(existing[0].id, {
      active: true,
      url: `${base}${PICKER_FALLBACK_HASH}`,
    });
    await chrome.windows.update(existing[0].windowId, { focused: true });
    return;
  }
  await chrome.tabs.create({ url: `${base}${PICKER_FALLBACK_HASH}` });
}

// Checks the demo server first; otherwise a stopped server gives a bare
// ERR_CONNECTION_REFUSED with no hint to start backend/run_dev.sh.
async function openDemoPage(demoPage) {
  const url = `${DEMO_ORIGIN}/${demoPage.file}`;
  try {
    // no-cors: python's http.server sends no CORS headers, but an opaque response still
    // rejects on connection failure. no-store: a cached earlier success would otherwise
    // pass the check after the server stopped.
    await fetchWithTimeout(
      `${DEMO_ORIGIN}/`,
      { mode: "no-cors", cache: "no-store" },
      3000,
    );
    chrome.tabs.create({ url });
  } catch (e) {
    console.warn("demo server unreachable:", e);
    notifyImmediately(
      "Demo server not running",
      "Start backend/run_dev.sh (it serves demo/ too), then try again.",
    );
    openInfoTabDelayed("error", {
      stage: "demo_unreachable",
      error: `demo server not reachable at ${DEMO_ORIGIN} [${e}]`,
    });
  }
}

function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...options, signal: controller.signal }).finally(() =>
    clearTimeout(timer),
  );
}

// Latest /health answer, so the menu summary can name the served tag ("Ollama Qwen2.5 ·
// 7B"). Only filled by checks made for other reasons (toggle preflight, picker opening);
// never polled.
let lastKnownHealth = null;

// Label fields persisted because the MV3 worker (and lastKnownHealth) dies after ~30 s
// idle, usually before the menu is opened. models_loaded is deliberately not stored: it
// goes stale.
const HEALTH_LABELS_STORAGE_KEY = "simplifierHealthLabels";

function rememberHealth(health) {
  lastKnownHealth = health;
  // any answer means reachable (menu's Home line)
  rememberBackendStatus(!!health);
  if (!health) return;
  chrome.storage.local.set({
    [HEALTH_LABELS_STORAGE_KEY]: {
      model_names: health.model_names || null,
      methods: health.methods || null,
      audience_models: health.audience_models || null,
      audiences: health.audiences || null,
    },
  });
}

async function healthForLabels() {
  if (lastKnownHealth) return lastKnownHealth;
  const stored = await chrome.storage.local.get(HEALTH_LABELS_STORAGE_KEY);
  return stored[HEALTH_LABELS_STORAGE_KEY] || null;
}

// Last observed backend reachability for the Home entry; in storage for the same reason
// as the labels, rather than polling on wake.
const BACKEND_STATUS_STORAGE_KEY = "simplifierBackendStatus";

function rememberBackendStatus(ok) {
  chrome.storage.local.set({
    [BACKEND_STATUS_STORAGE_KEY]: { ok: !!ok, at: Date.now() },
  });
}

// --- menu refresh ---
// Fills in the placeholder titles. Called on install, on selection or run-lock changes,
// and after each health answer (which replaces "Open LLM" with the served tag).

// "Home · Simplifying Recycling — Wikipedia" / "Home · Backend ready". An active run takes
// precedence over backend state; with neither known, just "Home".
function homeMenuTitle(activeRun, backendStatus) {
  if (activeRun) return `Home · ${activeRunSummary(activeRun)}`;
  if (backendStatus)
    return `Home · ${backendStatus.ok ? "Backend ready" : "Backend not reachable"}`;
  return "Home";
}

function selectionMenuTitle(selection, health) {
  return selectionSummary(
    selection.family,
    selection.granularity,
    selection.audience,
    health,
  );
}

function updateMenuItem(id, props) {
  chrome.contextMenus.update(id, props, () => {
    // item may not exist yet if the worker woke before onInstalled ran; reading lastError
    // marks it handled
    if (chrome.runtime.lastError) return;
  });
}

function refreshMenu() {
  Promise.all([
    readSelection(),
    healthForLabels(),
    readActiveRun(),
    chrome.storage.local.get(BACKEND_STATUS_STORAGE_KEY),
  ]).then(([selection, health, activeRun, stored]) => {
    updateMenuItem(MENU_HOME_ID, {
      title: homeMenuTitle(activeRun, stored[BACKEND_STATUS_STORAGE_KEY]),
    });
    updateMenuItem(MENU_SELECTION_ID, {
      title: selectionMenuTitle(selection, health),
    });

    // Visibility needs live health (cached labels omit models_loaded); without it, show
    // every key rather than a possibly wrong subset.
    const loaded =
      lastKnownHealth && Array.isArray(lastKnownHealth.models_loaded)
        ? lastKnownHealth.models_loaded
        : null;
    const names = health && health.model_names;
    const granularities =
      (lastKnownHealth && lastKnownHealth.granularities) || null;
    MODEL_KEYS.forEach((key) => {
      const active = key === selection.model;
      updateMenuItem(`${MENU_MODEL_PREFIX}${key}`, {
        // tick instead of a checkbox item: a disabled checkbox looks broken
        title: `${modelDisplayNameWithScope(key, names, granularities)}${active ? "  ✓" : ""}`,
        visible: loaded ? loaded.includes(key) : true,
      });
    });

    updateMenuItem(MENU_STOP_ID, {
      // names the run it would stop; "Stop: <summary>" keeps the page title's case
      title: activeRun
        ? `Stop: ${activeRunSummary(activeRun)}`
        : "Stop current process",
      enabled: !!activeRun,
    });
  });
}

// Selection changes (panel, toolbar, Error page) are written straight to storage, not
// messaged here, so the menu watches storage.
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local") return;
  // not HEALTH_LABELS_STORAGE_KEY: rememberHealth() writes it alongside a refresh, which
  // would double every refresh
  const watched = [
    MODEL_STORAGE_KEY,
    GRANULARITY_STORAGE_KEY,
    AUDIENCE_STORAGE_KEY,
    ACTIVE_RUN_STORAGE_KEY,
    BACKEND_STATUS_STORAGE_KEY,
  ];
  if (watched.some((key) => key in changes)) refreshMenu();
});

// Preflight on each toggle-on, in stages so the content script can report the specific
// cause: (1) backend reachable, (2) model loaded, (3) test request returns output.
async function checkBackendHealth() {
  let health;
  try {
    const res = await fetchWithTimeout(`${BACKEND_ORIGIN}/health`, {}, 5000);
    if (!res.ok) {
      rememberBackendStatus(false);
      return {
        ok: false,
        stage: "reachable",
        error: `backend responded with HTTP ${res.status}`,
      };
    }
    health = await res.json();
  } catch (e) {
    rememberBackendStatus(false);
    return {
      ok: false,
      stage: "reachable",
      error: `backend not reachable at ${BACKEND_ORIGIN} [${e}]`,
    };
  }

  if (!health || !health.model_loaded) {
    return {
      ok: false,
      stage: "model_loaded",
      error: "backend is up but the model has not finished loading yet",
    };
  }

  rememberHealth(health);
  refreshMenu();

  const selectedModel = await getSelectedModel();
  if (
    Array.isArray(health.models_loaded) &&
    !health.models_loaded.includes(selectedModel)
  ) {
    // unlike "model_loaded", waiting won't fix this: at least one model is ready, but not
    // the selected one (e.g. SIMPLIFIER_MODEL_FINETUNED unset)
    return {
      ok: false,
      stage: "model_unavailable",
      error: `selected model '${selectedModel}' isn't loaded on the backend (loaded: ${health.models_loaded.join(", ") || "none"})`,
    };
  }

  const selectedAudience = await getSelectedAudience();
  try {
    const res = await fetchWithTimeout(
      `${BACKEND_ORIGIN}/simplify`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: "This is a short connectivity test sentence.",
          model: selectedModel,
          audience: selectedAudience,
        }),
      },
      // prompted LLMs on CPU/Metal are far slower than BART, and this also absorbs
      // first-load cost
      selectedModel.startsWith("llm_") ? 90000 : 20000,
    );
    if (!res.ok) {
      return {
        ok: false,
        stage: "model_response",
        error: `test request failed with HTTP ${res.status}`,
      };
    }
    const data = await res.json();
    if (
      !data ||
      typeof data.simplified !== "string" ||
      !data.simplified.trim()
    ) {
      return {
        ok: false,
        stage: "model_response",
        error: "test request returned no usable output",
      };
    }
  } catch (e) {
    return {
      ok: false,
      stage: "model_response",
      error: `test request failed: ${e}`,
    };
  }

  // The content script needs the granularity before collecting page text (the two walk
  // the DOM differently); describeSelection() also feeds the progress notice's header.
  return {
    ok: true,
    ...describeSelection(selectedModel, selectedAudience, health),
    // per model: seq2seq trained max_length (512) vs. a much larger LLM context window
    documentMaxTokens: resolveDocumentMaxTokens(selectedModel, health),
  };
}

// --- one run at a time ---
// Lock shape and rationale: shared/active-run.js. This worker owns all writes, since
// competing pages can't see each other.
//
// Every read-modify-write goes through this queue; otherwise two simultaneous claims could
// both read "free" and both win.
let activeRunQueue = Promise.resolve();

function withActiveRunLock(fn) {
  const next = activeRunQueue.then(fn, fn);
  // swallow rejections so one failure doesn't wedge later claims
  activeRunQueue = next.then(
    () => undefined,
    () => undefined,
  );
  return next;
}

function tabStillExists(tabId) {
  if (tabId == null) return Promise.resolve(false);
  return chrome.tabs.get(tabId).then(
    () => true,
    () => false,
  );
}

// The record, or null if nobody is running. Clears records that can never be released:
// stale (tab reloaded mid-run, content script threw) or tab gone (closed, browser
// restarted).
async function readActiveRun() {
  const stored = await chrome.storage.local.get(ACTIVE_RUN_STORAGE_KEY);
  const run = stored[ACTIVE_RUN_STORAGE_KEY] || null;
  if (!run) return null;
  if (activeRunIsStale(run)) {
    await chrome.storage.local.remove(ACTIVE_RUN_STORAGE_KEY);
    return null;
  }
  if (!(await tabStillExists(run.tabId))) {
    await chrome.storage.local.remove(ACTIVE_RUN_STORAGE_KEY);
    return null;
  }
  return run;
}

// Takes the lock for `tabId`, or returns the holder. Re-claiming one's own lock replaces
// the record (toggle off/on is the same tab's next run).
function claimActiveRun(tabId, run) {
  return withActiveRunLock(async () => {
    if (tabId == null) return { ok: false, holder: null };
    const holder = await readActiveRun();
    if (holder && holder.tabId !== tabId) return { ok: false, holder };
    const now = Date.now();
    const record = {
      ...run,
      // set here, not by the claimant, so it can be trusted
      tabId,
      startedAt: now,
      updatedAt: now,
    };
    await chrome.storage.local.set({ [ACTIVE_RUN_STORAGE_KEY]: record });
    return { ok: true, holder: record };
  });
}

function releaseActiveRun(tabId) {
  return withActiveRunLock(async () => {
    const stored = await chrome.storage.local.get(ACTIVE_RUN_STORAGE_KEY);
    const run = stored[ACTIVE_RUN_STORAGE_KEY] || null;
    // Not readActiveRun(): a closing tab would see its own record as stale / tab gone
    // and skip clearing it. Only ownership matters here.
    if (!run) return { released: false };
    if (tabId != null && run.tabId !== tabId) return { released: false };
    await chrome.storage.local.remove(ACTIVE_RUN_STORAGE_KEY);
    return { released: true };
  });
}

// Holder's progress counts, for other tabs' "already running" notice. Also the heartbeat:
// always writes `updatedAt`.
function updateActiveRunProgress(tabId, patch) {
  return withActiveRunLock(async () => {
    const stored = await chrome.storage.local.get(ACTIVE_RUN_STORAGE_KEY);
    const run = stored[ACTIVE_RUN_STORAGE_KEY] || null;
    if (!run || run.tabId !== tabId) return { updated: false };
    await chrome.storage.local.set({
      [ACTIVE_RUN_STORAGE_KEY]: {
        ...run,
        ...patch,
        tabId: run.tabId,
        updatedAt: Date.now(),
      },
    });
    return { updated: true };
  });
}

// Stops the active run from elsewhere (blocked page's "Stop that page", menu Stop). The
// lock is released even if the tab doesn't answer, else it blocks until stale.
async function stopActiveRun() {
  const run = await readActiveRun();
  if (!run) return { stopped: false, run: null };
  const stopped = await new Promise((resolve) => {
    chrome.tabs.sendMessage(run.tabId, { cmd: "stopRun" }, (resp) => {
      if (chrome.runtime.lastError) {
        // reading lastError marks it handled; the tab isn't listening
        console.warn(
          "couldn't reach the running tab to stop it:",
          chrome.runtime.lastError.message,
        );
        resolve(false);
        return;
      }
      resolve(!!(resp && resp.stopped));
    });
  });
  await releaseActiveRun(run.tabId);
  return { stopped, run };
}

// Closing the tab is a common way to abandon a run; release now instead of waiting for
// staleness.
chrome.tabs.onRemoved.addListener((tabId) => {
  releaseActiveRun(tabId);
});

chrome.action.onClicked.addListener((tab) => {
  if (tab.id == null) return;

  // chrome://, about:, file:// and the new-tab page can't run content scripts; show the
  // error tab instead of doing nothing
  const url = tab.url || "";
  if (!url.startsWith("http://") && !url.startsWith("https://")) {
    console.warn("extension clicked on unsupported page", url);
    notifyImmediately(
      "Can't simplify this page",
      "Browser pages like chrome:// or file:// can't run extensions. Opening troubleshooting info...",
    );
    openInfoTabDelayed("error", {
      stage: "unsupported_page",
      error: `cannot run on this page: ${url || "(unknown URL)"}`,
    });
    return;
  }

  chrome.tabs.sendMessage(tab.id, { cmd: "toggle" }, (resp) => {
    if (chrome.runtime.lastError) {
      // no content script, almost always because the tab predates the extension
      // install/reload
      console.warn("sendMessage failed:", chrome.runtime.lastError.message);
      notifyImmediately(
        "Refresh this tab",
        "This tab needs to be refreshed before the extension can run on it. Opening troubleshooting info...",
      );
      openInfoTabDelayed("error", {
        stage: "content_script_unreachable",
        error: `no content script listening in tab: ${chrome.runtime.lastError.message}`,
      });
    }
  });
});

// messages from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.cmd === "updateBadge") {
    const tabId = sender.tab && sender.tab.id;
    if (tabId == null) return;
    const text = message.isSimplified ? "ON" : "";
    chrome.action.setBadgeText({ text, tabId });
    chrome.action.setBadgeBackgroundColor({
      color: message.isSimplified ? "#4688F1" : "#000000",
      tabId,
    });
    return;
  }

  // Proxied: Chrome's Local Network Access gate blocks content-script fetches to
  // 127.0.0.1, but not the service worker.
  if (message.cmd === "fetchSimplify") {
    Promise.all([getSelectedModel(), getSelectedAudience()]).then(
      ([model, audience]) => {
        fetch(`${BACKEND_ORIGIN}/simplify`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          // `page`: the History log's session id, so the backend can drop the page's
          // cached units when it leaves the log. Not part of the cache key.
          body: JSON.stringify({
            text: message.text,
            model,
            audience,
            page: message.page || null,
          }),
        })
          .then((r) => r.json())
          // `audience` is the requested one; data.audience is what the backend used
          // (normalized for models that ignore it) and what the history log records
          .then((data) => sendResponse({ ok: true, data, model, audience }))
          .catch((e) => sendResponse({ ok: false, error: String(e) }));
      },
    );
    return true; // keep the message channel open for the async sendResponse
  }

  if (message.cmd === "fetchSimplifyWithModel") {
    Promise.all([
      Promise.resolve(message.model || "online"),
      Promise.resolve(message.audience || DEFAULT_AUDIENCE),
    ]).then(([model, audience]) => {
      fetch(`${BACKEND_ORIGIN}/simplify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // comparison arms pass their own session, so their cache entries drop with them
        body: JSON.stringify({
          text: message.text,
          model,
          audience,
          page: message.page || null,
        }),
      })
        .then((r) => r.json())
        .then((data) => sendResponse({ ok: true, data, model, audience }))
        .catch((e) => sendResponse({ ok: false, error: String(e) }));
    });
    return true;
  }

  if (message.cmd === "healthCheck") {
    checkBackendHealth().then(sendResponse);
    return true; // keep the message channel open for the async sendResponse
  }

  // Raw /health for the in-page picker. Not checkBackendHealth(): its test generation
  // can take up to a minute on the prompted-LLM path. Proxied for Local Network Access.
  if (message.cmd === "health") {
    fetchWithTimeout(`${BACKEND_ORIGIN}/health`, {}, 5000)
      .then((res) =>
        res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`)),
      )
      .then((health) => {
        rememberHealth(health);
        refreshMenu();
        sendResponse({ ok: true, health });
      })
      .catch((e) => {
        rememberBackendStatus(false);
        sendResponse({ ok: false, error: String(e) });
      });
    return true; // keep the message channel open for the async sendResponse
  }

  // Storage-only selection description, so the notice can name the model immediately
  // instead of after the preflight (up to a minute for a prompted LLM).
  if (message.cmd === "selectionInfo") {
    Promise.all([getSelectedModel(), getSelectedAudience()]).then(
      ([model, audience]) => {
        sendResponse(describeSelection(model, audience, null));
      },
    );
    return true; // keep the message channel open for the async sendResponse
  }

  if (message.cmd === "openErrorPage") {
    openInfoTabDelayed("error", { stage: message.stage, error: message.error });
    return;
  }

  if (message.cmd === "recordHistory") {
    recordHistory(
      message.page,
      message.entries,
      message.session || null,
      !!message.report,
    );
    return;
  }

  // One comparison arm, merged into the comparison's report (shared/report-store.js's
  // saveComparisonRun). recordHistory separately logs it as its own page.
  if (message.cmd === "recordComparison") {
    saveComparisonRun(message.run).catch((e) => {
      // the arm's History row is written either way
      console.warn("couldn't fold this arm into its comparison report:", e);
    });
    return;
  }

  if (message.cmd === "comparisonFinished") {
    notifyComparisonFinished(message);
    return;
  }

  // --- one-run-at-a-time lock ---
  // claim before starting, report progress, release when done. Tab id comes from the
  // sender, not the message, so claims can't be faked.
  if (message.cmd === "claimRun") {
    claimActiveRun(sender.tab && sender.tab.id, message.run || {}).then(
      sendResponse,
    );
    return true; // keep the message channel open for the async sendResponse
  }

  if (message.cmd === "releaseRun") {
    releaseActiveRun(sender.tab && sender.tab.id).then(sendResponse);
    return true;
  }

  if (message.cmd === "runProgress") {
    updateActiveRunProgress(
      sender.tab && sender.tab.id,
      message.progress || {},
    ).then(sendResponse);
    return true;
  }

  // Read-only lock query. No in-tree caller (the blocked notice uses the rejected
  // claimRun, then watches storage); kept for API completeness and
  // tests/one-run-at-a-time.test.js.
  if (message.cmd === "activeRun") {
    readActiveRun().then((run) => sendResponse({ run }));
    return true;
  }

  // "Stop that page" on a blocked page, and the toolbar menu's Stop entry.
  if (message.cmd === "stopActiveRun") {
    stopActiveRun().then(sendResponse);
    return true;
  }

  // From the History page on delete; routed here so the backend origin and endpoint live
  // in one place.
  if (message.cmd === "dropCachedPages") {
    dropCachedPages(message.pages).then(sendResponse);
    return true; // keep the message channel open for the async sendResponse
  }

  // from shared/toolbar.js's demo button and the homepage's fixture buttons
  if (message.cmd === "openDemoPage") {
    const demoPage = DEMO_PAGES.find((p) => p.file === message.file);
    if (demoPage) openDemoPage(demoPage);
    return;
  }
});
