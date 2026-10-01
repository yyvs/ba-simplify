// Toolbar "Clear cache" button: empties the backend's result cache (what makes a History
// row read "Cached").
//
//   cd extension/tests && npm install && node toolbar-clear-cache.test.js
//
// Regression: on the History page it once called chrome.storage.local.remove on the
// History log, like "Delete all". Hence both assertions: the request goes to the backend,
// and nothing is removed from storage.
//
// Same harness as model-picker.test.js; files load in the extension pages' order.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { JSDOM } = require("jsdom");

const SHARED = path.join(__dirname, "..", "shared");
const SOURCES = [
  "model-labels.js",
  "model-selection.js",
  "demo-pages.js",
  "picker.js",
  "toolbar.js",
].map((f) => fs.readFileSync(path.join(SHARED, f), "utf8"));

let failures = 0;
function check(label, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${ok || !detail ? "" : ` — ${detail}`}`);
  if (!ok) failures += 1;
}

const HEALTH = {
  models_loaded: ["online", "finetuned"],
  model_names: {
    online: "eilamc14/bart-large-text-simplification",
    finetuned: "yunvs/bart-base-wikilarge-simplification",
  },
  granularities: { online: "sentence_by_sentence", finetuned: "sentence_by_sentence" },
  audience_models: [],
  audiences: [{ value: "general_adult", label: "General adult" }],
};

// `clearImpl` stands in for /cache/clear; /health is answered for the picker read-out
// the bar fetches on load. `confirmAnswer` answers the confirmation dialog.
function load({ clearImpl, confirmAnswer = true } = {}) {
  const dom = new JSDOM(`<!doctype html><html><body></body></html>`, {
    pretendToBeVisual: true,
    url: "chrome-extension://test/history.html",
  });
  const { window } = dom;
  const store = { simplifyHistory: [{ sessionId: "session-1", entries: [] }] };
  const calls = [];
  const removals = [];

  const context = {
    window,
    document: window.document,
    location: window.location,
    history: window.history,
    setTimeout,
    clearTimeout,
    console,
    confirm: () => confirmAnswer,
    fetch: (url, options) => {
      if (String(url).endsWith("/health")) {
        return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(HEALTH) });
      }
      calls.push({ url, options });
      return clearImpl(url, options);
    },
    chrome: {
      runtime: {
        getURL: (p) => `chrome-extension://test/${p}`,
        sendMessage: () => Promise.resolve({}),
      },
      storage: {
        local: {
          get: (keys) => {
            const names = Array.isArray(keys) ? keys : [keys];
            const out = {};
            names.forEach((k) => {
              if (k in store) out[k] = store[k];
            });
            return Promise.resolve(out);
          },
          set: (values) => {
            Object.assign(store, values);
            return Promise.resolve();
          },
          remove: (key) => {
            removals.push(key);
            delete store[key];
            return Promise.resolve();
          },
        },
        onChanged: { addListener() {} },
      },
    },
  };
  context.globalThis = context;
  vm.createContext(context);
  SOURCES.forEach((source) => vm.runInContext(source, context));

  const button = window.document.getElementById("ext-toolbar-clear-cache");
  return { context, window, store, calls, removals, button };
}

const tick = (ms = 0) => new Promise((resolve) => setTimeout(resolve, ms));
const ok = (body) => () =>
  Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) });

async function press(setup) {
  const ctx = load(setup);
  await tick();
  ctx.button.click();
  await tick(10);
  return ctx;
}

(async () => {
  // --- where it sits and what it says
  {
    const { window, button } = load({ clearImpl: ok({ cleared: 0 }) });
    check("the bar carries a clear-cache button", button !== null);
    check(
      "...labelled without naming the backend",
      button && button.textContent === "Clear cache",
      button && `label is "${button.textContent}"`
    );
    check(
      "...saying in its tooltip what it empties and what it spares",
      button && /cache/i.test(button.title) && /history/i.test(button.title),
      button && button.title
    );
    check("...marked as destructive, so it renders red", button && button.classList.contains("ext-toolbar-danger"));

    const pickerField = window.document.getElementById("ext-toolbar-picker").closest(".ext-toolbar-field");
    check(
      "...and sits directly left of the model picker",
      button && button.nextElementSibling === pickerField,
      button && `next sibling is ${button.nextElementSibling && button.nextElementSibling.className}`
    );
    check(
      "...as a sibling of the bar itself, not inside the nav list",
      button && button.parentElement && button.parentElement.id === "ext-toolbar",
      button && button.parentElement && button.parentElement.tagName
    );
  }

  // --- the happy path
  {
    const { calls, store, removals, button } = await press({ clearImpl: ok({ cleared: 7 }) });
    check("clearing posts to the backend", calls.length === 1, `${calls.length} requests`);
    check(
      "...at /cache/clear on the backend origin",
      calls[0] && calls[0].url === "http://127.0.0.1:8000/cache/clear",
      calls[0] && String(calls[0].url)
    );
    check(
      "...as a POST",
      calls[0] && calls[0].options && calls[0].options.method === "POST",
      calls[0] && JSON.stringify(calls[0].options)
    );
    // the regression: clearing the cache must not delete the log
    check("...and leaves the history log in storage", Array.isArray(store.simplifyHistory));
    check("...without removing anything from storage", removals.length === 0, removals.join(", "));
    check("...reporting the outcome on the button", button.textContent === "Cache cleared", button.textContent);
    check(
      "...with the count in its tooltip, where a number is readable",
      /7 cached simplifications/.test(button.title),
      button.title
    );
  }

  {
    const { button } = await press({ clearImpl: ok({ cleared: 1 }) });
    check("a single cleared entry is singular", /1 cached simplification\b/.test(button.title), button.title);
  }

  // older backend: 200 without `cleared`
  {
    const { button } = await press({ clearImpl: ok({}) });
    check(
      "a response with no count still confirms the clear",
      button.textContent === "Cache cleared" && !/undefined|NaN/.test(button.title),
      `${button.textContent} / ${button.title}`
    );
  }

  // --- declining
  {
    const { calls, button } = await press({ clearImpl: ok({ cleared: 3 }), confirmAnswer: false });
    check("declining the confirmation sends nothing", calls.length === 0, `${calls.length} requests`);
    check("...and leaves the button as it was", button.textContent === "Clear cache" && !button.disabled);
  }

  // --- failure states
  {
    const { button, store } = await press({
      clearImpl: () => Promise.reject(new Error("Failed to fetch")),
    });
    check("an unreachable backend is reported as not cleared", button.textContent === "Not cleared", button.textContent);
    check("...saying why in the tooltip", /reachable/i.test(button.title), button.title);
    check("...and the log survives the failure", Array.isArray(store.simplifyHistory));
  }

  {
    const { button } = await press({
      clearImpl: () => Promise.resolve({ ok: false, status: 500, json: () => Promise.resolve({}) }),
    });
    check("an HTTP error is reported as not cleared", button.textContent === "Not cleared", button.textContent);
    check("...with the status in the tooltip", /500/.test(button.title), button.title);
  }

  // The outcome label is temporary; reset driven directly instead of waiting out the delay.
  {
    const { context, button } = await press({ clearImpl: ok({ cleared: 2 }) });
    check("the button is held while its outcome is on it", button.disabled === true);
    context.resetClearCacheButton(button, 0);
    await tick(10);
    check("...then goes back to offering the action", button.textContent === "Clear cache", button.textContent);
    check("...and becomes pressable again", button.disabled === false);
    check("...with its resting tooltip back", /Leaves the History log alone/.test(button.title), button.title);
  }

  console.log(
    failures === 0 ? "\nall clear-cache checks passed" : `\n${failures} clear-cache check(s) failed`
  );
  process.exit(failures === 0 ? 0 : 1);
})();
