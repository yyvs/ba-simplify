// Model picker: the (family, granularity, audience) -> backend key rules in
// shared/model-selection.js, and the panel shared/picker.js renders from them.
//
//   cd extension/tests && npm install && node model-picker.test.js
//
// model-labels.js, model-selection.js and picker.js are non-module content scripts
// (globals, expect `chrome` and a document), so they run in one vm context with a jsdom
// window and a fake chrome.storage.
//
// Covers: four models offered over seven keys, granularity combinations that are not all
// populated (no document-scope comparison checkpoint), a recoverable fallback, and
// per-model audience memory.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { JSDOM } = require("jsdom");

const SHARED = path.join(__dirname, "..", "shared");
const SOURCES = ["model-labels.js", "model-selection.js", "picker.js"].map((f) =>
  fs.readFileSync(path.join(SHARED, f), "utf8")
);

// Everything registered except the two document-scope prompted keys (on by default in
// backend/main.py's MODEL_ENV_CONFIG). Degraded setup: env vars emptied, or Ollama serving
// only one tag. Whole sections is then unavailable for reasons that differ per family.
const HEALTH = {
  models_loaded: ["online", "finetuned", "llm_7b", "llm_3b", "document"],
  model_names: {
    online: "eilamc14/bart-large-text-simplification",
    finetuned: "yunvs/bart-base-wikilarge-simplification",
    llm_7b: "qwen2.5:7b-instruct-q4_K_M",
    llm_3b: "qwen2.5:3b-instruct-q4_K_M",
    document: "yunvs/bart-base-dwikipedia-simplification",
  },
  granularities: {
    online: "sentence_by_sentence",
    finetuned: "sentence_by_sentence",
    llm_7b: "sentence_by_sentence",
    llm_3b: "sentence_by_sentence",
    document: "whole_sections",
  },
  audience_models: ["llm_7b", "llm_3b", "llm_doc_7b", "llm_doc_3b"],
  audiences: [
    { value: "non_native_speakers", label: "Non-native speakers" },
    { value: "children", label: "Children · 8–12" },
    { value: "low_literacy", label: "Low literacy" },
    { value: "cognitive_disability", label: "Cognitive disabilities" },
    { value: "general_adult", label: "General adult" },
  ],
};

function load({ stored = {}, health = HEALTH } = {}) {
  const dom = new JSDOM(`<!doctype html><html><body><p>Some page text.</p></body></html>`, {
    pretendToBeVisual: true,
  });
  const { window } = dom;
  const store = { ...stored };
  const storageListeners = [];
  const chrome = {
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
          // Chrome fires storage.onChanged in every context, the writer included; the
          // toolbar read-out relies on this.
          const changes = {};
          Object.keys(values).forEach((k) => (changes[k] = { newValue: values[k] }));
          storageListeners.forEach((fn) => fn(changes, "local"));
          return Promise.resolve();
        },
      },
      onChanged: { addListener: (fn) => storageListeners.push(fn) },
    },
    runtime: {
      lastError: null,
      // for content.js's top-level listener
      onMessage: { addListener() {} },
      // /health via the service worker; `health: null` = backend not running. Callers
      // without a callback (content.js's badge update) are allowed.
      sendMessage(_msg, cb) {
        if (typeof cb !== "function") return;
        cb(health ? { ok: true, health } : { ok: false, error: "not reachable" });
      },
    },
  };

  const context = {
    window,
    document: window.document,
    Node: window.Node,
    chrome,
    console,
    store,
  };
  context.globalThis = context;
  vm.createContext(context);
  SOURCES.forEach((src) => vm.runInContext(src, context));
  const run = (expr) => vm.runInContext(expr, context);
  const evalAsync = (expr) => vm.runInContext(`(async () => (${expr}))()`, context);
  return { run, evalAsync, window, store, document: window.document };
}

const results = [];
const check = (name, condition) => results.push([name, !!condition]);

// --- key layout

async function familiesCoverEveryKey() {
  const { run } = load();
  const keys = run("MODEL_KEYS.slice()");
  const claimed = keys.filter((k) => run(`modelFamilyOf(${JSON.stringify(k)})`));
  check("every backend model key belongs to exactly one family", claimed.length === keys.length);
  check(
    "a family plus its granularity resolves back to the same key",
    keys.every((k) => {
      const family = run(`modelFamilyOf(${JSON.stringify(k)})`);
      const granularity = run(`modelFamilyGranularityOf(${JSON.stringify(k)})`);
      return run(`familyModelKey(${JSON.stringify(family)}, ${JSON.stringify(granularity)})`) === k;
    })
  );
  check("the four families are the ones the picker offers", run("MODEL_FAMILY_ORDER.length") === 4);
}

async function statusExplainsWhyNot() {
  const { run } = load();
  const missing = run(`familyGranularityStatus("online", "whole_sections", ${JSON.stringify(HEALTH.models_loaded)})`);
  check("the comparison model reports no document-scope model at all", missing.available === false && missing.key === null);
  check("...and says so rather than blaming loading", /no whole-sections model/.test(missing.reason));

  const notLoaded = run(`familyGranularityStatus("llm_7b", "whole_sections", ${JSON.stringify(HEALTH.models_loaded)})`);
  check("a document key that exists but isn't served reports as not loaded", notLoaded.available === false);
  check("...naming the backend as the reason", /not loaded/.test(notLoaded.reason));
  check("...while still naming the key it would use", notLoaded.key === "llm_doc_7b");

  const fine = run(`familyGranularityStatus("finetuned", "whole_sections", ${JSON.stringify(HEALTH.models_loaded)})`);
  check("a loaded document checkpoint is selectable", fine.available === true && fine.key === "document");

  const unknown = run(`familyGranularityStatus("llm_7b", "whole_sections", null)`);
  check("without /health nothing is claimed to be unloaded", unknown.available === true);
}

// --- what a selection stores

async function granularityIsRememberedAcrossModels() {
  const { evalAsync } = load({ stored: { simplifierModel: "finetuned" } });
  const loaded = JSON.stringify(HEALTH.models_loaded);

  await evalAsync(`writeSelection({ granularity: "whole_sections" }, ${loaded})`);
  let selection = await evalAsync("readSelection()");
  check("asking for sections on the fine-tuned family selects its document checkpoint", selection.model === "document");

  // no document-scope comparison model: the key falls back, but the intent must survive
  await evalAsync(`writeSelection({ family: "online" }, ${loaded})`);
  selection = await evalAsync("readSelection()");
  check("switching to a family with no document model falls back to its sentence key", selection.model === "online");
  check("...and reports the granularity that will actually run", selection.granularity === "sentence_by_sentence");
  check("...while remembering that sections were asked for", selection.granularityIntent === "whole_sections");

  await evalAsync(`writeSelection({ family: "finetuned" }, ${loaded})`);
  selection = await evalAsync("readSelection()");
  check("switching back to a family that has one restores sections", selection.model === "document");
}

async function audienceIsRememberedPerFamily() {
  const { evalAsync } = load({ stored: { simplifierModel: "llm_7b" } });
  const loaded = JSON.stringify(HEALTH.models_loaded);

  await evalAsync(`writeSelection({ audience: "children" }, ${loaded})`);
  await evalAsync(`writeSelection({ family: "llm_3b" }, ${loaded})`);
  await evalAsync(`writeSelection({ audience: "low_literacy" }, ${loaded})`);
  let selection = await evalAsync("readSelection()");
  check("each prompted model keeps its own audience", selection.audience === "low_literacy");

  await evalAsync(`writeSelection({ family: "llm_7b" }, ${loaded})`);
  selection = await evalAsync("readSelection()");
  check("returning to a model restores the audience it was last used with", selection.audience === "children");

  // §6: switching to a model that ignores the field must not reset it
  await evalAsync(`writeSelection({ family: "finetuned" }, ${loaded})`);
  await evalAsync(`writeSelection({ family: "llm_7b" }, ${loaded})`);
  selection = await evalAsync("readSelection()");
  check("a detour through a model without audiences doesn't clear it", selection.audience === "children");
}

async function storedKeysFromEarlierVersionsStillResolve() {
  const { evalAsync } = load({ stored: { simplifierModel: "local" } });
  const selection = await evalAsync("readSelection()");
  check("a renamed key ('local') migrates to its current name", selection.model === "finetuned");
  check("...and lands in the right family", selection.family === "finetuned");

  const legacyDoc = load({ stored: { simplifierModel: "llm_doc_3b" } });
  const doc = await legacyDoc.evalAsync("readSelection()");
  check("a document-scope key is read back as its family at document granularity",
    doc.family === "llm_3b" && doc.granularity === "whole_sections");
}

async function summaryNamesOnlyWhatApplies() {
  const { run } = load();
  const llm = run(`selectionSummary("llm_7b", "whole_sections", "children", ${JSON.stringify(HEALTH)})`);
  check("the summary names model, granularity and audience for a prompted model",
    llm === "Ollama Qwen2.5 · 7B · Whole sections · Children · 8–12");
  const seq = run(`selectionSummary("finetuned", "sentence_by_sentence", "children", ${JSON.stringify(HEALTH)})`);
  check("the summary omits the audience for a model that ignores it",
    seq === "Fine-tuned model · Sentence by sentence");
  const noHealth = run(`selectionSummary("llm_7b", "sentence_by_sentence", "children", null)`);
  check("the summary still reads before /health answers", /· Sentence by sentence · Children/.test(noHealth));
}

// --- panel

const settle = () => new Promise((resolve) => setTimeout(resolve, 20));

function panelRows(document) {
  const host = document.getElementById("simplify-model-picker");
  return host ? Array.from(host.shadowRoot.querySelectorAll(".option[data-family]")) : [];
}

async function panelRendersTheFamilies() {
  const { run, document } = load({ stored: { simplifierModel: "llm_7b" } });
  run("showModelPicker()");
  await settle();

  const rows = panelRows(document);
  check("the panel offers one row per family", rows.length === 4);
  check(
    "each row is a title plus one line saying what the model is",
    rows.every((r) => r.querySelector(".option-title") && r.querySelectorAll(".option-detail").length === 1)
  );
  const checked = rows.filter((r) => r.getAttribute("aria-checked") === "true");
  check("exactly one row is marked as selected", checked.length === 1);
  check("...and it is the stored model's family", checked[0].dataset.family === "llm_7b");
  check("...marked with a filled dot, the others with a ring", checked[0].querySelector(".mark").textContent === "●");
  check(
    "the prompted rows are named from the served tag rather than a hardcoded name",
    checked[0].querySelector(".option-title").textContent === "Ollama Qwen2.5 · 7B"
  );
  check(
    // same method label as the notice, status page and History header (shared/model-labels.js)
    "...with method and trade-off on that one line",
    checked[0].querySelector(".option-detail").textContent === "Prompted LLM · Highest quality · slower"
  );
  const comparison = rows.find((r) => r.dataset.family === "online");
  check(
    "the comparison row's line is its verbatim repo id, alone",
    comparison.querySelector(".option-detail").textContent === HEALTH.model_names.online
  );
  check("...in code style", comparison.querySelector(".option-detail").classList.contains("is-id"));
}

// One flat surface, every choice visible; full-screen backdrop behind the card.
async function panelIsOneFlatSurface() {
  const { run, document } = load({ stored: { simplifierModel: "llm_7b" } });
  run("showModelPicker()");
  await settle();
  const root = document.getElementById("simplify-model-picker").shadowRoot;

  check("the picker fills the viewport behind its card", !!root.querySelector(".overlay > .panel"));
  check("...as a modal, since the page behind it can't be reached", root.querySelector(".panel").getAttribute("aria-modal") === "true");
  check("...with the page's own scrolling held while it covers it", document.documentElement.style.overflow === "hidden");
  check("no dropdown anywhere in the panel", root.querySelectorAll("select").length === 0);
  check("granularity is a segmented control, both options visible", root.querySelectorAll(".segment").length === 2);
  check("audience is an inline list, every option visible", root.querySelectorAll(".option[data-audience]").length === 5);
  const headings = Array.from(root.querySelectorAll(".section-label")).map((el) => el.textContent);
  check("the three groups are named", headings.join("|") === "Model|Granularity|Audience");
  check(
    "every choice in the panel is a radio",
    Array.from(root.querySelectorAll(".option, .segment")).every((el) => el.getAttribute("role") === "radio")
  );
}

async function panelDisablesWithAReason() {
  const { run, document } = load({ stored: { simplifierModel: "llm_7b" } });
  run("showModelPicker()");
  await settle();

  const host = document.getElementById("simplify-model-picker");
  const sections = host.shadowRoot.querySelector('.segment[data-granularity="whole_sections"]');
  check("sections is disabled when the family's document model isn't served", sections.disabled === true);
  check("...with the reason stated in the panel, not only as a tooltip",
    /not loaded on the backend/.test(host.shadowRoot.textContent));
  check("sentence scope stays selectable", host.shadowRoot.querySelector('.segment[data-granularity="sentence_by_sentence"]').disabled === false);
  check("the audience list is shown for a prompted model", !!host.shadowRoot.querySelector(".option[data-audience]"));
}

async function panelShowsAndWritesTheRememberedAudience() {
  const { run, document, store } = load({
    stored: { simplifierModel: "llm_7b", simplifierAudience: "children" },
  });
  run("showModelPicker()");
  await settle();

  const root = () => document.getElementById("simplify-model-picker").shadowRoot;
  const marked = () =>
    Array.from(root().querySelectorAll(".option[data-audience]")).filter(
      (r) => r.getAttribute("aria-checked") === "true"
    );
  check("the audience in force is the one marked", marked().length === 1 && marked()[0].dataset.audience === "children");

  root().querySelector('.option[data-audience="low_literacy"]').click();
  await settle();
  check("clicking another persists it immediately", store.simplifierAudience === "low_literacy");
  check("...and remembers it against this model", store.simplifierAudienceByModel.llm_7b === "low_literacy");
  check("...and the mark moves", marked()[0].dataset.audience === "low_literacy");
  check("...with the panel still open", !!document.getElementById("simplify-model-picker"));
}

async function panelHidesAudienceForSeq2Seq() {
  const { run, document } = load({ stored: { simplifierModel: "finetuned" } });
  run("showModelPicker()");
  await settle();
  const host = document.getElementById("simplify-model-picker");
  check("no audience control for a model that ignores the field", !host.shadowRoot.querySelector(".option[data-audience]"));
  check("...and no heading for it either", !/Audience/.test(host.shadowRoot.textContent));
  check(
    "sections is selectable for the fine-tuned family, whose document checkpoint is served",
    host.shadowRoot.querySelector('.segment[data-granularity="whole_sections"]').disabled === false
  );
}

async function clickingAModelSelectsItImmediately() {
  const { run, document, store } = load({ stored: { simplifierModel: "online" } });
  run("showModelPicker()");
  await settle();

  const host = document.getElementById("simplify-model-picker");
  host.shadowRoot.querySelector('.option[data-family="llm_3b"]').click();
  await settle();
  check("clicking a model persists it with no Save step", store.simplifierModel === "llm_3b");
  check("the panel stays open so its options can be reached", !!document.getElementById("simplify-model-picker"));
  const checked = panelRows(document).filter((r) => r.getAttribute("aria-checked") === "true");
  check("the checkmark moves to the clicked model", checked.length === 1 && checked[0].dataset.family === "llm_3b");

  host.shadowRoot.querySelector('.option[data-family="online"]').click();
  await settle();
  check("switching again writes the new family's key", store.simplifierModel === "online");
}

async function clickingADisabledRowDoesNothing() {
  // only the comparison model registered; clicking another family must not store a key that would 400
  const health = { ...HEALTH, models_loaded: ["online"] };
  const { run, document, store } = load({ stored: { simplifierModel: "online" }, health });
  run("showModelPicker()");
  await settle();

  const host = document.getElementById("simplify-model-picker");
  const unavailable = host.shadowRoot.querySelector('.option[data-family="llm_7b"]');
  check("a family with nothing loaded is disabled rather than hidden", unavailable && unavailable.disabled === true);
  unavailable.click();
  await settle();
  check("clicking it selects nothing", store.simplifierModel === "online");
}

async function panelWorksWithoutABackend() {
  const { run, document } = load({ stored: { simplifierModel: "finetuned" }, health: null });
  run("showModelPicker()");
  await settle();
  const host = document.getElementById("simplify-model-picker");
  check("the panel still opens with the backend down", !!host);
  check("...saying availability is unknown rather than implying it's fine",
    /availability unknown/.test(host.shadowRoot.textContent));
  check("...and every family stays selectable",
    panelRows(document).every((r) => r.disabled === false));
}

async function escapeAndOutsideClickClose() {
  const { run, window, document } = load({ stored: { simplifierModel: "finetuned" } });
  run("showModelPicker()");
  await settle();
  const escape = new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true });
  window.document.dispatchEvent(escape);
  check("Escape closes the panel", !document.getElementById("simplify-model-picker"));

  run("showModelPicker()");
  await settle();
  document.querySelector("p").dispatchEvent(new window.Event("pointerdown", { bubbles: true, composed: true }));
  check("a click on the page closes the panel", !document.getElementById("simplify-model-picker"));

  run("showModelPicker()");
  await settle();
  const host = document.getElementById("simplify-model-picker");
  host.shadowRoot.querySelector(".panel").dispatchEvent(new window.Event("pointerdown", { bubbles: true, composed: true }));
  check("a click inside the card does not", !!document.getElementById("simplify-model-picker"));

  host.shadowRoot.querySelector(".overlay").dispatchEvent(new window.Event("pointerdown", { bubbles: true, composed: true }));
  check("a click on the backdrop around it does", !document.getElementById("simplify-model-picker"));
  check("...and gives the page its scrolling back", document.documentElement.style.overflow === "");
}

// --- after a change
// A change makes an already simplified page stale. The panel reports it once, on close;
// the page owner decides what to do.

async function commitIsReportedOnceOnClose() {
  const ctx = load({ stored: { simplifierModel: "llm_7b" } });
  ctx.run("commits = 0; onPickerSelectionCommitted = () => { commits += 1; };");
  ctx.run("showModelPicker()");
  await settle();

  const root = () => ctx.document.getElementById("simplify-model-picker").shadowRoot;
  root().querySelector('.option[data-family="finetuned"]').click();
  await settle();
  check("nothing is reported while the panel is still open", ctx.run("commits") === 0);

  ctx.run("hideModelPicker()");
  check("closing after a change reports it once", ctx.run("commits") === 1);
}

async function nonChangesAreNotReported() {
  const ctx = load({ stored: { simplifierModel: "llm_7b" } });
  ctx.run("commits = 0; onPickerSelectionCommitted = () => { commits += 1; };");
  ctx.run("showModelPicker()");
  await settle();
  const root = () => ctx.document.getElementById("simplify-model-picker").shadowRoot;

  ctx.run("hideModelPicker()");
  check("opening and closing the panel changes nothing", ctx.run("commits") === 0);

  ctx.run("showModelPicker()");
  await settle();
  root().querySelector('.option[data-family="llm_7b"]').click();
  await settle();
  ctx.run("hideModelPicker()");
  check("re-picking the model already selected is not a change", ctx.run("commits") === 0);

  ctx.run("showModelPicker()");
  await settle();
  // llm_doc_7b isn't served in HEALTH, so this resolves back to the same key
  root().querySelector('.segment[data-granularity="whole_sections"]').click();
  await settle();
  ctx.run("hideModelPicker()");
  check("a granularity that resolves back to the same model is not a change", ctx.run("commits") === 0);
}

// content.js reloads only where there is stale output to drop.
function loadContentScript({ stored = {}, simplified = false } = {}) {
  const ctx = load({ stored });
  const src = fs.readFileSync(path.join(__dirname, "..", "content.js"), "utf8");
  ctx.run(`
    reloads = 0;
    location = { href: "https://example.com/", hostname: "example.com", reload() { reloads += 1; } };
    performance = { now: () => 0 };
    MutationObserver = window.MutationObserver;
    crypto = { randomUUID: () => "test" };
    getComputedStyle = () => ({ color: "rgb(0,0,0)" });
    setInterval = () => 0; clearInterval = () => {}; setTimeout = () => 0; clearTimeout = () => {};
    fetch = () => Promise.reject(new Error("no network"));
  `);
  ctx.run(src);
  ctx.run(`isSimplified = ${simplified};`);
  return ctx;
}

async function aSimplifiedPageReloads() {
  const on = loadContentScript({ simplified: true });
  on.run("onPickerSelectionCommitted()");
  check("a simplified page reloads, dropping the previous model's output", on.run("reloads") === 1);

  const off = loadContentScript({ simplified: false });
  off.run("onPickerSelectionCommitted()");
  check("an untouched page is left alone, scroll and form state included", off.run("reloads") === 0);
}

// --- extension pages' toolbar
// One button reads out the active configuration and opens the same panel; a selection
// made there comes back to the button.

function loadToolbar({ stored = {}, health = HEALTH, hash = "" } = {}) {
  const ctx = load({ stored, health });
  const demo = fs.readFileSync(path.join(SHARED, "demo-pages.js"), "utf8");
  const toolbar = fs.readFileSync(path.join(SHARED, "toolbar.js"), "utf8");
  // toolbar.js is a page script: needs location, getURL, direct /health, history/hash.
  ctx.run(`location = { pathname: "/home.html", hash: ${JSON.stringify(hash)}, reload() {} };`);
  ctx.run(`history = { replaceState() { location.hash = ""; } };`);
  ctx.run(`chrome.runtime.getURL = (p) => "chrome-extension://test/" + p;`);
  ctx.run(`fetch = () => Promise.resolve({ json: () => Promise.resolve(${JSON.stringify(health || {})}) });`);
  ctx.run(demo);
  ctx.run(toolbar);
  return ctx;
}

const pickerButton = (ctx) => ctx.document.getElementById("ext-toolbar-picker");

async function toolbarReadsOutTheSelection() {
  const ctx = loadToolbar({ stored: { simplifierModel: "llm_doc_3b", simplifierAudience: "children" } });
  await settle();
  check("the bar carries no picker of its own any more", ctx.document.querySelectorAll("#ext-toolbar select").length === 0);
  check(
    "one button reads out the whole configuration",
    pickerButton(ctx).textContent === "Ollama Qwen2.5 · 3B · Whole sections · Children · 8–12"
  );

  await ctx.evalAsync(`writeSelection({ family: "finetuned" }, ${JSON.stringify(HEALTH.models_loaded)})`);
  await settle();
  check(
    "a change made anywhere updates it, audience dropped where it doesn't apply",
    pickerButton(ctx).textContent === "Fine-tuned model · Whole sections"
  );
}

async function toolbarButtonOpensTheSamePanel() {
  const ctx = loadToolbar({ stored: { simplifierModel: "llm_7b" } });
  await settle();
  pickerButton(ctx).click();
  await settle();

  const host = ctx.document.getElementById("simplify-model-picker");
  check("the button opens the panel", !!host);
  check("...the same one the web-page surface uses", host.shadowRoot.querySelectorAll(".option[data-family]").length === 4);
  check("...with no dropdown in it", host.shadowRoot.querySelectorAll("select").length === 0);

  host.shadowRoot.querySelector('.option[data-family="finetuned"]').click();
  await settle();
  check("a choice made in the panel is stored", ctx.store.simplifierModel === "finetuned");
  check("...and read back by the button", pickerButton(ctx).textContent.startsWith("Fine-tuned model"));

  pickerButton(ctx).click();
  await settle();
  check("clicking the button again dismisses the panel", !ctx.document.getElementById("simplify-model-picker"));
}

async function toolbarOpensThePanelFromTheFallbackHash() {
  // background.js's fallback where the panel can't be injected
  const ctx = loadToolbar({ stored: { simplifierModel: "llm_7b" }, hash: "#picker" });
  await settle();
  check("arriving at #picker opens the panel", !!ctx.document.getElementById("simplify-model-picker"));
  check("...and clears the hash so a reload doesn't reopen it", ctx.run("location.hash") === "");
}

async function main() {
  await familiesCoverEveryKey();
  await statusExplainsWhyNot();
  await granularityIsRememberedAcrossModels();
  await audienceIsRememberedPerFamily();
  await storedKeysFromEarlierVersionsStillResolve();
  await summaryNamesOnlyWhatApplies();
  await panelRendersTheFamilies();
  await panelIsOneFlatSurface();
  await panelDisablesWithAReason();
  await panelShowsAndWritesTheRememberedAudience();
  await panelHidesAudienceForSeq2Seq();
  await clickingAModelSelectsItImmediately();
  await clickingADisabledRowDoesNothing();
  await panelWorksWithoutABackend();
  await escapeAndOutsideClickClose();
  await commitIsReportedOnceOnClose();
  await nonChangesAreNotReported();
  await aSimplifiedPageReloads();
  await toolbarReadsOutTheSelection();
  await toolbarButtonOpensTheSamePanel();
  await toolbarOpensThePanelFromTheFallbackHash();

  let failed = 0;
  results.forEach(([name, ok]) => {
    if (!ok) failed += 1;
    console.log(`${ok ? "PASS" : "FAIL"}  ${name}`);
  });
  console.log(`\n${results.length - failed}/${results.length} checks passed`);
  process.exit(failed === 0 ? 0 : 1);
}

main();
