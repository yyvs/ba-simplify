// "Models loaded" list on the status dashboard (shared/status.js's renderModelsLoaded),
// the one surface showing every backend key.
//
//   cd extension/tests && npm install && node status-models.test.js
//
// One row per served artifact, not per key. Four prompted keys address two Ollama tags
// (llm_7b/llm_doc_7b share a tag, likewise the 3B pair) and must merge; the fine-tuned
// family's two granularities are different checkpoints and must stay apart.
//
// status.js is a non-module script that polls /health at load; it runs in a vm context
// with jsdom, a fake chrome and a never-resolving fetch.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { JSDOM } = require("jsdom");

const SHARED = path.join(__dirname, "..", "shared");
const SOURCES = ["model-labels.js", "model-selection.js", "status.js"].map((f) =>
  fs.readFileSync(path.join(SHARED, f), "utf8")
);

// Default local setup: document-scope prompted keys use the same tags as their
// sentence-scope siblings (backend/main.py's MODEL_ENV_CONFIG). Verified against a real /health.
const HEALTH = {
  model_loaded: true,
  models_loaded: ["online", "finetuned", "document", "llm_3b", "llm_7b", "llm_doc_3b", "llm_doc_7b"],
  model_names: {
    online: "eilamc14/bart-large-text-simplification",
    finetuned: "yunvs/bart-base-wikilarge-simplification",
    document: "yunvs/bart-base-dwikipedia-simplification",
    llm_3b: "qwen2.5:3b-instruct-q4_K_M",
    llm_7b: "qwen2.5:7b-instruct-q4_K_M",
    llm_doc_3b: "qwen2.5:3b-instruct-q4_K_M",
    llm_doc_7b: "qwen2.5:7b-instruct-q4_K_M",
  },
  granularities: {
    online: "sentence_by_sentence",
    finetuned: "sentence_by_sentence",
    document: "whole_sections",
    llm_3b: "sentence_by_sentence",
    llm_7b: "sentence_by_sentence",
    llm_doc_3b: "whole_sections",
    llm_doc_7b: "whole_sections",
  },
};

function load({ health = HEALTH } = {}) {
  const dom = new JSDOM(
    `<!doctype html><html><body><div id="dash-models">Checking…</div></body></html>`,
    { pretendToBeVisual: true }
  );
  const { window } = dom;
  const context = {
    window,
    document: window.document,
    Node: window.Node,
    URL: window.URL,
    console,
    // renderModelsLoaded is driven directly; the poll is left hanging
    fetch: () => new Promise(() => {}),
    setInterval: () => 0,
    clearInterval: () => {},
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

  vm.runInContext(`renderModelsLoaded(${JSON.stringify(health)})`, context);
  const rows = Array.from(window.document.querySelectorAll(".loaded-model-item")).map((row) => ({
    name: row.querySelector(".loaded-model-name").textContent,
    scopes: row.querySelector(".model-scope")
      ? row.querySelector(".model-scope").textContent
      : null,
    id: row.querySelector(".model-id") ? row.querySelector(".model-id").textContent : null,
  }));
  const run = (expr) => vm.runInContext(expr, context);
  return { rows, run, window };
}

const results = [];
const check = (name, condition) => results.push([name, !!condition]);

function sharedTagsCollapse() {
  const { rows } = load();
  check("seven loaded keys render as five rows", rows.length === 5);

  const ids = rows.map((r) => r.id);
  check("no served id is listed twice", new Set(ids).size === ids.length);

  const sevenB = rows.filter((r) => r.id === "qwen2.5:7b-instruct-q4_K_M");
  check("the 7B tag is one row, not two", sevenB.length === 1);
  check(
    "...named without a scope parenthetical, since the row states the scopes itself",
    sevenB[0].name === "LLM (Qwen2.5 7B)"
  );
  check(
    "...and names both scopes it is serving",
    /sentence by sentence/.test(sevenB[0].scopes) && /whole sections/.test(sevenB[0].scopes)
  );
}

function differentCheckpointsStayApart() {
  const { rows } = load();
  const finetuned = rows.filter((r) => /bart-base-(wikilarge|dwikipedia)/.test(r.id));
  check(
    "the two fine-tuned checkpoints stay two rows, being two different artifacts",
    finetuned.length === 2
  );
  check(
    "...each stating the scope it works at beside the name",
    finetuned.some((r) => r.scopes === "sentence by sentence") &&
      finetuned.some((r) => r.scopes === "whole sections")
  );
}

function everyRowStatesItsScope() {
  const { rows } = load();
  check("every row names the unit of text it works at", rows.every((r) => !!r.scopes));

  // only llm_doc_7b loaded: labels carry no scope, so the span is the only indicator
  const health = {
    ...HEALTH,
    models_loaded: ["llm_doc_7b"],
    model_names: { llm_doc_7b: "qwen2.5:7b-instruct-q4_K_M" },
    granularities: { llm_doc_7b: "whole_sections" },
  };
  const lone = load({ health }).rows;
  check("a lone document-scope key is one row", lone.length === 1);
  check("...named without a scope parenthetical", lone[0].name === "LLM (Qwen2.5 7B)");
  check("...and scoped by its own span instead", lone[0].scopes === "whole sections");
}

function unreportedIdsAreNotMerged() {
  // no ids reported: two unknowns must not merge
  const health = {
    ...HEALTH,
    models_loaded: ["finetuned", "document"],
    model_names: {},
    granularities: { finetuned: "sentence_by_sentence", document: "whole_sections" },
  };
  const { rows } = load({ health });
  check("keys with no reported id each keep their own row", rows.length === 2);
  check("...and show no id", rows.every((r) => r.id === null));
}

// The dashboard is where verbatim ids live (pickers show readable names); grouping must keep them.
function everyRowKeepsItsExactId() {
  const { rows } = load();
  const online = rows.find((r) => r.name === "Comparison fine-tuned model");
  check("the comparison model shows its exact repo id beside the readable name",
    !!online && online.id === "eilamc14/bart-large-text-simplification");
  check("every row reports the id the backend named", rows.every((r) => r.id !== null));
}

function noBackendYet() {
  const { rows, window } = load({ health: null });
  check("no health yet says none loaded rather than rendering rows", rows.length === 0);
  check(
    "...in the container's own text",
    window.document.getElementById("dash-models").textContent === "none loaded"
  );
}

function main() {
  sharedTagsCollapse();
  differentCheckpointsStayApart();
  everyRowStatesItsScope();
  unreportedIdsAreNotMerged();
  everyRowKeepsItsExactId();
  noBackendYet();

  let failed = 0;
  results.forEach(([name, ok]) => {
    if (!ok) failed += 1;
    console.log(`${ok ? "PASS" : "FAIL"}  ${name}`);
  });
  console.log(`\n${results.length - failed}/${results.length} checks passed`);
  process.exit(failed === 0 ? 0 : 1);
}

main();
