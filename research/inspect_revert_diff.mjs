/**
 * Where a reverted page still differs from the page that was served.
 *
 *   cd research && node inspect_revert_diff.mjs --site allbirds --cut sentence
 *
 * In audit_sites.mjs the text and attribute-set revert checks are clean on every page,
 * but the raw HTML differs by 1-4 bytes on six pages and by +913 on one. Not
 * serialisation noise: re-parse/re-serialise is idempotent under jsdom for these pages.
 */
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const require = createRequire(import.meta.url);
const { JSDOM } = require("jsdom");
const HERE = path.dirname(fileURLToPath(import.meta.url));
const EXTENSION_DIR = path.join(HERE, "..", "extension");
const CONTENT_SCRIPTS = JSON.parse(
  fs.readFileSync(path.join(EXTENSION_DIR, "manifest.json"), "utf8")
).content_scripts[0].js;

const args = { site: "allbirds", cut: "sentence", backend: "http://127.0.0.1:8000", context: "260" };
for (let i = 2; i < process.argv.length; i++) {
  const [flag, inline] = process.argv[i].split("=");
  const value = inline !== undefined ? inline : process.argv[++i];
  args[flag.replace(/^--/, "")] = value;
}
const CUTS = {
  sentence: { granularity: "sentence_by_sentence", model: "finetuned" },
  document: { granularity: "whole_sections", model: "document" },
};
const cut = CUTS[args.cut];
const meta = JSON.parse(fs.readFileSync(path.join(HERE, "scratch", "sites", `${args.site}.json`), "utf8"));
const html = fs.readFileSync(path.join(HERE, "scratch", "sites", `${args.site}.html`), "utf8");

const dom = new JSDOM(html, { url: meta.final_url || meta.url, pretendToBeVisual: true });
const { window } = dom;
const context = {
  window,
  document: window.document,
  Node: window.Node,
  MutationObserver: window.MutationObserver,
  getComputedStyle: window.getComputedStyle.bind(window),
  location: window.location,
  performance: window.performance,
  setTimeout: window.setTimeout.bind(window),
  clearTimeout: window.clearTimeout.bind(window),
  setInterval: window.setInterval.bind(window),
  clearInterval: window.clearInterval.bind(window),
  console: { log() {}, warn() {}, error() {} },
  crypto: { randomUUID: () => "revert-diff" },
  fetch,
  chrome: {
    runtime: {
      lastError: null,
      onMessage: { addListener() {} },
      sendMessage(message, callback) {
        if (message.cmd === "fetchSimplify") {
          fetch(`${args.backend}/simplify`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: message.text, model: cut.model, audience: "non_native_speakers" }),
          })
            .then(async (res) => {
              const data = await res.json();
              callback({ ok: res.ok, data, model: cut.model });
            })
            .catch((e) => callback({ ok: false, error: String(e) }));
          return;
        }
        if (message.cmd === "healthCheck") return callback(context.__health);
        if (message.cmd === "selectionInfo")
          return callback({ methodLabel: cut.model, modelLabel: cut.model, audienceLabel: null });
        if (callback) callback({ ok: true });
      },
    },
    storage: { local: { get: (_k, cb) => cb && cb({}) } },
  },
};
context.globalThis = context;
vm.createContext(context);
for (const script of CONTENT_SCRIPTS) {
  vm.runInContext(fs.readFileSync(path.join(EXTENSION_DIR, script), "utf8"), context);
}
vm.runInContext(`showNotice = function () {};`, context);

const health = await (await fetch(`${args.backend}/health`)).json();
const budgets = health.document_max_tokens;
const maxTokens = (typeof budgets === "object" && budgets !== null ? budgets[cut.model] : budgets) ?? 512;
context.__health = { ok: true, model: cut.model, granularity: cut.granularity, documentMaxTokens: maxTokens };

const before = vm.runInContext(`document.body.innerHTML`, context);
vm.runInContext(`beginSession(); requestSelectionInfo();`, context);
await vm.runInContext(`simplifyPage(${JSON.stringify(cut.granularity)}, ${maxTokens})`, context);
vm.runInContext(`stopObserver(); revertPage(); endSession();`, context);
const after = vm.runInContext(`document.body.innerHTML`, context);

console.log(`${args.site}/${args.cut}: ${before.length} -> ${after.length} (${after.length - before.length})`);
if (before === after) {
  console.log("byte-identical");
  dom.window.close();
  process.exit(0);
}
// Written out for a real diff tool: a hand-rolled string walker reports one shifted byte
// as many divergences and misses a gain and loss that cancel out.
const dir = process.env.TMPDIR || "/tmp";
const beforeFile = path.join(dir, `revert_${args.site}_${args.cut}_before.html`);
const afterFile = path.join(dir, `revert_${args.site}_${args.cut}_after.html`);
// one tag per line, so a line diff lands on the change
const split = (h) => h.replace(/></g, ">\n<");
fs.writeFileSync(beforeFile, split(before));
fs.writeFileSync(afterFile, split(after));
console.log(`wrote ${beforeFile}`);
console.log(`wrote ${afterFile}`);
dom.window.close();
process.exit(0);
