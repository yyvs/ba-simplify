/**
 * The audit's `anchor_labels_emptied` count, turned back into markup a person can read.
 *
 *   cd research && node inspect_emptied_labels.mjs --site bbc --cut sentence
 *
 * A non-zero count is either §5.4's fragmentation defect (an <a> kept, its label lost)
 * recurring or a detector flaw; the element in both states tells them apart.
 *
 * Same loader rules and real backend as audit_sites.mjs.
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

const args = { site: "bbc", cut: "sentence", backend: "http://127.0.0.1:8000", limit: "8" };
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
const requests = [];
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
  crypto: { randomUUID: () => "inspect" },
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
              requests.push({ input: message.text, output: data.simplified || "" });
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
const maxTokens =
  (typeof budgets === "object" && budgets !== null ? budgets[cut.model] : budgets) ?? 512;
context.__health = { ok: true, model: cut.model, granularity: cut.granularity, documentMaxTokens: maxTokens };

// Parent element recorded too: an emptied label's words usually ended up in another node
// of the same element.
const before = JSON.parse(
  vm.runInContext(
    `(() => JSON.stringify([...document.body.querySelectorAll("a")].map((a) => ({
        href: a.getAttribute("href"),
        text: a.textContent.replace(/\\s+/g, " ").trim(),
        parentHtml: a.parentElement ? a.parentElement.innerHTML.slice(0, 700) : null,
      }))))()`,
    context
  )
);

vm.runInContext(`beginSession(); requestSelectionInfo();`, context);
await vm.runInContext(`simplifyPage(${JSON.stringify(cut.granularity)}, ${maxTokens})`, context);

const after = JSON.parse(
  vm.runInContext(
    `(() => JSON.stringify([...document.body.querySelectorAll("a")].map((a) => ({
        href: a.getAttribute("href"),
        text: a.textContent.replace(/\\s+/g, " ").trim(),
        parentHtml: a.parentElement ? a.parentElement.innerHTML.slice(0, 700) : null,
        parentOriginal: a.closest("[data-original-html]")
          ? a.closest("[data-original-html]").dataset.originalHtml.slice(0, 700)
          : null,
      }))))()`,
    context
  )
);

console.log(`${args.site} / ${args.cut}: ${before.length} anchors before, ${after.length} after`);
if (before.length !== after.length) {
  console.log("anchor count changed -- positional comparison is not valid here");
  process.exit(0);
}
const emptied = [];
for (let i = 0; i < before.length; i++) {
  if (before[i].text && !after[i].text) emptied.push({ i, before: before[i], after: after[i] });
}
console.log(`${emptied.length} anchors kept their element and lost their label\n`);
for (const e of emptied.slice(0, Number(args.limit))) {
  console.log(`--- anchor #${e.i}  href=${JSON.stringify(e.before.href)}`);
  console.log(`    label before : ${JSON.stringify(e.before.text)}`);
  console.log(`    parent before: ${JSON.stringify(e.before.parentHtml)}`);
  console.log(`    parent after : ${JSON.stringify(e.after.parentHtml)}`);
  console.log(`    element as sent (data-original-html): ${JSON.stringify(e.after.parentOriginal)}`);
  const hit = requests.find((r) => e.before.text && r.input.includes(e.before.text.slice(0, 30)));
  if (hit) {
    console.log(`    a request containing the label:`);
    console.log(`      in : ${JSON.stringify(hit.input.slice(0, 300))}`);
    console.log(`      out: ${JSON.stringify(hit.output.slice(0, 300))}`);
  }
  console.log();
}
dom.window.close();
process.exit(0);
