/**
 * Latency profile of the deployed extension+backend, per request regime.
 *
 *   cd research && npm install
 *   node profile_deployment.mjs --conditions finetuned,document
 *   node profile_deployment.mjs --page igda --conditions finetuned,online,document --repeats 2
 *
 * §6.4: the sentence and document paths have request profiles an order of magnitude
 * apart (173 small requests for one Wikipedia article vs. ~one per section), but no
 * latency figure for either. This measures them.
 *
 * - The extension's own code collects and queues: content.js runs under jsdom, including
 *   the 8-slot queue (MAX_CONCURRENT_SIMPLIFY). Only chrome.runtime is stubbed;
 *   fetchSimplify does real HTTP to a real backend.
 * - Real Wikipedia articles, not demo fixtures (the fragmentation defect only appeared on
 *   a real article). Cached under scratch/pages/, source URL and revision id recorded.
 * - Cold and warm passes are never averaged: the backend caches on (model, cleaned_text).
 *   --repeats 2 reports both, labelled; the gap is what re-toggling a page costs.
 *
 * Not measured: browser DOM write cost (jsdom is not Chrome), and CPU contention with the
 * page. Both stated in the artifact's `notes`.
 */

import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const require = createRequire(import.meta.url);
const { JSDOM } = require("jsdom");

const HERE = path.dirname(fileURLToPath(import.meta.url));
const EXTENSION_DIR = path.join(HERE, "..", "extension");
const PAGES_DIR = path.join(HERE, "scratch", "pages");

/**
 * Content scripts in manifest order. Loading content.js alone broke `--conditions
 * document` after commit 963a29b (2026-08-21) moved the granularity constants to
 * shared/model-labels.js (`GRANULARITY_WHOLE_SECTIONS is not defined`). The P3 row in
 * RESULTS.md §2c (measured 2026-08-17) is unaffected.
 */
const CONTENT_SCRIPTS = JSON.parse(
  fs.readFileSync(path.join(EXTENSION_DIR, "manifest.json"), "utf8"),
).content_scripts[0].js;

// Wikipedia REST HTML: the article markup without wikitext or skin chrome.
const PAGES = {
  igda: {
    title: "International Game Developers Association",
    url: "https://en.wikipedia.org/api/rest_v1/page/html/International_Game_Developers_Association",
    why: "the article §6.4's Finding 1 and Finding 7 were measured on, so request counts here are comparable to those",
  },
  text_simplification: {
    title: "Text simplification",
    url: "https://en.wikipedia.org/api/rest_v1/page/html/Text_simplification",
    why: "a short article: shows how the two profiles differ when there is barely a page to divide",
  },
};

function parseArgs(argv) {
  const args = {
    page: "igda",
    conditions: "finetuned,document",
    backend: "http://127.0.0.1:8000",
    // shared/model-labels.js DEFAULT_AUDIENCE; only the prompted conditions use it
    audience: "non_native_speakers",
    repeats: 1,
    out: path.join(HERE, "results", "deployment_profile"),
  };
  for (let i = 2; i < argv.length; i++) {
    const [flag, inline] = argv[i].split("=");
    const value = inline !== undefined ? inline : argv[++i];
    const key = flag
      .replace(/^--/, "")
      .replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    if (!(key in args)) throw new Error(`unknown flag ${flag}`);
    args[key] = key === "repeats" ? Number(value) : value;
  }
  return args;
}

async function loadPage(name) {
  const spec = PAGES[name];
  const file = spec ? path.join(PAGES_DIR, `${name}.html`) : path.resolve(name);
  if (!fs.existsSync(file)) {
    if (!spec) throw new Error(`page file not found: ${file}`);
    fs.mkdirSync(PAGES_DIR, { recursive: true });
    process.stderr.write(`fetching ${spec.url}\n`);
    const res = await fetch(spec.url, {
      headers: { "User-Agent": "thesis-latency-profiler/1.0" },
    });
    if (!res.ok) throw new Error(`fetch failed: HTTP ${res.status}`);
    fs.writeFileSync(file, await res.text());
  }
  const html = fs.readFileSync(file, "utf8");
  // revision id from <html about="…/revision/1234">: articles get edited
  const revision = (html.match(/\/revision\/(\d+)/) || [])[1] || null;
  return {
    name,
    file: path.relative(HERE, file),
    title: spec?.title ?? name,
    url: spec?.url ?? null,
    why: spec?.why ?? "supplied by --page",
    revision,
    bytes: html.length,
    sha: createHash("sha256").update(html).digest("hex").slice(0, 16),
    html,
  };
}

const median = (xs) => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
};
const quantile = (xs, q) => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
};
const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const round = (x, places = 1) => (x == null ? null : Number(x.toFixed(places)));

/**
 * Loads content.js with a jsdom document and a chrome stub whose fetchSimplify does
 * real HTTP. Returns the loaded context plus the request log it writes into.
 */
function loadExtension(
  pageHtml,
  { backend, model, audience, requests, concurrency },
) {
  const dom = new JSDOM(pageHtml, {
    url: "https://en.wikipedia.org/",
    pretendToBeVisual: true,
  });
  const { window } = dom;

  const context = {
    window,
    document: window.document,
    Node: window.Node,
    MutationObserver: window.MutationObserver,
    getComputedStyle: window.getComputedStyle.bind(window),
    // read by content.js when it flushes a History batch
    location: window.location,
    performance: window.performance,
    setTimeout: window.setTimeout.bind(window),
    clearTimeout: window.clearTimeout.bind(window),
    setInterval: window.setInterval.bind(window),
    clearInterval: window.clearInterval.bind(window),
    console: { log() {}, warn: console.warn, error: console.error },
    crypto: { randomUUID: () => "profiler-session" },
    fetch,
    chrome: {
      runtime: {
        lastError: null,
        onMessage: { addListener() {} },
        sendMessage(message, callback) {
          if (message.cmd === "fetchSimplify") {
            // HTTP call only, excluding queue wait: same boundary as the extension's
            // own timer (sendSimplifyRequest).
            const startedAt = performance.now();
            concurrency.now += 1;
            concurrency.peak = Math.max(concurrency.peak, concurrency.now);
            fetch(`${backend}/simplify`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ text: message.text, model, audience }),
            })
              .then(async (res) => {
                const data = await res.json();
                const elapsed = performance.now() - startedAt;
                concurrency.now -= 1;
                requests.push({
                  chars: message.text.length,
                  words: message.text.trim().split(/\s+/).filter(Boolean)
                    .length,
                  latencyMs: elapsed,
                  cached: !!data.cached,
                  changed: data.simplified !== message.text,
                  outputChars: (data.simplified || "").length,
                  fallbackReason: data.fallback_reason || null,
                });
                callback({ ok: res.ok, data, model, audience });
              })
              .catch((e) => {
                concurrency.now -= 1;
                callback({ ok: false, error: String(e) });
              });
            return;
          }
          if (message.cmd === "healthCheck") {
            callback(context.__health);
            return;
          }
          if (message.cmd === "selectionInfo") {
            callback({
              methodLabel: model,
              modelLabel: model,
              audienceLabel: null,
            });
            return;
          }
          // updateBadge / recordHistory / openErrorPage: service-worker side, not measured
          if (callback) callback({ ok: true });
        },
      },
      storage: { local: { get: (_k, cb) => cb && cb({}) } },
    },
  };
  context.globalThis = context;
  vm.createContext(context);
  for (const script of CONTENT_SCRIPTS) {
    vm.runInContext(
      fs.readFileSync(path.join(EXTENSION_DIR, script), "utf8"),
      context,
    );
  }
  // the notice is the only part that needs a rendered document; keep it out of timings
  vm.runInContext(`showNotice = function () {};`, context);
  return { context, window, run: (expr) => vm.runInContext(expr, context) };
}

/**
 * Mirrors background.js checkBackendHealth(): GET /health, model loaded, test generation.
 * A fixed per-toggle cost (§3.4), reported separately from the page run.
 */
async function preflight({ backend, model, audience }) {
  const startedAt = performance.now();
  const health = await (await fetch(`${backend}/health`)).json();
  if (!health.models_loaded?.includes(model)) {
    throw new Error(
      `model '${model}' not loaded on the backend (loaded: ${health.models_loaded?.join(", ") || "none"})`,
    );
  }
  const testRes = await fetch(`${backend}/simplify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: "This is a short connectivity test sentence.",
      model,
      audience,
    }),
  });
  if (!testRes.ok)
    throw new Error(`preflight test request failed: HTTP ${testRes.status}`);
  await testRes.json();
  return { elapsedMs: performance.now() - startedAt, health };
}

function summarise(requests, wallMs) {
  const latencies = requests.map((r) => r.latencyMs);
  const cold = requests.filter((r) => !r.cached);
  const chars = sum(requests.map((r) => r.chars));
  return {
    requests: requests.length,
    served_from_cache: requests.length - cold.length,
    chars_sent_total: chars,
    chars_per_request_median: median(requests.map((r) => r.chars)),
    chars_per_request_max: requests.length
      ? Math.max(...requests.map((r) => r.chars))
      : null,
    words_per_request_median: median(requests.map((r) => r.words)),
    latency_ms: {
      median: round(median(latencies)),
      mean: round(latencies.length ? sum(latencies) / latencies.length : null),
      p95: round(quantile(latencies, 0.95)),
      max: round(latencies.length ? Math.max(...latencies) : null),
      min: round(latencies.length ? Math.min(...latencies) : null),
    },
    // what the user waits; per-request latency is not, since eight run at once
    wall_ms: round(wallMs),
    requests_per_second: round(requests.length / (wallMs / 1000), 2),
    // comparable across regimes (40 sentences vs. 9 sections); requests/s is not
    chars_per_second: round(chars / (wallMs / 1000)),
    changed: requests.filter((r) => r.changed).length,
    rejected_by_guards: requests.filter((r) => r.fallbackReason).length,
  };
}

async function profileCondition({ page, model, args, pass }) {
  const requests = [];
  const concurrency = { now: 0, peak: 0 };
  const loadBefore = os.loadavg().map((x) => round(x, 2));
  const { context, run } = loadExtension(page.html, {
    backend: args.backend,
    model,
    audience: args.audience,
    requests,
    concurrency,
  });

  const pre = await preflight({
    backend: args.backend,
    model,
    audience: args.audience,
  });
  const granularity =
    pre.health.granularities?.[model] ?? "sentence_by_sentence";
  // Per-model map (checkpoint max_length vs. LLM context window); older backends
  // report a single number.
  const budgets = pre.health.document_max_tokens;
  const maxTokens =
    (typeof budgets === "object" && budgets !== null
      ? budgets[model]
      : budgets) ??
    pre.health.document_max_tokens_default ??
    512;
  context.__health = {
    ok: true,
    model,
    granularity,
    documentMaxTokens: maxTokens,
  };

  // Same calls as toggleSimplification(), which itself returns nothing to await.
  run(`beginSession(); requestSelectionInfo();`);
  const startedAt = performance.now();
  await run(`simplifyPage(${JSON.stringify(granularity)}, ${maxTokens})`);
  const wallMs = performance.now() - startedAt;

  const counters = run(
    `({ found: foundTotal, skipped: skippedCount, sent: sentCount, changed: changedCount })`,
  );
  run(`stopObserver(); endSession();`);

  return {
    model,
    granularity,
    pass,
    // Host state moves these numbers more than design does: two cold runs of the same
    // condition, hours apart on the same host, differed 7x in wall clock (§6.2: ~4x on
    // the prompted path).
    host_load: {
      start: loadBefore,
      end: os.loadavg().map((x) => round(x, 2)),
      cores: os.cpus().length,
    },
    document_max_tokens: granularity === "whole_sections" ? maxTokens : null,
    preflight_ms: round(pre.elapsedMs),
    // the extension's own tally, as shown in its progress notice
    page_units: counters,
    peak_concurrent_requests: concurrency.peak,
    ...summarise(requests, wallMs),
    latency_samples_ms: requests.map((r) => round(r.latencyMs)),
    request_chars: requests.map((r) => r.chars),
  };
}

function gitCommit() {
  try {
    return execFileSync("git", ["rev-parse", "--short", "HEAD"], {
      cwd: HERE,
      encoding: "utf8",
    }).trim();
  } catch {
    return null;
  }
}

async function main() {
  const args = parseArgs(process.argv);
  const page = await loadPage(args.page);
  const models = args.conditions
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  const startedAt = new Date().toISOString();
  const results = [];
  for (let pass = 1; pass <= args.repeats; pass++) {
    for (const model of models) {
      process.stderr.write(
        `profiling ${model} (pass ${pass}/${args.repeats})…\n`,
      );
      const row = await profileCondition({ page, model, args, pass });
      process.stderr.write(
        `  ${row.requests} requests, median ${row.latency_ms.median}ms, ` +
          `wall ${row.wall_ms}ms, ${row.chars_per_second} chars/s, ` +
          `${row.served_from_cache} cached\n`,
      );
      // The backend cache lives as long as its process; re-profiling without a restart
      // measures cache lookups. A cold pass only hits it where the page repeats a string.
      if (
        pass === 1 &&
        row.requests > 0 &&
        row.served_from_cache / row.requests > 0.25
      ) {
        process.stderr.write(
          `  WARNING: ${row.served_from_cache}/${row.requests} served from cache on a ` +
            `pass-1 run — restart the backend before treating this as a cold measurement\n`,
        );
      }
      results.push(row);
    }
  }

  const health = await (await fetch(`${args.backend}/health`)).json();
  const report = {
    run: {
      script: "profile_deployment.mjs",
      script_version: "1.0",
      started_at: startedAt,
      finished_at: new Date().toISOString(),
      git_commit: gitCommit(),
      node: process.version,
      host: {
        platform: `${os.platform()} ${os.release()}`,
        cpu: os.cpus()[0]?.model ?? null,
        cores: os.cpus().length,
        total_memory_gb: round(os.totalmem() / 1024 ** 3),
      },
    },
    page,
    backend: {
      origin: args.backend,
      models_loaded: health.models_loaded,
      model_names: health.model_names,
      granularities: health.granularities,
      document_max_tokens: health.document_max_tokens,
    },
    client: {
      max_concurrent_simplify: 8,
      source: `${CONTENT_SCRIPTS.join(", ")} (manifest.json's content_scripts), loaded under jsdom`,
      audience: args.audience,
    },
    passes: args.repeats,
    results,
    notes: [
      "Pass 1 is cold; later passes are served largely from the backend's (model, text) cache and measure the cache, not the model.",
      "Per-request latency is measured around the HTTP call only, excluding the client's 8-slot queue wait -- the same boundary the extension's own notice uses.",
      "Wall clock covers collection plus every request for one page; the preflight is reported separately because it is a fixed per-toggle cost (the thesis §3.4).",
      "jsdom is not a browser: DOM write and layout costs are not included, and no page JavaScript competes for the CPU.",
      "Measured on an otherwise-idle machine. Prompted-LLM throughput specifically degrades ~4x under sustained load (the thesis §6.2), so LLM rows here are a best case.",
      "Run the backend WITHOUT uvicorn --reload (i.e. not via run_dev.sh): an edit landing mid-run restarts the app, which both clears the cache and changes the system under test, and the resulting numbers look ordinary. This happened once during development and is why the note exists.",
      "The backend loads seq2seq checkpoints with no device placement, so these are CPU numbers on the profiling host -- not comparable to the T4 figures in RESULTS.md's evaluation rows.",
    ],
  };

  delete report.page.html;
  fs.mkdirSync(args.out, { recursive: true });
  const stamp = startedAt.replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
  const file = path.join(args.out, `profile_${page.name}_${stamp}.json`);
  fs.writeFileSync(file, JSON.stringify(report, null, 2));
  process.stderr.write(`\nWrote ${path.relative(HERE, file)}\n`);
}

main().catch((e) => {
  process.stderr.write(`${e.stack || e}\n`);
  process.exit(1);
});
