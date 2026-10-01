/**
 * RQ2's site-sample audit: what the extension's content selection does to real pages.
 *
 *   cd research && npm install
 *   node audit_sites.mjs                          # every site, both cuts
 *   node audit_sites.mjs --sites bbc,mdn --cuts sentence
 *
 * Measures how much of a real page the extension touches, how much it leaves alone, and
 * whether the page survives the visit (the "no systematic testing across real websites"
 * gap noted in §1.3, §5.6, §7.1). Twelve pages, six categories.
 *
 * - The extension's own code does the selecting: content.js runs under jsdom, only
 *   chrome.runtime is stubbed (same loader shape as profile_deployment.mjs).
 *   Reimplementing the selection would measure a system nobody runs.
 * - Skip reasons come from content.js's skipBreakdown(), not a copy of the guard order.
 * - Preservation is checked by diff: anchors (href + text), images, <sup>s and the tag
 *   multiset before vs. after, then revertPage() output vs. the pristine HTML.
 * - Pages are cached under scratch/sites/ with fetch time and a sha256 prefix, so a
 *   finding can be traced to the exact markup.
 *
 * Not covered: jsdom does not run page JS, so these are shipped DOMs, not post-hydration
 * DOMs (the two SPA entries are server-rendered for that reason). No layout, so no
 * visual regressions.
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
const SITES_DIR = path.join(HERE, "scratch", "sites");

// Content scripts in manifest order, read from the manifest so none can go missing.
// content.js alone is not enough: GRANULARITY_WHOLE_SECTIONS, model labels and the run
// lock live in shared/ (ReferenceError in document mode otherwise).
const CONTENT_SCRIPTS = JSON.parse(
  fs.readFileSync(path.join(EXTENSION_DIR, "manifest.json"), "utf8")
).content_scripts[0].js;

// Several sites serve a bot-challenge page to non-browser UAs.
const UA =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " +
  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

/**
 * Two pages per category, picked before any was run. `why` records what each is in the
 * sample for.
 *
 * E-commerce is a compromise: Amazon, eBay, Waterstones, Decathlon and Patagonia all
 * answered a scripted fetch with a bot challenge, so these are shops that serve their
 * product page to anyone. The write-up states this limit.
 */
const SITES = {
  // news
  bbc: {
    category: "news",
    title: "BBC News article",
    url: "https://www.bbc.com/news/articles/c0k3700zljjo",
    why: "a mainstream English news article: dense boilerplate around a short body, and an <article> to find it with",
  },
  tagesschau: {
    category: "news",
    title: "tagesschau article (German)",
    url: "https://www.tagesschau.de/ausland/asien/gaza-wiederaufbau-100.html",
    why: "a German article. Both checkpoints are English-only, so this is the case where selection can be right and simplification cannot be",
  },
  // encyclopedic
  wikipedia: {
    category: "encyclopedic",
    title: "Wikipedia: Photosynthesis",
    url: "https://en.wikipedia.org/api/rest_v1/page/html/Photosynthesis",
    why: "the shape the training data was drawn from, and the page type §6.4's findings were measured on: heavy citation markup around clean prose",
  },
  sep: {
    category: "encyclopedic",
    title: "Stanford Encyclopedia of Philosophy: Consciousness",
    url: "https://plato.stanford.edu/entries/consciousness/",
    why: "long-form academic prose, hand-written HTML, hundreds of paragraphs: encyclopedic without being Wikipedia-shaped",
  },
  // government
  govuk: {
    category: "government",
    title: "GOV.UK guidance: Living in Germany",
    url: "https://www.gov.uk/guidance/living-in-germany",
    why: "public-information prose already written to a plain-English standard -- the case where the correct output is close to the input",
  },
  europa: {
    category: "government",
    title: "Your Europe: entry/exit rules for EU citizens",
    url: "https://europa.eu/youreurope/citizens/travel/entry-exit/eu-citizen/index_en.htm",
    why: "an EU portal page: accordions, country selectors and legalese, i.e. interface and prose interleaved in the same containers",
  },
  // e-commerce
  ikea: {
    category: "ecommerce",
    title: "IKEA product page (BILLY bookcase)",
    url: "https://www.ikea.com/gb/en/p/billy-bookcase-white-00263850/",
    why: "a product page: prices, badge counters, measurement tables and icon glyphs -- the content that produced the '0'/'+' hallucination in §5.4",
  },
  allbirds: {
    category: "ecommerce",
    title: "Allbirds product page (Wool Runner Go)",
    url: "https://allbirds.com/products/mens-wool-runner-go",
    why: "a Shopify storefront: marketing copy in utility-class <div>s with no semantic article anywhere",
  },
  // documentation
  mdn: {
    category: "documentation",
    title: "MDN: CSS flex",
    url: "https://developer.mozilla.org/en-US/docs/Web/CSS/flex",
    why: "the code-adjacent case the shouldSimplifyElement() code filter exists for: prose and syntax blocks in the same document",
  },
  pydocs: {
    category: "documentation",
    title: "Python docs: asyncio tasks",
    url: "https://docs.python.org/3/library/asyncio-task.html",
    why: "reference documentation where inline <code> sits mid-sentence rather than in a block -- the filter's harder half",
  },
  // SPA / modern web app
  reactdev: {
    category: "spa",
    title: "react.dev: Thinking in React",
    url: "https://react.dev/learn/thinking-in-react",
    why: "a server-rendered React app: prose inside a component tree, with hydration wrappers between every paragraph and its container",
  },
  nextjs: {
    category: "spa",
    title: "Next.js docs: Installation",
    url: "https://nextjs.org/docs/app/getting-started/installation",
    why: "an App Router site whose shipped DOM carries RSC payload and utility classes around the same prose",
  },
};

const CUTS = {
  sentence: { granularity: "sentence_by_sentence", model: "finetuned" },
  document: { granularity: "whole_sections", model: "document" },
};

function parseArgs(argv) {
  const args = {
    sites: Object.keys(SITES).join(","),
    cuts: "sentence,document",
    backend: "http://127.0.0.1:8000",
    audience: "non_native_speakers",
    refetch: "no",
    out: path.join(HERE, "results", "site_audit"),
  };
  for (let i = 2; i < argv.length; i++) {
    const [flag, inline] = argv[i].split("=");
    const value = inline !== undefined ? inline : argv[++i];
    const key = flag.replace(/^--/, "").replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    if (!(key in args)) throw new Error(`unknown flag ${flag}`);
    args[key] = value;
  }
  return args;
}

async function loadSite(name, { refetch }) {
  const spec = SITES[name];
  if (!spec) throw new Error(`unknown site '${name}' (have: ${Object.keys(SITES).join(", ")})`);
  const file = path.join(SITES_DIR, `${name}.html`);
  const metaFile = path.join(SITES_DIR, `${name}.json`);
  if (refetch === "yes" || !fs.existsSync(file)) {
    fs.mkdirSync(SITES_DIR, { recursive: true });
    process.stderr.write(`fetching ${spec.url}\n`);
    const res = await fetch(spec.url, { headers: { "User-Agent": UA }, redirect: "follow" });
    if (!res.ok) throw new Error(`fetch failed for ${name}: HTTP ${res.status}`);
    const body = await res.text();
    fs.writeFileSync(file, body);
    // fetch time kept with the cache: a news article's "as of" is part of the finding
    fs.writeFileSync(
      metaFile,
      JSON.stringify({ url: spec.url, fetched_at: new Date().toISOString(), final_url: res.url }, null, 2)
    );
  }
  const html = fs.readFileSync(file, "utf8");
  const meta = fs.existsSync(metaFile) ? JSON.parse(fs.readFileSync(metaFile, "utf8")) : {};
  return {
    name,
    category: spec.category,
    title: spec.title,
    url: spec.url,
    why: spec.why,
    file: path.relative(HERE, file),
    fetched_at: meta.fetched_at ?? null,
    final_url: meta.final_url ?? null,
    bytes: html.length,
    sha256: createHash("sha256").update(html).digest("hex").slice(0, 16),
    html,
  };
}

/**
 * Loads the content scripts against one page's DOM; chrome.runtime is stubbed,
 * fetchSimplify hits the real backend.
 *
 * Copied from profile_deployment.mjs's loader rather than shared: that one carries
 * request-log and timing hooks this audit does not need.
 */
function loadExtension(pageHtml, { pageUrl, backend, model, audience, requests }) {
  const dom = new JSDOM(pageHtml, { url: pageUrl, pretendToBeVisual: true });
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
    crypto: { randomUUID: () => "audit-session" },
    fetch,
    chrome: {
      runtime: {
        lastError: null,
        onMessage: { addListener() {} },
        sendMessage(message, callback) {
          if (message.cmd === "fetchSimplify") {
            fetch(`${backend}/simplify`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ text: message.text, model, audience }),
            })
              .then(async (res) => {
                const data = await res.json();
                requests.push({
                  chars: message.text.length,
                  words: message.text.trim().split(/\s+/).filter(Boolean).length,
                  changed: data.simplified !== message.text,
                  // a warm cache makes wall_ms a cache lookup, not a page
                  cached: !!data.cached,
                  // Backend guard (change_guard.py), distinct from a client-side skip.
                  // Means "at least one sentence hit a guard", not "request rejected":
                  // backend/main.py reports only the first sentence that fell back, so a
                  // flagged request can still come back changed (SEP: 435 of 480
                  // flagged, 405 changed).
                  fallbackReason: data.fallback_reason || null,
                  sentences: data.sentence_split?.outputs?.length ?? null,
                  input: message.text,
                  output: data.simplified || "",
                });
                callback({ ok: res.ok, data, model, audience });
              })
              .catch((e) => callback({ ok: false, error: String(e) }));
            return;
          }
          if (message.cmd === "healthCheck") {
            callback(context.__health);
            return;
          }
          if (message.cmd === "selectionInfo") {
            callback({ methodLabel: model, modelLabel: model, audienceLabel: null });
            return;
          }
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
  // The notice box is the only UI a run adds to the page; keep it out of the DOM diffs.
  vm.runInContext(`showNotice = function () {};`, context);
  return { context, window, dom, run: (expr) => vm.runInContext(expr, context) };
}

// what the page looked like, before and after

/**
 * Page structure a text rewrite should not change, snapshotted before and after.
 * Anchors are (href, text) pairs, not a count: the §5.4 defect kept every <a> and
 * emptied its label.
 */
const SNAPSHOT_EXPR = `(() => {
  const scope = document.body;
  const tags = {};
  for (const el of scope.querySelectorAll("*")) {
    tags[el.tagName] = (tags[el.tagName] || 0) + 1;
  }
  return JSON.stringify({
    anchors: [...scope.querySelectorAll("a")].map((a) => ({
      href: a.getAttribute("href"),
      text: a.textContent.replace(/\\s+/g, " ").trim(),
    })),
    images: [...scope.querySelectorAll("img")].map((i) => i.getAttribute("src") || ""),
    sups: scope.querySelectorAll("sup").length,
    tags,
    text_chars: scope.textContent.length,
    html_chars: scope.innerHTML.length,
  });
})()`;

/**
 * Every element's attribute *names*, in document order. Catches gained/lost attributes
 * that a tag census misses (seen on the first page: an added empty `class=""`, a page
 * `title` removed by the revert). Names only: a changed value is the rewrite, a changed
 * attribute set is damage.
 *
 * Joined by document order, so only valid when no element was added or removed;
 * otherwise the comparison is skipped.
 */
const ATTR_NAMES_EXPR = `(() => {
  const out = [];
  for (const el of document.body.querySelectorAll("*")) {
    out.push(el.tagName + ":" + [...el.attributes].map((a) => a.name).sort().join(","));
  }
  return JSON.stringify(out);
})()`;

/**
 * Outcome of every registered leaf, from content.js's itemStatus (the same map the
 * notice counters use).
 *
 * Placeholders (<input>/<textarea>) are not in CANDIDATE_SELECTOR, so `units` counts
 * them and this list does not: off by one or two on pages with a search box. Left as-is
 * since all judgements here are about prose; the discrepancy is stated where quoted.
 *
 * `words` and `in_boilerplate` separate a skipped nav label (filter working) from a
 * skipped paragraph inside <article> (filter misfiring).
 */
const OUTCOMES_EXPR = `(() => {
  const scope = findContentScope() || document.body;
  const out = [];
  for (const el of document.querySelectorAll(CANDIDATE_SELECTOR)) {
    const status = itemStatus.get(el);
    if (status === undefined) continue;
    const text = el.textContent.replace(/\\s+/g, " ").trim();
    out.push({
      tag: el.tagName,
      status,
      changed: changedItems.has(el),
      words: wordCount(text),
      in_scope: scope.contains(el),
      // only per-element evidence of "reached" in document mode (see analyseOutcomes)
      rewritten: el.dataset.originalHtml !== undefined,
      in_boilerplate: el.closest("nav, footer, header, aside, form") !== null,
      reason: status === "skipped" ? (el.dataset.originalHtml ? "already-simplified" : elementSkipReason(el)) : null,
      text: text.slice(0, 200),
    });
  }
  return JSON.stringify(out);
})()`;

// Attribute sets that changed across the revert, split by direction. Nothing is
// excluded: data-original-* should be gone by now, so a leftover one counts as gained.
function diffAttributes(before, after) {
  if (before.length !== after.length) {
    return { comparable: false, reason: "element count differs; positions do not align" };
  }
  const gained = [];
  const lost = [];
  for (let i = 0; i < before.length; i++) {
    if (before[i] === after[i]) continue;
    const [tag, b = ""] = [before[i].split(":")[0], before[i].split(":").slice(1).join(":")];
    const a = after[i].split(":").slice(1).join(":");
    const setB = new Set(b ? b.split(",") : []);
    const setA = new Set(a ? a.split(",") : []);
    const extra = [...setA].filter((x) => !setB.has(x));
    const missing = [...setB].filter((x) => !setA.has(x));
    if (extra.length) gained.push({ tag, attributes: extra });
    if (missing.length) lost.push({ tag, attributes: missing });
  }
  const tally = (rows) =>
    rows.reduce((acc, r) => {
      for (const a of r.attributes) acc[a] = (acc[a] || 0) + 1;
      return acc;
    }, {});
  return {
    comparable: true,
    elements_with_gained_attributes: gained.length,
    elements_with_lost_attributes: lost.length,
    gained_by_attribute: tally(gained),
    lost_by_attribute: tally(lost),
    samples: { gained: gained.slice(0, 5), lost: lost.slice(0, 5) },
  };
}

/**
 * Splits emptied link labels by cause:
 * (a) the model deleted the linked phrase -- content loss (model / input side);
 * (b) the model kept the words but the write-back put them in another node of the same
 *     element -- the §5.4 class, owned by alignToNodes().
 * Decided by whether the label's words are still inside the rewritten host element.
 * Needs the live DOM (each anchor's rewritten host), so it runs in-page.
 */
const anchorLossExpr = (beforeAnchors) => `(() => {
  const before = ${JSON.stringify(beforeAnchors)};
  const anchors = [...document.body.querySelectorAll("a")];
  if (anchors.length !== before.length) return JSON.stringify({ comparable: false });
  const misplaced = [], deleted = [], orphaned = [];
  for (let i = 0; i < anchors.length; i++) {
    const was = before[i].text;
    const now = anchors[i].textContent.replace(/\s+/g, " ").trim();
    if (!was || now) continue;
    const host = anchors[i].closest("[data-original-html]");
    if (!host) {
      // no rewritten host: neither cause, i.e. the metric is measuring something else
      orphaned.push({ href: before[i].href, label: was });
      continue;
    }
    const hostText = host.textContent.replace(/\s+/g, " ");
    (hostText.includes(was) ? misplaced : deleted).push({
      href: before[i].href,
      label: was,
      host_tag: host.tagName,
      original: (host.dataset.originalText || "").slice(0, 200),
      now: hostText.slice(0, 200),
    });
  }
  return JSON.stringify({
    comparable: true,
    emptied: misplaced.length + deleted.length + orphaned.length,
    misplaced_by_write_back: misplaced.length,
    deleted_by_model: deleted.length,
    orphaned_no_rewritten_host: orphaned.length,
    samples: { misplaced: misplaced.slice(0, 4), deleted: deleted.slice(0, 4), orphaned: orphaned.slice(0, 4) },
  });
})()`;

function diffSnapshots(before, after) {
  const tagsLost = {};
  for (const [tag, n] of Object.entries(before.tags)) {
    const now = after.tags[tag] || 0;
    if (now < n) tagsLost[tag] = n - now;
  }
  const tagsGained = {};
  for (const [tag, n] of Object.entries(after.tags)) {
    const was = before.tags[tag] || 0;
    if (n > was) tagsGained[tag] = n - was;
  }

  // Matched by position; meaningless if the counts differ, so those fields go null.
  const hrefsBefore = before.anchors.map((a) => a.href);
  const hrefsAfter = after.anchors.map((a) => a.href);
  const sameLength = hrefsBefore.length === hrefsAfter.length;
  const hrefsChanged = sameLength
    ? hrefsBefore.filter((h, i) => h !== hrefsAfter[i]).length
    : null;
  const labelsEmptied = sameLength
    ? before.anchors.filter((a, i) => a.text && !after.anchors[i].text).length
    : null;
  const labelsChanged = sameLength
    ? before.anchors.filter((a, i) => a.text && after.anchors[i].text && a.text !== after.anchors[i].text).length
    : null;

  return {
    anchors_before: before.anchors.length,
    anchors_after: after.anchors.length,
    hrefs_changed: hrefsChanged,
    // §5.4 defect signature: the element survives, its label does not
    anchor_labels_emptied: labelsEmptied,
    // not a defect: a prose label being simplified is the feature
    anchor_labels_rewritten: labelsChanged,
    images_before: before.images.length,
    images_after: after.images.length,
    images_changed: before.images.length === after.images.length
      ? before.images.filter((s, i) => s !== after.images[i]).length
      : null,
    sups_before: before.sups,
    sups_after: after.sups,
    tags_lost: tagsLost,
    tags_gained: tagsGained,
    text_chars_before: before.text_chars,
    text_chars_after: after.text_chars,
  };
}

// one page, one cut

const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const round = (x, places = 3) => (x == null ? null : Number(x.toFixed(places)));
const share = (n, d) => (d ? round(n / d) : null);
const median = (xs) => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
};

// "A sentence, not a label": three times the send threshold, so nothing near the
// boundary counts as a false negative.
const PROSE_WORDS = 12;

/**
 * Prose skipped (candidate false negatives) and boilerplate sent.
 *
 * The sent side is only measurable for the sentence cut: document mode tallies a section
 * against its first unit object, not an element (content.js simplifySection, to avoid
 * resolving an element twice), so sent-side counts would read as zero and are null
 * instead. The skipped side is element-keyed in both cuts; `rewritten` is reported for
 * document mode.
 */
function analyseOutcomes(outcomes, cutName) {
  const skipped = outcomes.filter((o) => o.status === "skipped");
  const sent = outcomes.filter((o) => o.status === "sent");
  const sentIsMeasurable = cutName === "sentence";
  const proseSkipped = skipped.filter((o) => o.in_scope && !o.in_boilerplate && o.words >= PROSE_WORDS);
  const byReason = {};
  for (const o of proseSkipped) {
    (byReason[o.reason ?? "no-simplifiable-chunk"] ||= []).push(o);
  }
  // not automatically wrong (a <footer> can hold a sentence); samples kept for reading
  const boilerplateSent = sent.filter((o) => o.in_boilerplate);
  // Counterfactual: what the sentence cut would stop sending if it used document mode's
  // content scope. Differs from boilerplateSent both ways (a <nav> can sit inside <main>).
  const sentOutsideScope = sent.filter((o) => !o.in_scope);
  return {
    registered_items: outcomes.length,
    sent_side_measurable: sentIsMeasurable,
    sent_outside_content_scope: sentIsMeasurable ? sentOutsideScope.length : null,
    sent_outside_content_scope_words: sentIsMeasurable ? sum(sentOutsideScope.map((o) => o.words)) : null,
    // prose that scoping would stop simplifying: the cost side of that change
    sent_outside_scope_prose: sentIsMeasurable
      ? sentOutsideScope.filter((o) => o.words >= PROSE_WORDS).length
      : null,
    sent_outside_scope_samples: sentIsMeasurable
      ? sentOutsideScope
          .filter((o) => o.words >= PROSE_WORDS)
          .slice(0, 5)
          .map((o) => ({ tag: o.tag, words: o.words, changed: o.changed, text: o.text }))
      : null,
    elements_rewritten: outcomes.filter((o) => o.rewritten).length,
    prose_skipped_in_scope: proseSkipped.length,
    prose_skipped_by_reason: Object.fromEntries(
      Object.entries(byReason).map(([k, v]) => [k, v.length])
    ),
    prose_skipped_samples: Object.fromEntries(
      Object.entries(byReason).map(([k, v]) => [
        k,
        v.slice(0, 3).map((o) => ({ tag: o.tag, words: o.words, text: o.text })),
      ])
    ),
    boilerplate_sent: sentIsMeasurable ? boilerplateSent.length : null,
    boilerplate_sent_samples: sentIsMeasurable ? boilerplateSent.slice(0, 5).map((o) => ({
      tag: o.tag,
      words: o.words,
      changed: o.changed,
      text: o.text,
    })) : null,
    words_in_scope_sent: sentIsMeasurable ? sum(sent.filter((o) => o.in_scope).map((o) => o.words)) : null,
    words_in_scope_skipped: sum(skipped.filter((o) => o.in_scope).map((o) => o.words)),
  };
}

async function auditCut({ site, cutName, args }) {
  const cut = CUTS[cutName];
  const requests = [];
  const { context, run, dom } = loadExtension(site.html, {
    pageUrl: site.final_url || site.url,
    backend: args.backend,
    model: cut.model,
    audience: args.audience,
    requests,
  });

  const health = await (await fetch(`${args.backend}/health`)).json();
  if (!health.models_loaded?.includes(cut.model)) {
    throw new Error(`model '${cut.model}' not loaded on the backend`);
  }
  const budgets = health.document_max_tokens;
  const maxTokens =
    (typeof budgets === "object" && budgets !== null ? budgets[cut.model] : budgets) ??
    health.document_max_tokens_default ??
    512;
  context.__health = { ok: true, model: cut.model, granularity: cut.granularity, documentMaxTokens: maxTokens };

  const before = JSON.parse(run(SNAPSHOT_EXPR));
  const pristineHtml = run(`document.body.innerHTML`);
  const pristineText = run(`document.body.textContent`);
  const pristineAttrs = JSON.parse(run(ATTR_NAMES_EXPR));

  const startedAt = performance.now();
  run(`beginSession(); requestSelectionInfo();`);
  await run(`simplifyPage(${JSON.stringify(cut.granularity)}, ${maxTokens})`);
  const wallMs = performance.now() - startedAt;

  const counters = run(
    `({ found: foundTotal, skipped: skippedCount, sent: sentCount, changed: changedCount,
        unwrittenUnits: unwrittenUnitCount })`
  );
  const skips = JSON.parse(run(`JSON.stringify(skipBreakdown())`));
  const outcomes = JSON.parse(run(OUTCOMES_EXPR));
  const after = JSON.parse(run(SNAPSHOT_EXPR));
  // must run before the revert destroys the evidence
  const anchorLoss = JSON.parse(run(anchorLossExpr(before.anchors)));
  const scopeSelector = run(`(() => {
    const scope = findContentScope();
    if (!scope) return null;
    for (const sel of CONTENT_SCOPE_SELECTORS) if (scope.matches(sel)) return sel;
    return scope.tagName.toLowerCase();
  })()`);

  // Round trip through revertPage(), at three strictnesses: byte-identical, identical
  // ignoring extension attributes (a leftover style="" is a blemish, not a lost
  // paragraph), and text-identical.
  run(`stopObserver(); revertPage();`);
  const revertedHtml = run(`document.body.innerHTML`);
  const revertedText = run(`document.body.textContent`);
  const revertedAttrs = JSON.parse(run(ATTR_NAMES_EXPR));
  const stripAttrs = (html) =>
    html
      .replace(/ data-original-(html|text|placeholder)="[^"]*"/g, "")
      .replace(/ style="[^"]*"/g, "")
      .replace(/ title="[^"]*"/g, "")
      .replace(/ class="simplified"/g, "")
      .replace(/(class="[^"]*?)\s*simplified\s*([^"]*")/g, "$1$2");
  const revert = {
    byte_identical: revertedHtml === pristineHtml,
    identical_ignoring_extension_attributes: stripAttrs(revertedHtml) === stripAttrs(pristineHtml),
    text_identical: revertedText === pristineText,
    html_chars_before: pristineHtml.length,
    html_chars_after: revertedHtml.length,
    attributes: diffAttributes(pristineAttrs, revertedAttrs),
  };
  run(`endSession();`);

  // content.js's intervals keep node's event loop alive; unclosed windows leak and
  // stopped the script from exiting.
  dom.window.close();

  const changedRequests = requests.filter((r) => r.changed);
  return {
    cut: cutName,
    model: cut.model,
    granularity: cut.granularity,
    content_scope: scopeSelector,
    // Smoke signal only, not comparable to §6.4: the backend caches on (model, text);
    // see requests.served_from_cache.
    wall_ms: Math.round(wallMs),
    // the extension's own tally, i.e. what its notice would show
    units: counters,
    coverage: {
      skipped_share: share(counters.skipped, counters.found),
      sent_share: share(counters.sent, counters.found),
      changed_share_of_sent: share(counters.changed, counters.sent),
      changed_share_of_found: share(counters.changed, counters.found),
    },
    requests: {
      total: requests.length,
      served_from_cache: requests.filter((r) => r.cached).length,
      changed: changedRequests.length,
      unchanged: requests.length - changedRequests.length,
      // See fallbackReason above. The share of guarded *sentences* is not derivable
      // (per-sentence model_result is not returned); sentences_total bounds it.
      requests_with_a_guarded_sentence: requests.filter((r) => r.fallbackReason).length,
      sentences_total: sum(requests.map((r) => r.sentences ?? 1)),
      guard_reasons: requests.reduce((acc, r) => {
        if (r.fallbackReason) acc[r.fallbackReason] = (acc[r.fallbackReason] || 0) + 1;
        return acc;
      }, {}),
      words_median: median(requests.map((r) => r.words)),
      words_max: requests.length ? Math.max(...requests.map((r) => r.words)) : null,
      compression_median: median(
        changedRequests.filter((r) => r.chars).map((r) => r.output.length / r.chars)
      ),
    },
    skips,
    selection: analyseOutcomes(outcomes, cutName),
    dom: diffSnapshots(before, after),
    anchor_loss: anchorLoss,
    revert,
    samples: {
      changed: changedRequests.slice(0, 4).map((r) => ({ in: r.input.slice(0, 240), out: r.output.slice(0, 240) })),
      unchanged: requests.filter((r) => !r.changed).slice(0, 4).map((r) => ({ in: r.input.slice(0, 240) })),
    },
    // written to the _items/_units JSONL files for the labelling pass
    __outcomes: outcomes,
    __requests: requests,
  };
}

function gitCommit() {
  try {
    return execFileSync("git", ["rev-parse", "--short", "HEAD"], { cwd: HERE, encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

async function main() {
  const args = parseArgs(process.argv);
  const siteNames = args.sites.split(",").map((s) => s.trim()).filter(Boolean);
  const cutNames = args.cuts.split(",").map((s) => s.trim()).filter(Boolean);
  for (const c of cutNames) if (!CUTS[c]) throw new Error(`unknown cut '${c}'`);

  const startedAt = new Date().toISOString();
  const health = await (await fetch(`${args.backend}/health`)).json();
  const rows = [];
  // one line per registered leaf, and one per request, across every page and cut
  const items = [];
  const units = [];
  for (const name of siteNames) {
    const site = await loadSite(name, { refetch: args.refetch });
    for (const cutName of cutNames) {
      process.stderr.write(`auditing ${name} / ${cutName}…\n`);
      let row;
      try {
        row = await auditCut({ site, cutName, args });
      } catch (e) {
        process.stderr.write(`  FAILED: ${e.message}\n`);
        row = { cut: cutName, model: CUTS[cutName].model, error: String(e.message || e) };
      }
      const { html, ...pageMeta } = site;
      const { __outcomes, __requests, ...summary } = row;
      if (__outcomes) {
        for (const o of __outcomes) {
          items.push({ site: site.name, category: site.category, cut: cutName, ...o });
        }
      }
      if (__requests) {
        for (const r of __requests) {
          units.push({
            site: site.name,
            category: site.category,
            cut: cutName,
            words: r.words,
            changed: r.changed,
            cached: r.cached,
            guard: r.fallbackReason,
            input: r.input,
            output: r.output,
          });
        }
      }
      rows.push({ site: pageMeta, ...summary });
      if (!summary.error) {
        process.stderr.write(
          `  found ${row.units.found}, skipped ${row.units.skipped} ` +
            `(${Math.round((row.coverage.skipped_share ?? 0) * 100)}%), ` +
            `sent ${row.units.sent}, changed ${row.units.changed}; ` +
            `prose skipped in scope ${row.selection.prose_skipped_in_scope}; ` +
            `labels emptied ${row.dom.anchor_labels_emptied} ` +
            `(${row.anchor_loss.misplaced_by_write_back ?? "?"} misplaced, ` +
            `${row.anchor_loss.deleted_by_model ?? "?"} model-deleted); ` +
            `revert ${row.revert.identical_ignoring_extension_attributes ? "clean" : "DIRTY"}\n`
        );
      }
    }
  }

  const report = {
    run: {
      script: "audit_sites.mjs",
      script_version: "1.0",
      started_at: startedAt,
      finished_at: new Date().toISOString(),
      git_commit: gitCommit(),
      node: process.version,
      host: { platform: `${os.platform()} ${os.release()}`, cores: os.cpus().length },
    },
    backend: {
      origin: args.backend,
      models_loaded: health.models_loaded,
      model_names: health.model_names,
      seq2seq_device: health.seq2seq_device,
      granularities: health.granularities,
    },
    client: {
      source: `${CONTENT_SCRIPTS.join(", ")} (manifest.json's content_scripts), loaded under jsdom`,
      audience: args.audience,
    },
    thresholds: { prose_words: PROSE_WORDS, min_words_to_simplify: 4 },
    rows,
    notes: [
      "Selection is done by the extension's own functions under jsdom; only chrome.runtime is stubbed. Every content script manifest.json injects is loaded, in manifest order.",
      "Skip reasons are reported by content.js's own skipBreakdown(), not recomputed here.",
      "jsdom does not run the page's JavaScript: these are the DOMs these sites ship, not post-hydration DOMs. The two SPA entries are server-rendered apps chosen for that reason; a client-only rendered app is outside this harness.",
      "No layout is computed, so visual regressions (overflow, clipping, reflow) are not visible to this audit.",
      "Pages are cached under scratch/sites/ with fetch time and a sha256 prefix; a news article or a shop page can differ on a later fetch.",
      "An emptied anchor label has two causes with different owners: the model deleted the linked phrase (content loss), or the write-back misplaced words it kept (a DOM-preservation defect). anchor_loss splits them; dom.anchor_labels_emptied is their sum.",
      "Sent-side selection counts (boilerplate_sent, sent_outside_content_scope, words_in_scope_sent) are null for the document cut by construction: it resolves a section against its first unit object rather than an element, so no element-keyed 'sent' status exists for it. The skipped side is element-keyed and measured in both cuts.",
      "'sent_outside_content_scope' is the counterfactual for giving the sentence cut findContentScope(): the items it would stop sending. It differs from boilerplate_sent in both directions, since a nav or aside can sit inside main.",
      "'prose_skipped_in_scope' counts leaf candidates of >=12 words inside the page's own content scope and outside nav/footer/header/aside/form -- candidate false negatives, to be read rather than trusted.",
      "revert is checked by comparing document.body.innerHTML before the run against after revertPage(), at three strictnesses, plus a per-element attribute-set comparison.",
      "wall_ms is not a latency figure: the backend caches on (model, text). Read requests.served_from_cache before comparing it to anything.",
    ],
  };

  fs.mkdirSync(args.out, { recursive: true });
  const stamp = startedAt.replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
  const file = path.join(args.out, `site_audit_${stamp}.json`);
  fs.writeFileSync(file, JSON.stringify(report, null, 2));
  // JSONL: read line by line, sampled and annotated by the labelling pass
  const itemsFile = path.join(args.out, `site_audit_${stamp}_items.jsonl`);
  const unitsFile = path.join(args.out, `site_audit_${stamp}_units.jsonl`);
  fs.writeFileSync(itemsFile, items.map((o) => JSON.stringify(o)).join("\n") + "\n");
  fs.writeFileSync(unitsFile, units.map((o) => JSON.stringify(o)).join("\n") + "\n");
  process.stderr.write(
    `\nWrote ${path.relative(HERE, file)}\n` +
      `      ${path.relative(HERE, itemsFile)} (${items.length} items)\n` +
      `      ${path.relative(HERE, unitsFile)} (${units.length} requests)\n`
  );
}

main()
  // explicit exit: a closed jsdom window can still keep the event loop alive
  .then(() => process.exit(0))
  .catch((e) => {
    process.stderr.write(`${e.stack || e}\n`);
    process.exit(1);
  });
