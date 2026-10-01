// shared/analysis.js - readers for stored runs (whatever build wrote them), shared
// number formatters, the page summary lines, and reference-free text metrics.
//
// Shared by History and Reports so a report's evaluation is the same computation as
// History's "Run advanced analysis".
//
// Plain <script>, not an ES module; load before the page script. Depends on
// shared/model-labels.js for method, granularity and audience wording.

// Matches background.js's HISTORY_STORAGE_KEY. Read by History and by Reports (to
// follow a run still writing to it).
const HISTORY_STORAGE_KEY = "simplifyHistory";

// --- reading a stored page -----------------------------------------------
// Pages recorded before the log restructure carry `original`/`simplified` per entry
// and no session-level fields. They are deliberately not migrated on read; read
// through these helpers so an old page falls back instead of breaking.

function entryInput(entry) {
  return typeof entry.input === "string" ? entry.input : entry.original || "";
}

function entryOutput(entry) {
  return typeof entry.output === "string" ? entry.output : entry.simplified || "";
}

// What the model returned; differs from the output only where a guard rejected it.
// Older entries didn't record it, so fall back to the served text.
function entryModelResult(entry) {
  return typeof entry.modelResult === "string" ? entry.modelResult : entryOutput(entry);
}

// Per-run counts from the session record, else derived from the entries. Old pages
// never recorded skips, so skippedItems is null ("not recorded"), not 0.
function pageStats(page) {
  if (page.stats) return page.stats;
  const entries = page.entries || [];
  const simplified = entries.filter((e) => e.changed).length;
  return {
    totalItems: entries.length,
    skippedItems: null,
    processedItems: entries.length,
    simplifiedItems: simplified,
    unchangedItems: entries.length - simplified,
    cachedResponses: entries.filter((e) => e.cached).length,
  };
}

function formatTimestamp(ms) {
  return new Date(ms).toLocaleString();
}

function truncateText(value, maxLength = 90) {
  if (!value) return "";
  if (value.length <= maxLength) return value;
  return `${value.slice(0, maxLength - 1).trimEnd()}…`;
}

// Latency ranges from ms (cache hits) to seconds (7B generation); keep ms below 1s
// rather than rounding to "0.0s". Same rule as content.js's formatLatency.
function formatLatency(ms) {
  if (ms == null || !isFinite(ms)) return "—";
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
}

function formatDuration(ms) {
  if (ms == null || !isFinite(ms)) return "—";
  const seconds = ms / 1000;
  return seconds < 10 ? `${seconds.toFixed(1)}s` : `${Math.round(seconds)}s`;
}

// Feedback for one-step actions (e.g. copying a row) instead of a blocking alert().
// Same notice box as content.js's #simplify-notice.
const NOTICE_MS = 2500;
const NOTICE_ERROR_MS = 5000;
let noticeTimer = null;

function showNotice(message, { error = false } = {}) {
  const el = document.getElementById("page-notice");
  if (!el) return;
  el.textContent = message;
  el.classList.toggle("page-notice-error", error);
  el.classList.remove("page-notice-hidden");
  if (noticeTimer) clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => {
    el.classList.add("page-notice-hidden");
    noticeTimer = null;
  }, error ? NOTICE_ERROR_MS : NOTICE_MS);
}

function sanitizeFilename(s) {
  return s.replace(/[^a-z0-9]+/gi, "-").replace(/^-+|-+$/g, "").slice(0, 60) || "page";
}

function downloadJSON(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// --- scoring -----------------------------------------------------------
// Each column takes (original, simplified) and returns a display value, or null if
// it doesn't apply. The table and the per-page aggregate pick up new entries
// automatically.

function countSyllables(word) {
  const w = word.toLowerCase().replace(/[^a-z]/g, "");
  if (!w) return 0;
  const groups = w.match(/[aeiouy]+/g) || [];
  let count = groups.length;
  if (w.endsWith("e") && count > 1) count -= 1; // silent e
  return Math.max(count, 1);
}

// Words ending in a period without ending a sentence. Dotted forms (a.m., U.S.A.)
// and single initials (J. Smith) are matched by pattern below instead.
const ABBREVIATIONS = new Set([
  "mr", "mrs", "ms", "dr", "prof", "rev", "hon", "st", "jr", "sr", "mt",
  "vs", "etc", "est", "approx", "dept", "univ", "inc", "ltd", "co", "corp",
  "fig", "no", "nos", "vol", "pp", "ed", "eds", "al", "cf", "ca", "min", "max",
  "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
  "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun",
]);

// "a.m", "U.S", "e.g": the token left once the final period is taken as the terminator.
const DOTTED_ABBREVIATION = /^(?:[A-Za-z]\.)+[A-Za-z]$/;
// Abbreviations whose period often doubles as the full stop ("... at 6 a.m. Collections
// finish at noon."). Only after these does a capital start a new sentence; after the
// rest ("Dr. Patel", "the U.S. Government") it is usually a name.
const SENTENCE_FINAL_ABBREVIATIONS = new Set(["a.m", "p.m", "etc"]);
// terminator, optional closing quote or bracket, then whitespace.
const SENTENCE_BOUNDARY = /([.!?]+["'\u2019\u201d)\]]?)(\s+)/g;
// a new sentence starts with a capital, a digit, or an opening quote.
const SENTENCE_START = /^["'\u2018\u201c(\[]?[A-Z0-9]/;

function isSentenceBoundary(before, terminator, after) {
  if (!SENTENCE_START.test(after)) return false;
  if (terminator.includes("!") || terminator.includes("?")) return true;

  // the word the period is attached to ("a.m" in "6 a.m.", "Dr" in "Dr. Smith");
  // the trailing period belongs to the terminator.
  const token = (before.slice(0, before.length - terminator.length).match(/[A-Za-z0-9'\u2019.]+$/) || [""])[0];
  if (!token) return true;
  const bare = token.toLowerCase().replace(/\.+$/, "");
  if (DOTTED_ABBREVIATION.test(token)) return SENTENCE_FINAL_ABBREVIATIONS.has(bare) && /^[A-Z]/.test(after);
  if (/^[A-Za-z]$/.test(token)) return false; // an initial
  if (/^\d+$/.test(token)) return false; // list numbering: "6. Put the bins out"
  if (!ABBREVIATIONS.has(bare)) return true;
  return SENTENCE_FINAL_ABBREVIATIONS.has(bare) && /^[A-Z]/.test(after);
}

// Lightweight stand-in for the backend's pysbd (main.py's split_sentences) that
// handles what a bare /[.!?]/ split gets wrong, e.g. "at approximately 6 a.m. on
// Tuesdays." as three sentences. Every words-per-sentence figure, readability
// formulas included, depends on this count.
function splitSentences(text) {
  const t = (text || "").trim();
  if (!t) return [];

  const sentences = [];
  let start = 0;
  let match;
  SENTENCE_BOUNDARY.lastIndex = 0;
  while ((match = SENTENCE_BOUNDARY.exec(t)) !== null) {
    const end = match.index + match[1].length;
    const rest = t.slice(end + match[2].length);
    if (!isSentenceBoundary(t.slice(start, end), match[1], rest)) continue;
    sentences.push(t.slice(start, end).trim());
    start = end + match[2].length;
  }

  const tail = t.slice(start).trim();
  if (tail) sentences.push(tail);
  // a fragment with no terminal punctuation is still one unit of text, never zero
  return sentences.length ? sentences : [t];
}

function countSentences(text) {
  return splitSentences(text).length;
}

// Flesch-Kincaid Grade Level. Only a rough estimate for short phrases (the typical
// case): an unpunctuated fragment counts as one sentence.
function fleschKincaidGrade(text) {
  const words = text.match(/[A-Za-z']+/g) || [];
  if (words.length === 0) return null;
  const sentenceCount = Math.max(countSentences(text), 1);
  const syllableCount = words.reduce((sum, w) => sum + countSyllables(w), 0);
  return 0.39 * (words.length / sentenceCount) + 11.8 * (syllableCount / words.length) - 15.59;
}

// signed percentage; zero gets no sign ("-0%" would read as a shortening).
function formatPercentDelta(pct) {
  const rounded = Math.round(pct);
  if (rounded === 0) return "0%";
  return `${rounded > 0 ? "-" : "+"}${Math.abs(rounded)}%`;
}

function percentWordsShortened(original, simplified) {
  const origWords = (original.match(/[A-Za-z']+/g) || []).length;
  const simpWords = (simplified.match(/[A-Za-z']+/g) || []).length;
  if (origWords === 0) return null;
  return Math.round((1 - simpWords / origWords) * 100);
}

// `label` heads the column; `aggregateLabel` is the fuller name on the summary line.
const SCORE_COLUMNS = [
  {
    label: "Readability",
    aggregateLabel: "Readability (FK grade)",
    tooltip:
      "Flesch-Kincaid Grade Level: the US school grade needed to read the text, " +
      "estimated from words per sentence and syllables per word. Lower is simpler.",
    perSentence(input, output) {
      const before = fleschKincaidGrade(input);
      const after = fleschKincaidGrade(output);
      if (before == null || after == null) return null;
      return { before, after, text: `${before.toFixed(1)} → ${after.toFixed(1)}` };
    },
    aggregate(values) {
      const avg = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
      return `avg ${avg(values.map((v) => v.before)).toFixed(1)} → ${avg(values.map((v) => v.after)).toFixed(1)}`;
    },
  },
  {
    label: "Words",
    aggregateLabel: "Words",
    tooltip: "How much shorter the output is than the input, by word count.",
    perSentence(input, output) {
      const pct = percentWordsShortened(input, output);
      if (pct == null) return null;
      return { pct, text: formatPercentDelta(pct) };
    },
    aggregate(values) {
      const avgPct = values.reduce((a, v) => a + v.pct, 0) / values.length;
      return `avg ${formatPercentDelta(avgPct)}`;
    },
  },
];

// --- page summary header -------------------------------------------------
// Lines: which page, what ran, outcome, speed, readability change.

// Status dot states, plus a fallback for pages recorded before runs had a status.
// The tooltip carries the explanation; the colour alone is ambiguous.
const STATUS_INDICATORS = {
  completed: {
    className: "status-dot-ok",
    tooltip: "Completed — every request came back with an answer the guards accepted.",
  },
  completed_with_warnings: {
    className: "status-dot-warn",
    tooltip:
      "Completed with fallbacks — at least one of the model's answers was rejected and " +
      "the original text kept in its place. The rows say which, and why.",
  },
  failed: {
    className: "status-dot-fail",
    tooltip: "Failed — the run stopped before it finished, so the page is only partly covered.",
  },
  // Not a failure: the user ended the run. Both leave the page partly covered, but
  // only "failed" needs investigating.
  stopped: {
    className: "status-dot-stopped",
    tooltip:
      "Stopped — you ended this run before it finished, so the counts describe the part " +
      "of the page it reached rather than all of it.",
  },
};

const UNKNOWN_STATUS_INDICATOR = {
  className: "status-dot-unknown",
  tooltip: "This page was recorded before the log kept a run status.",
};

function buildStatusDot(status) {
  const spec = STATUS_INDICATORS[status] || UNKNOWN_STATUS_INDICATOR;
  const dot = document.createElement("span");
  dot.className = `status-dot ${spec.className}`;
  dot.title = spec.tooltip;
  // expose the tooltip to screen readers too
  dot.setAttribute("role", "img");
  dot.setAttribute("aria-label", spec.tooltip);
  return dot;
}

// "Fine-tuned seq2seq · <id> · Sentence by sentence · Non-native speakers".
// Only the id (a copyable string) is set in monospace.
function buildSelectionLine(page) {
  const line = document.createElement("div");
  line.className = "page-aggregate";
  if (!page.method && !page.modelId && !page.granularity) {
    line.textContent = "Model and method weren't recorded for this page.";
    return line;
  }

  const parts = [];
  if (page.method) parts.push(document.createTextNode(methodLabel(page.method)));
  if (page.modelId) {
    const id = document.createElement("code");
    id.className = "page-model-id";
    id.textContent = page.modelId;
    parts.push(id);
  }
  if (page.granularity) parts.push(document.createTextNode(granularityLabel(page.granularity)));
  // recorded only for the prompted models, which are the only ones that act on it
  if (page.audience) parts.push(document.createTextNode(audienceDisplayName(page.audience)));

  parts.forEach((part, i) => {
    if (i > 0) line.appendChild(document.createTextNode(" · "));
    line.appendChild(part);
  });
  return line;
}

// "19 simplified · 4 unchanged · 70 items skipped (e.g. too short)". Skipped count is
// omitted when not recorded rather than shown as 0.
function buildOutcomeLine(stats) {
  const line = document.createElement("div");
  line.className = "page-aggregate";
  const parts = [`${stats.simplifiedItems} simplified`, `${stats.unchangedItems} unchanged`];
  if (stats.skippedItems != null && stats.skippedItems > 0) {
    parts.push(`${stats.skippedItems} ${stats.skippedItems === 1 ? "item" : "items"} skipped (e.g. too short)`);
  }
  // Document runs only: a section is one item however many paragraphs it spans, so
  // "19 simplified" can hide paragraphs that were sent but left as they were.
  // Omitted when absent (sentence runs don't count it).
  if (stats.unwrittenUnits != null && stats.unwrittenUnits > 0) {
    parts.push(
      `${stats.unwrittenUnits} ${stats.unwrittenUnits === 1 ? "paragraph" : "paragraphs"} ` +
        `left as-is (outside the answer)`
    );
  }
  line.textContent = parts.join(" · ");
  return line;
}

// "Completed in 4.7s · 4.9 items/s · 1.4s/request (3 reused from cache)".
// Throughput and latency only when there is something to divide by.
function buildSpeedLine(page, stats) {
  const timings = (page.entries || [])
    .map((e) => e.requestTimeMs)
    .filter((ms) => typeof ms === "number");
  if (page.totalTimeMs == null && timings.length === 0) return null;

  const line = document.createElement("div");
  line.className = "page-aggregate";
  const parts = [];
  if (page.totalTimeMs != null) {
    parts.push(
      `${page.status === "failed" ? "Stopped after" : "Completed in"} ${formatDuration(page.totalTimeMs)}`
    );
    // items, not requests: one item can cost several requests when a leaf splits at <br>s.
    if (page.totalTimeMs > 0 && stats.processedItems > 0) {
      parts.push(`${(stats.processedItems / (page.totalTimeMs / 1000)).toFixed(1)} items/s`);
    }
  }
  if (timings.length > 0) {
    const mean = timings.reduce((a, b) => a + b, 0) / timings.length;
    const cached = stats.cachedResponses;
    parts.push(
      `${formatLatency(mean)}/request${cached > 0 ? ` (${cached} reused from cache)` : ""}`
    );
  }
  line.textContent = parts.join(" · ");
  return line;
}

// "Readability (FK grade): avg 15.9 → 9.7 · Words: avg -20%", over changed rows only;
// unchanged rows would pull the average toward "no change" by how simple the page
// already was, which says nothing about the model.
function buildMetricsLine(entries) {
  const changed = entries.filter((e) => e.changed);
  const line = document.createElement("div");
  line.className = "page-aggregate page-aggregate-metrics";
  if (changed.length === 0) return line;

  const metrics = [];
  SCORE_COLUMNS.forEach((col) => {
    const values = changed.map((e) => col.perSentence(entryInput(e), entryOutput(e))).filter(Boolean);
    if (values.length > 0) metrics.push(`${col.aggregateLabel}: ${col.aggregate(values)}`);
  });
  line.textContent = metrics.join(" · ");
  return line;
}

function buildPageSummary(page) {
  const stats = pageStats(page);
  const entries = page.entries || [];

  const heading = document.createElement("h2");
  heading.appendChild(buildStatusDot(page.status));
  heading.appendChild(document.createTextNode(page.title || page.url));

  const meta = document.createElement("div");
  meta.className = "page-meta";
  meta.textContent = `${truncateText(page.url, 90)} · ${formatTimestamp(page.timestamp)}`;

  const rows = [heading, meta, buildSelectionLine(page), buildOutcomeLine(stats)];
  const speed = buildSpeedLine(page, stats);
  if (speed) rows.push(speed);
  rows.push(buildMetricsLine(entries));
  return rows;
}

// The entries an Analyze modal covers: one row, or every row on a page.
function analyzeEntries(pageOrEntry) {
  const entries = Array.isArray(pageOrEntry) ? pageOrEntry : (pageOrEntry.entries || [pageOrEntry]);
  return entries
    .filter((entry) => entry && (entry.input != null || entry.original != null))
    .map((entry) => ({ ...entry, input: entryInput(entry), output: entryOutput(entry) }));
}

// --- advanced analysis ---------------------------------------------------
// Reference-free only: history has no human reference simplification, so SARI,
// BERTScore and LENS stay in the offline evaluation scripts. Computed here:
// readability formulas, their structural counts, and a token-level rewriting
// breakdown.
//
// Counts are summed over all entries and the formulas applied once to the totals
// (corpus-level, as in the sentence-level evaluation), so many one-word nav
// fragments can't dominate by row count.

function textStats(text) {
  const words = text.match(/[A-Za-z']+/g) || [];
  const syllables = words.map(countSyllables);
  return {
    words: words.length,
    sentences: words.length ? Math.max(countSentences(text), 1) : 0,
    syllables: syllables.reduce((a, b) => a + b, 0),
    polysyllables: syllables.filter((s) => s >= 3).length, // Fog/SMOG's "complex words"
    letters: (text.match(/[A-Za-z]/g) || []).length,
  };
}

function sumStats(list) {
  return list.reduce(
    (acc, s) => ({
      words: acc.words + s.words,
      sentences: acc.sentences + s.sentences,
      syllables: acc.syllables + s.syllables,
      polysyllables: acc.polysyllables + s.polysyllables,
      letters: acc.letters + s.letters,
    }),
    { words: 0, sentences: 0, syllables: 0, polysyllables: 0, letters: 0 }
  );
}

// Six standard formulas from the same five counts. Grade-level scores (FKGL, Fog,
// SMOG, Coleman-Liau, ARI) fall as text gets easier; Reading Ease rises.
function readabilityScores(s) {
  if (!s.words || !s.sentences) return null;
  const wordsPerSentence = s.words / s.sentences;
  const syllablesPerWord = s.syllables / s.words;
  const lettersPerWord = s.letters / s.words;
  const polyShare = s.polysyllables / s.words;
  return {
    fkgl: 0.39 * wordsPerSentence + 11.8 * syllablesPerWord - 15.59,
    readingEase: 206.835 - 1.015 * wordsPerSentence - 84.6 * syllablesPerWord,
    fog: 0.4 * (wordsPerSentence + 100 * polyShare),
    smog: 1.043 * Math.sqrt(s.polysyllables * (30 / s.sentences)) + 3.1291,
    colemanLiau: 0.0588 * lettersPerWord * 100 - 0.296 * (s.sentences / s.words) * 100 - 15.8,
    ari: 4.71 * lettersPerWord + 0.5 * wordsPerSentence - 21.43,
  };
}

function lowerCaseTokens(text) {
  return text.toLowerCase().match(/[a-z0-9']+/g) || [];
}

// Per-entry multiset edit against the entry's own original: kept/deleted as shares of
// the original, added as share of the output. SARI's keep/delete/add decomposition, but
// without references it describes the edit rather than scoring it, so the three are
// reported separately and never combined.
function rewritingStats(entries) {
  let originalTokens = 0;
  let outputTokens = 0;
  let kept = 0;
  let split = 0;
  let merged = 0;
  const originalTypes = new Set();
  const outputTypes = new Set();

  entries.forEach((entry) => {
    const before = lowerCaseTokens(entry.input);
    const after = lowerCaseTokens(entry.output);
    before.forEach((t) => originalTypes.add(t));
    after.forEach((t) => outputTypes.add(t));

    const pool = new Map();
    before.forEach((t) => pool.set(t, (pool.get(t) || 0) + 1));
    after.forEach((t) => {
      const left = pool.get(t) || 0;
      if (left > 0) {
        pool.set(t, left - 1);
        kept += 1;
      }
    });

    originalTokens += before.length;
    outputTokens += after.length;

    const beforeSentences = countSentences(entry.input);
    const afterSentences = countSentences(entry.output);
    if (afterSentences > beforeSentences) split += 1;
    else if (afterSentences < beforeSentences) merged += 1;
  });

  return {
    originalTokens,
    outputTokens,
    kept,
    added: outputTokens - kept,
    split,
    merged,
    originalTTR: originalTokens ? originalTypes.size / originalTokens : null,
    outputTTR: outputTokens ? outputTypes.size / outputTokens : null,
  };
}

function formatNumber(value, decimals) {
  if (value == null || !isFinite(value)) return "n/a";
  return value.toFixed(decimals);
}

// Before/after row plus signed change. `better` is the simplifying direction, used to
// mark the change cell; null for counts, where more or fewer isn't good or bad.
function metricRow(label, before, after, { decimals = 1, unit = "", changeUnit = null, better = null, detail = "" } = {}) {
  const change = before == null || after == null ? null : after - before;
  let verdict = null;
  if (change != null && better && Math.abs(change) >= (decimals === 0 ? 1 : 0.05)) {
    verdict = (change < 0) === (better === "lower") ? "better" : "worse";
  }
  return {
    detail,
    verdict,
    cells: [
      label,
      before == null ? "n/a" : `${formatNumber(before, decimals)}${unit}`,
      after == null ? "n/a" : `${formatNumber(after, decimals)}${unit}`,
      change == null ? "n/a" : `${change >= 0 ? "+" : "-"}${formatNumber(Math.abs(change), decimals)}${changeUnit == null ? unit : changeUnit}`,
    ],
  };
}

function valueRow(label, value, detail = "") {
  return { detail, verdict: null, cells: [label, value] };
}

const BEFORE_AFTER_COLUMNS = ["Metric", "Original", "Simplified", "Change"];

function computeAdvancedMetrics(entries) {
  if (!entries.length) return null;

  const before = sumStats(entries.map((e) => textStats(e.input)));
  const after = sumStats(entries.map((e) => textStats(e.output)));
  const rBefore = readabilityScores(before);
  const rAfter = readabilityScores(after);
  if (!rBefore || !rAfter) return null;

  const score = (key, label, opts) => metricRow(label, rBefore[key], rAfter[key], opts);
  const rewriting = rewritingStats(entries);
  const pct = (n, d) => (d ? (n / d) * 100 : null);

  return [
    {
      title: "Readability",
      columns: BEFORE_AFTER_COLUMNS,
      rows: [
        score("fkgl", "Flesch-Kincaid grade", { better: "lower", detail: "US school grade needed to read the text. Over the totals, so it won't match the quick stat above, which averages the rows." }),
        score("readingEase", "Flesch reading ease", { better: "higher", detail: "0-100; higher is easier" }),
        score("fog", "Gunning fog index", { better: "lower", detail: "Sentence length plus the share of 3+ syllable words" }),
        score("smog", "SMOG grade", { better: "lower", detail: "Grade from the count of 3+ syllable words" }),
        score("colemanLiau", "Coleman-Liau index", { better: "lower", detail: "Grade from letters per word, not syllables" }),
        score("ari", "Automated readability index", { better: "lower", detail: "Grade from characters per word" }),
      ],
    },
    {
      title: "Structure",
      columns: BEFORE_AFTER_COLUMNS,
      rows: [
        metricRow("Sentences", before.sentences, after.sentences, { decimals: 0, detail: "Abbreviation-aware count: \"6 a.m.\" is not a sentence boundary" }),
        metricRow("Words", before.words, after.words, { decimals: 0 }),
        metricRow("Words per sentence", before.words / before.sentences, after.words / after.sentences, { better: "lower" }),
        metricRow("Syllables per word", before.syllables / before.words, after.syllables / after.words, { decimals: 2, better: "lower" }),
        metricRow("Characters per word", before.letters / before.words, after.letters / after.words, { decimals: 2, better: "lower" }),
        metricRow("Long words", pct(before.polysyllables, before.words), pct(after.polysyllables, after.words), { unit: "%", changeUnit: " pp", better: "lower", detail: "Share of words with 3 or more syllables" }),
        metricRow("Type-token ratio", rewriting.originalTTR, rewriting.outputTTR, { decimals: 2, detail: "Distinct words over total words; a rough vocabulary spread" }),
      ],
    },
    {
      title: "Rewriting",
      columns: ["Metric", "Value"],
      note: "How the output differs from the input, word for word. A description of the edit, not a score.",
      rows: [
        valueRow("Words kept", `${formatNumber(pct(rewriting.kept, rewriting.originalTokens), 1)}%`, "Share of the original's words still present in the output"),
        valueRow("Words dropped", `${formatNumber(pct(rewriting.originalTokens - rewriting.kept, rewriting.originalTokens), 1)}%`, "Share of the original's words the output no longer has"),
        valueRow("Words added", `${formatNumber(pct(rewriting.added, rewriting.outputTokens), 1)}%`, "Share of the output's words that were not in the original"),
        valueRow("Rows split into more sentences", `${rewriting.split} of ${entries.length}`),
        valueRow("Rows merged into fewer sentences", `${rewriting.merged} of ${entries.length}`),
      ],
    },
  ];
}

// --- comparison across columns --------------------------------------------
// All columns scored in one pass, one table per section: the original once, then a
// column per (cut, arm), with the best cell per row marked where direction matters.
// Columns, not arms: a comparison can run both cuts, so one arm can have a section
// column and a leaf-element column (the document-scope question).
//
//  - All columns are scored over the same entries: the intersection of what every
//    column answered (the count is stated). Otherwise a lost request would put the
//    columns on different text.
//  - The original is computed once from those entries, so all columns share one baseline.
//
// Reference-free, as above: a report has no human reference simplification.

// One column's result for one entry, or undefined. Results are keyed by cut, then arm
// (see shared/report-store.js); only this function relies on that.
function comparisonColumnResult(entry, column) {
  const byArm = entry && entry.results && entry.results[column.cut];
  return byArm ? byArm[column.arm.label] : undefined;
}

// Entries every column answered (see above).
function comparableComparisonUnits(entries, columns) {
  return (entries || []).filter((entry) =>
    columns.every((column) => {
      const result = comparisonColumnResult(entry, column);
      return result && typeof result.output === "string";
    })
  );
}

// "A" for a one-cut comparison, "A · sections" for two, where one arm has two columns.
function comparisonColumnLabel(column, cutCount) {
  return cutCount > 1 ? `${column.arm.label} · ${cutShortLabel(column.cut)}` : column.arm.label;
}

function comparisonCell(value, decimals, unit) {
  if (value == null || Number.isNaN(value)) return "\u2014";
  return `${formatNumber(value, decimals)}${unit}`;
}

// Indices into `values` of the winning columns; empty if the row has no direction
// ("Words", "Sentences") or nothing is comparable. All tied columns are marked: the
// same model over both cuts answers identically where a section is one leaf element,
// so ties are common.
function bestComparisonArms(values, better) {
  if (!better) return [];
  const usable = values.filter((value) => value != null && !Number.isNaN(value));
  if (usable.length === 0) return [];
  const bestValue = better === "lower" ? Math.min(...usable) : Math.max(...usable);
  return values.reduce((acc, value, i) => (value === bestValue ? acc.concat(i) : acc), []);
}

// Comparison row: label, the original (if the metric has one), then a cell per column.
// `verdicts` is per cell, since the marked cell can be anywhere in the row.
function comparisonRow(label, original, values, { decimals = 1, unit = "", better = null, detail = "" } = {}) {
  const cells = [label];
  const verdicts = [null];
  if (original !== undefined) {
    cells.push(comparisonCell(original, decimals, unit));
    verdicts.push(null);
  }
  const best = bestComparisonArms(values, better);
  values.forEach((value, i) => {
    cells.push(comparisonCell(value, decimals, unit));
    verdicts.push(best.includes(i) ? "best" : null);
  });
  return { detail, verdict: null, verdicts, cells };
}

function meanLatency(results) {
  const timings = results.map((r) => r && r.requestTimeMs).filter((ms) => typeof ms === "number");
  if (timings.length === 0) return null;
  return timings.reduce((sum, ms) => sum + ms, 0) / timings.length;
}

// `columns` is shared/report-store.js's comparisonColumns(report), passed in so this
// file stays independent of report storage.
function computeComparisonMetrics(columns, entries) {
  if (!columns || columns.length === 0) return null;
  const units = comparableComparisonUnits(entries, columns);
  if (units.length === 0) return null;

  const cutCount = new Set(columns.map((column) => column.cut)).size;
  const headings = columns.map((column) => comparisonColumnLabel(column, cutCount));

  const before = sumStats(units.map((unit) => textStats(unit.input)));
  const rBefore = readabilityScores(before);
  if (!rBefore) return null;

  const scored = columns.map((column) => {
    const results = units.map((unit) => comparisonColumnResult(unit, column));
    const rows = units.map((unit, i) => ({ input: unit.input, output: results[i].output }));
    const after = sumStats(rows.map((row) => textStats(row.output)));
    return { column, results, rows, after, readability: readabilityScores(after), rewriting: rewritingStats(rows) };
  });
  if (scored.some((s) => !s.readability)) return null;

  const columnHeadings = ["Metric"].concat(headings);
  const withOriginal = ["Metric", "Original"].concat(headings);
  const values = (fn) => scored.map(fn);
  const pct = (n, d) => (d ? (n / d) * 100 : null);
  const score = (key, label, opts) =>
    comparisonRow(label, rBefore[key], values((s) => s.readability[key]), opts);

  return [
    {
      title: "Readability",
      columns: withOriginal,
      // stated once, on the first table: which entries were scored, all against one original
      note:
        `Scored over the ${units.length} ${units.length === 1 ? "entry" : "entries"} every model/method answered, ` +
        `out of ${(entries || []).length} on the page — the same text for all of them, each scored against the ` +
        `one original. The best of each row is marked, on every model/method that holds it — two readings of ` +
        `one model answer identically wherever a section is a single element, so a best is often shared. ` +
        `Computed over the totals, not averaged per row.`,
      rows: [
        score("fkgl", "Flesch-Kincaid grade", { better: "lower", detail: "US school grade needed to read the text" }),
        score("readingEase", "Flesch reading ease", { better: "higher", detail: "0-100; higher is easier" }),
        score("fog", "Gunning fog index", { better: "lower", detail: "Sentence length plus the share of 3+ syllable words" }),
        score("smog", "SMOG grade", { better: "lower", detail: "Grade from the count of 3+ syllable words" }),
        score("colemanLiau", "Coleman-Liau index", { better: "lower", detail: "Grade from letters per word, not syllables" }),
        score("ari", "Automated readability index", { better: "lower", detail: "Grade from characters per word" }),
      ],
    },
    {
      title: "Structure",
      columns: withOriginal,
      note:
        "No model/method is marked on the counts: fewer words is not better in itself, and a " +
        "section read whole legitimately loses content that the same text read a leaf " +
        "element at a time cannot.",
      rows: [
        comparisonRow("Sentences", before.sentences, values((s) => s.after.sentences), { decimals: 0 }),
        comparisonRow("Words", before.words, values((s) => s.after.words), { decimals: 0 }),
        comparisonRow("Words per sentence", before.words / before.sentences, values((s) => s.after.words / s.after.sentences), { better: "lower" }),
        comparisonRow("Syllables per word", before.syllables / before.words, values((s) => s.after.syllables / s.after.words), { decimals: 2, better: "lower" }),
        comparisonRow("Characters per word", before.letters / before.words, values((s) => s.after.letters / s.after.words), { decimals: 2, better: "lower" }),
        comparisonRow("Long words", pct(before.polysyllables, before.words), values((s) => pct(s.after.polysyllables, s.after.words)), { unit: "%", better: "lower", detail: "Share of words with 3 or more syllables" }),
        comparisonRow("Type-token ratio", scored[0].rewriting.originalTTR, values((s) => s.rewriting.outputTTR), { decimals: 2, detail: "Distinct words over total words; a rough vocabulary spread" }),
      ],
    },
    {
      title: "Rewriting",
      columns: columnHeadings,
      note: "How far each model/method moved from the same input, word for word. A description of the edit, not a score.",
      rows: [
        comparisonRow("Words kept", undefined, values((s) => pct(s.rewriting.kept, s.rewriting.originalTokens)), { unit: "%", detail: "Share of the original's words still present in the output" }),
        comparisonRow("Words dropped", undefined, values((s) => pct(s.rewriting.originalTokens - s.rewriting.kept, s.rewriting.originalTokens)), { unit: "%", detail: "Share of the original's words the output no longer has" }),
        comparisonRow("Words added", undefined, values((s) => pct(s.rewriting.added, s.rewriting.outputTokens)), { unit: "%", detail: "Share of the output's words that were not in the original" }),
        comparisonRow("Entries split into more sentences", undefined, values((s) => s.rewriting.split), { decimals: 0 }),
        comparisonRow("Entries merged into fewer sentences", undefined, values((s) => s.rewriting.merged), { decimals: 0 }),
      ],
    },
    {
      title: "What each model/method did",
      columns: columnHeadings,
      note:
        "The run itself rather than the text: how much each model/method changed, how often " +
        "a guard rejected what it produced, and what it cost. Times are per entry and " +
        "include cache hits, so one that reused work looks faster because it was — and one " +
        "that sent a whole section as one request is being compared against one that sent " +
        "every leaf element in it.",
      rows: [
        comparisonRow("Entries changed", undefined, values((s) => s.results.filter((r) => r.changed).length), { decimals: 0, detail: `of ${units.length}` }),
        comparisonRow("Entries left unchanged", undefined, values((s) => s.results.filter((r) => !r.changed).length), { decimals: 0, better: "lower" }),
        comparisonRow("Output rejected by a guard", undefined, values((s) => s.results.filter((r) => r.fallbackReason).length), { decimals: 0, better: "lower", detail: "The original was served instead; fallbackReason says which guard" }),
        comparisonRow("Reused from cache", undefined, values((s) => s.results.filter((r) => r.cached).length), { decimals: 0 }),
        comparisonRow("Requests sent", undefined, values((s) => (s.column.run.stats && s.column.run.stats.requests) || null), { decimals: 0, detail: "One per entry, or one per unit inside it" }),
        comparisonRow("Mean time per entry", undefined, values((s) => meanLatency(s.results)), { decimals: 0, unit: " ms", better: "lower" }),
        comparisonRow("Total time", undefined, values((s) => (s.column.run.totalTimeMs == null ? null : s.column.run.totalTimeMs / 1000)), { decimals: 1, unit: " s", better: "lower" }),
      ],
    },
  ];
}

function buildMetricsTable(section) {
  const table = document.createElement("table");
  table.className = "analyze-table";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  section.columns.forEach((label, i) => {
    const th = document.createElement("th");
    // Width floor on an inner span: min-width on a table cell is only a hint to table
    // layout, on a block inside it a hard minimum.
    if (i === 0) {
      const span = document.createElement("span");
      span.className = "metric-head-label";
      span.textContent = label;
      th.appendChild(span);
    } else {
      th.textContent = label;
    }
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);

  const tbody = document.createElement("tbody");
  section.rows.forEach((row) => {
    const tr = document.createElement("tr");
    row.cells.forEach((value, i) => {
      const td = document.createElement("td");
      if (i === 0) {
        const label = document.createElement("span");
        label.className = "metric-label";
        label.textContent = value;
        td.appendChild(label);
        if (row.detail) {
          const detail = document.createElement("span");
          detail.className = "metric-detail";
          detail.textContent = row.detail;
          td.appendChild(detail);
        }
      } else {
        td.textContent = value;
        td.className = "metric-value";
        // before/after rows mark the trailing change cell; comparison rows mark per cell
        const verdict = Array.isArray(row.verdicts)
          ? row.verdicts[i]
          : i === row.cells.length - 1
            ? row.verdict
            : null;
        if (verdict) td.classList.add(`metric-${verdict}`);
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.append(thead, tbody);

  // Wide comparisons (e.g. five models x two cuts) would squeeze the metric name to a
  // letter per line; scroll sideways instead, with the metric column pinned
  // (see .analyze-table-scroll).
  const scroller = document.createElement("div");
  scroller.className = "analyze-table-scroll";
  scroller.appendChild(table);
  return scroller;
}
