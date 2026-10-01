// History page's Analyze menu: the sentence segmentation behind every words-per-sentence
// figure, and the metrics "Run advanced analysis" computes.
//
//   cd extension/tests && npm install && node history-analysis.test.js
//
// history.js is a non-module script expecting `chrome` and the history.html DOM; it runs
// in a vm context with jsdom built from that markup, after the label files the page loads.
//
// Covers: abbreviations ("6 a.m.") are not sentence boundaries, quick stats carry no
// sentence count, and the run button produces numbers.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { JSDOM } = require("jsdom");

const EXT = path.join(__dirname, "..");
const SHARED = path.join(EXT, "shared");
const SOURCES = ["model-labels.js", "model-selection.js", "analysis.js", "history.js"].map((f) =>
  fs.readFileSync(path.join(SHARED, f), "utf8")
);
// markup without <script> tags; the scripts run below in the test's context
const PAGE_HTML = fs
  .readFileSync(path.join(EXT, "history.html"), "utf8")
  .replace(/<script[^>]*>[\s\S]*?<\/script>/g, "");

let failures = 0;
function check(label, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${ok || !detail ? "" : ` — ${detail}`}`);
  if (!ok) failures += 1;
}

function entry(input, output) {
  return {
    input,
    modelResult: output,
    output,
    changed: input !== output,
    cached: false,
    fallbackReason: null,
    requestTimeMs: 820,
  };
}

const PAGE = {
  sessionId: "session-1",
  title: "Bin collections",
  url: "https://example.com/bins",
  timestamp: 1700000000000,
  method: "prompted_llm",
  modelKey: "llm_7b",
  modelId: "qwen2.5:7b-instruct",
  granularity: "sentence_by_sentence",
  audience: "non_native_speakers",
  totalTimeMs: 4700,
  status: "completed",
  stats: {
    totalItems: 5,
    skippedItems: 2,
    processedItems: 3,
    simplifiedItems: 2,
    unchangedItems: 1,
    cachedResponses: 0,
  },
  entries: [
    entry(
      "Collections will commence at approximately 6 a.m. on Tuesdays.",
      "We collect the bins from about 6 a.m. on Tuesdays."
    ),
    entry(
      "Residents who require an assisted collection should contact the council in advance.",
      "Ask the council first if you need help. They can collect your bin from your door."
    ),
    entry("Contact us", "Contact us"),
  ],
};

function load() {
  const dom = new JSDOM(PAGE_HTML, { pretendToBeVisual: true, url: "chrome-extension://test/history.html" });
  const { window } = dom;
  const store = { simplifyHistory: [PAGE] };

  const context = {
    window,
    document: window.document,
    Blob: window.Blob,
    URL: window.URL,
    navigator: { clipboard: { writeText: () => Promise.resolve() } },
    confirm: () => true,
    setTimeout,
    clearTimeout,
    console,
    chrome: {
      runtime: { sendMessage: () => Promise.resolve({ ok: false, error: "not used here" }) },
      storage: {
        local: {
          get: (key) => Promise.resolve(key in store ? { [key]: store[key] } : {}),
          set: (values) => {
            Object.assign(store, values);
            return Promise.resolve();
          },
          remove: (key) => {
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
  return { context, window };
}

const { context, window } = load();
const tick = (ms = 0) => new Promise((resolve) => setTimeout(resolve, ms));

// --- segmentation
const SENTENCE_CASES = [
  ["Collections will commence at approximately 6 a.m. on Tuesdays.", 1],
  ["Bins go out at 6 a.m. Collections finish by noon.", 2],
  ["Dr. Patel signed the form.", 1],
  ["Ask Dr. Patel. She signed the form.", 2],
  ["The U.S. rules still apply.", 1],
  ["J. Smith wrote the report.", 1],
  ["Put the bins out. Take them in again.", 2],
  ["Is the bin full? Put it out! The lorry comes early.", 3],
  ["Bring one bag (green, not black). Leave the rest.", 2],
  ["Recycling, food waste, etc. go in the caddy.", 1],
  ["Bins are collected fortnightly, e.g. every other Tuesday.", 1],
  ["Contact us", 1],
  ["", 0],
];
SENTENCE_CASES.forEach(([text, expected]) => {
  const got = context.countSentences(text);
  check(`countSentences(${JSON.stringify(text.slice(0, 44))}) === ${expected}`, got === expected, `got ${got}`);
});

// FKGL divides words by sentences, so phantom boundaries skew the grade.
const example = "Collections will commence at approximately 6 a.m. on Tuesdays.";
const exampleWords = example.match(/[A-Za-z']+/g);
const exampleSyllables = exampleWords.reduce((sum, w) => sum + context.countSyllables(w), 0);
const oneSentence = 0.39 * exampleWords.length + 11.8 * (exampleSyllables / exampleWords.length) - 15.59;
const threeSentences = 0.39 * (exampleWords.length / 3) + 11.8 * (exampleSyllables / exampleWords.length) - 15.59;
const grade = context.fleschKincaidGrade(example);
check("fleschKincaidGrade divides by one sentence, not three", Math.abs(grade - oneSentence) < 1e-9,
  `got ${grade == null ? "null" : grade.toFixed(2)}, one sentence is ${oneSentence.toFixed(2)}, three was ${threeSentences.toFixed(2)}`);

// --- quick stats
context.openAnalyzeModal(PAGE, "page");
const summaryText = window.document.getElementById("analyze-summary").textContent;
check("quick stats no longer show a sentence count", !summaryText.includes("Sentence count"), summaryText);
check("quick stats show FKGL", summaryText.includes("FKGL"), summaryText);
const summaryTextHasNegativeZero = /-0%/.test(summaryText);
check("quick stats show the rows simplified", summaryText.includes("Rows simplified") && summaryText.includes("2/3"), summaryText);
check("advanced metrics are not computed on open", window.document.getElementById("analyze-results").textContent === "");

// --- advanced analysis
(async () => {
  window.document.getElementById("analyze-run").click();
  await tick(20);

  const results = window.document.getElementById("analyze-results");
  const status = window.document.getElementById("analyze-status").textContent;
  const headings = Array.from(results.querySelectorAll("h3")).map((h) => h.textContent);

  check("run advanced analysis renders sections", headings.join(",") === "Readability,Structure,Rewriting", headings.join(","));
  check("status reports the scope", /3 rows/.test(status), status);

  const rows = Array.from(results.querySelectorAll(".analyze-table tbody tr")).map((tr) =>
    Array.from(tr.querySelectorAll("td")).map((td) => td.textContent.trim())
  );
  const byLabel = (label) => rows.find((cells) => cells[0].startsWith(label));

  const fkgl = byLabel("Flesch-Kincaid grade");
  check("Flesch-Kincaid row has before, after and a signed change", !!fkgl && /^[+-]/.test(fkgl[3]), fkgl && fkgl.join(" | "));
  check("reading ease is reported too", !!byLabel("Flesch reading ease"));

  // 1 + 1 originals + unchanged "Contact us"; the second output is two sentences (+1)
  const sentences = byLabel("Sentences");
  check("sentence counts moved into Structure, abbreviation-aware", !!sentences && sentences[1] === "3" && sentences[2] === "4",
    sentences && sentences.join(" | "));

  const words = byLabel("Words");
  check("word counts are reported", !!words && Number(words[1]) === 23 && Number(words[2]) === 28, words && words.join(" | "));

  const longWords = byLabel("Long words");
  check("a change in a percentage is reported in points", !!longWords && / pp$/.test(longWords[3]), longWords && longWords.join(" | "));
  check("an unchanged word count keeps no sign", !summaryTextHasNegativeZero, summaryText);

  const kept = byLabel("Words kept");
  check("rewriting breakdown reports kept words as a percentage", !!kept && /%$/.test(kept[1]), kept && kept.join(" | "));
  check("rewriting breakdown reports dropped and added words", !!byLabel("Words dropped") && !!byLabel("Words added"));
  const split = byLabel("Rows split into more sentences");
  check("split rows are counted", !!split && split[1] === "1 of 3", split && split.join(" | "));

  check("no metric is left as a placeholder", !/Reference required|On demand only/.test(results.textContent));
  check("the reference-based metrics are accounted for", /SARI/.test(results.textContent) && /BERTScore/.test(results.textContent));

  const marked = results.querySelectorAll(".metric-better, .metric-worse");
  check("change cells are marked by direction", marked.length > 0, `${marked.length} marked`);

  // the row menu's Analyze passes a single entry
  context.openAnalyzeModal({ entries: [PAGE.entries[0]] }, "entry");
  check("reopening clears the previous results", window.document.getElementById("analyze-results").textContent === "");
  window.document.getElementById("analyze-run").click();
  await tick(20);
  check("a single row analyzes too", /1 row\b/.test(window.document.getElementById("analyze-status").textContent),
    window.document.getElementById("analyze-status").textContent);

  console.log(failures === 0 ? "\nall history analysis checks passed" : `\n${failures} check(s) failed`);
  process.exit(failures === 0 ? 0 : 1);
})();
