// Document mode: what it leaves in the page, what it records, and what it counts.
//
//   cd extension/tests && npm install && node document-sections.test.js
//
// content.js is loaded as in document-watcher.test.js (vm context, jsdom, minimal chrome.*).
//
// inline-markup.test.js pins the same contract for the sentence path. Regressions pinned
// here: a section folded into its first paragraph with the rest hidden (deleting links and
// list items); the model input recorded as the original (invented full stops, one run-on
// line); and skipped leaves left uncounted (25 leaves reported as one item).
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { JSDOM } = require("jsdom");

const source = fs.readFileSync(path.join(__dirname, "..", "content.js"), "utf8");
const LABELS_JS = fs.readFileSync(path.join(__dirname, "..", "shared", "model-labels.js"), "utf8");
// loaded between model-selection.js and content.js in manifest.json: lock key, run kinds, wording
const ACTIVE_RUN_JS = fs.readFileSync(path.join(__dirname, "..", "shared", "active-run.js"), "utf8");

let failures = 0;
// `detail` is printed only on failure
function check(label, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${ok || !detail ? "" : ` — ${detail}`}`);
  if (!ok) failures += 1;
}

// `simplify` maps a section's document to the backend's answer; undefined returns it
// unchanged (a real backend response too).
function loadContentScript(html, simplify) {
  const dom = new JSDOM(`<!doctype html><html><body><main>${html}</main></body></html>`, {
    pretendToBeVisual: true,
    url: "https://example.com/article",
  });
  const { window } = dom;
  const sent = [];
  const notices = [];
  const recorded = [];

  const context = {
    window,
    document: window.document,
    location: window.location,
    history: window.history,
    Node: window.Node,
    MutationObserver: window.MutationObserver,
    getComputedStyle: window.getComputedStyle.bind(window),
    performance: window.performance,
    setTimeout: window.setTimeout.bind(window),
    clearTimeout: window.clearTimeout.bind(window),
    setInterval: window.setInterval.bind(window),
    clearInterval: window.clearInterval.bind(window),
    console,
    crypto: { randomUUID: () => "test-page-session" },
    chrome: {
      runtime: {
        onMessage: { addListener() {} },
        lastError: null,
        sendMessage(message, callback) {
          // one-run-at-a-time lock (shared/active-run.js): uncontended here, always granted
          if (message.cmd === "claimRun") {
            if (callback) callback({ ok: true, holder: null });
            return;
          }
          if (message.cmd === "runProgress" || message.cmd === "releaseRun") {
            if (callback) callback({ ok: true });
            return;
          }
          if (message.cmd === "recordHistory") {
            recorded.push(message);
            return;
          }
          if (message.cmd !== "fetchSimplify") return;
          sent.push(message.text);
          const answer = simplify(message.text);
          callback({
            ok: true,
            model: "test-model",
            audience: "non_native_speakers",
            data: { simplified: answer === undefined ? message.text : answer, cached: false },
          });
        },
      },
      storage: { local: { get: (_keys, cb) => cb && cb({}) } },
    },
    notices,
  };
  context.globalThis = context;
  vm.createContext(context);
  vm.runInContext(LABELS_JS, context);
  vm.runInContext(ACTIVE_RUN_JS, context);
  vm.runInContext(source, context);
  // Notices as text: appending the real one is a mutation the document watcher would see.
  vm.runInContext(
    `showNotice = function (content) {
       notices.push(typeof content === "string" ? content : content.map((n) => n.textContent).join(" "));
     };`,
    context
  );

  return {
    window,
    sent,
    notices,
    recorded,
    run: (expression) => vm.runInContext(expression, context),
    el: (selector) => window.document.querySelector(selector),
    all: (selector) => Array.from(window.document.querySelectorAll(selector)),
    // what the toolbar click runs after the health preflight (see toggleSimplification);
    // the phase decides whether the notice draws the preflight box or the run's tally
    simplifyPage: () =>
      vm.runInContext(
        `beginSession();
         session.phase = "simplifying";
         session.startedAt = performance.now();
         simplifyPage(GRANULARITY_WHOLE_SECTIONS, 512)`,
        context
      ),
    entries: () => recorded.flatMap((message) => message.entries),
    stats: () => (recorded[0] || {}).session && recorded[0].session.stats,
  };
}

// Reported case, reduced: a paragraph and a list of linked items in one section. Folding
// hid every list item and its links. The links are `translate="no"` as on the source page,
// so their labels are never sent, and losing them was a deletion, not a rewrite.
const LIST_SECTION = `
  <h1>Extension Demo Pages</h1>
  <p>Static test pages for manually verifying the browser extension end to end.</p>
  <ul>
    <li><a href="01-basic.html" translate="no">01 — Basic paragraphs</a> shows the baseline toggle and revert behaviour.</li>
    <li><a href="02-dom.html" translate="no">02 — DOM targeting</a> covers headings, lists, tables and quotes.</li>
  </ul>`;

const LIST_SECTION_ANSWER =
  "The pages check the extension end to end. " +
  "The first page shows toggling and reverting. " +
  "The second page covers headings, lists, tables and quotes.";

async function everyElementOfASectionSurvives() {
  const ctx = loadContentScript(LIST_SECTION, () => LIST_SECTION_ANSWER);
  await ctx.simplifyPage();

  check("the section went out as one request", ctx.sent.length === 1);
  check("both list items are still in the page", ctx.all("li").length === 2);
  check("...and neither is hidden", ctx.all("li").every((li) => li.style.display !== "none"));
  check("both links are still in the page", ctx.all("li a").length === 2);
  check(
    "...still pointing where they did",
    ctx.all("li a").map((a) => a.getAttribute("href")).join() === "01-basic.html,02-dom.html"
  );
  check(
    "...and still holding their own labels",
    ctx.all("li a").map((a) => a.textContent).join("|") === "01 — Basic paragraphs|02 — DOM targeting"
  );
  check(
    "the answer is spread across the section, not piled into its first element",
    ctx.el("p").textContent === "The pages check the extension end to end."
  );
  check(
    "each list item holds its own share of it",
    ctx.all("li")[1].textContent.includes("The second page covers headings, lists, tables and quotes.")
  );
  check("the heading is untouched", ctx.el("h1").textContent === "Extension Demo Pages");
  check("...and was never sent", !ctx.sent.join(" ").includes("Extension Demo Pages"));
}

// demo/index.html verbatim: each item's subject is an opted-out link. Stripping the
// opt-outs left dangling colons ("…for each page.  : baseline toggle/revert behaviour.")
// and the model invented prose. Document mode reads the labels as context but never
// writes them.
const LABELLED_LIST = `
  <h1>Extension Demo Pages</h1>
  <p>Static test pages for manually verifying the browser extension end to end.</p>
  <ul>
    <li><a href="01-basic-paragraphs.html" translate="no" class="notranslate">01 — Basic paragraphs</a>: baseline toggle/revert behaviour.</li>
    <li><a href="02-dom-targeting.html" translate="no" class="notranslate">02 — DOM targeting</a>: headings, lists, tables, semantic sectioning tags, quotes.</li>
  </ul>`;

const LABELLED_LIST_ANSWER =
  "The pages test the extension from end to end. " +
  "01 — Basic paragraphs shows how to turn it on and off. " +
  "02 — DOM targeting covers headings, lists and tables.";

async function optedOutLabelsAreDocumentContext() {
  const ctx = loadContentScript(LABELLED_LIST, () => LABELLED_LIST_ANSWER);
  await ctx.simplifyPage();

  check("the section went out as one request", ctx.sent.length === 1);
  check(
    "the document carries each item's link label",
    ctx.sent[0] ===
      "Static test pages for manually verifying the browser extension end to end. " +
        "01 — Basic paragraphs: baseline toggle/revert behaviour. " +
        "02 — DOM targeting: headings, lists, tables, semantic sectioning tags, quotes."
  );
  check("...so no sentence in it begins on a dangling separator", !/[.\s]:\s/.test(ctx.sent[0]));

  // the log shows the document as sent, labels included
  const entry = ctx.entries()[0];
  check(
    "the recorded original names each item too",
    entry && entry.input.includes("01 — Basic paragraphs: baseline toggle/revert behaviour.")
  );

  check("both links are still in the page", ctx.all("li a").length === 2);
  check(
    "...still holding their own labels, unrewritten",
    ctx.all("li a").map((a) => a.textContent).join("|") === "01 — Basic paragraphs|02 — DOM targeting"
  );
  check(
    "a label the answer repeated is not written into the page a second time",
    ctx.all("li")[0].textContent === "01 — Basic paragraphs shows how to turn it on and off."
  );
  check(
    "...and the same for the second item",
    ctx.all("li")[1].textContent === "02 — DOM targeting covers headings, lists and tables."
  );
  check(
    "each item's tooltip is what the page said, label included",
    ctx.all("li")[0].title === "01 — Basic paragraphs: baseline toggle/revert behaviour."
  );

  ctx.run("revertPage()");
  check(
    "reverting restores the opted-out markup exactly",
    ctx.all("li")[0].innerHTML ===
      `<a href="01-basic-paragraphs.html" translate="no" class="notranslate">01 — Basic paragraphs</a>: baseline toggle/revert behaviour.`
  );
}

async function revertRestoresTheWholeSection() {
  const ctx = loadContentScript(LIST_SECTION, () => LIST_SECTION_ANSWER);
  await ctx.simplifyPage();
  ctx.run("revertPage()");

  check(
    "reverting restores the paragraph exactly",
    ctx.el("p").innerHTML === "Static test pages for manually verifying the browser extension end to end."
  );
  check(
    "...and each list item's markup with it",
    ctx.all("li")[0].innerHTML ===
      `<a href="01-basic.html" translate="no">01 — Basic paragraphs</a> shows the baseline toggle and revert behaviour.`
  );
  check("nothing is left marked as simplified", ctx.all(".simplified").length === 0);
  check("...and nothing is left hidden", ctx.all("[style*='display: none']").length === 0);
}

// A <div> with only inline children is a paragraph; once excluded, its prose was silently never sent.
async function aLeafDivIsBodyProse() {
  const ctx = loadContentScript(
    `<div id="pane">The movement argued that attempts to rewrite the political reality of the last election would neither strengthen democracy nor promote national unity.</div>`,
    () => "The movement said rewriting the last election would not help democracy or unity."
  );
  await ctx.simplifyPage();

  check("a leaf div's prose is sent", ctx.sent.length === 1);
  check(
    "...and the answer is written into it",
    ctx.el("#pane").textContent === "The movement said rewriting the last election would not help democracy or unity."
  );
  check("...and it is marked as simplified", ctx.el("#pane").classList.contains("simplified"));
}

// Tooltip and History log show the page, not the request (ensureTerminalPunctuation and
// whitespace collapse apply only to what is sent).
async function theRecordedOriginalIsThePage() {
  const ctx = loadContentScript(
    `<p>The association was founded in 1988<br>Its headquarters are in San Jose</p>
     <p>The association represents developers of video games worldwide.</p>`,
    () => "The group started in 1988. Its office is in San Jose. It speaks for game developers everywhere."
  );
  await ctx.simplifyPage();

  const entry = ctx.entries()[0];
  check("one history entry per section, not per element", ctx.entries().length === 1);
  check(
    "the recorded input keeps the <br> as a line break",
    entry && entry.input.startsWith("The association was founded in 1988\nIts headquarters are in San Jose")
  );
  check(
    "...and separates the section's elements with a blank line",
    entry && entry.input.includes("San Jose\n\nThe association represents")
  );
  check(
    "...and invents no punctuation the page didn't have",
    entry && !entry.input.includes("1988.")
  );
  check(
    "the model, meanwhile, was sent one line with terminal punctuation",
    ctx.sent[0] ===
      "The association was founded in 1988. Its headquarters are in San Jose. " +
        "The association represents developers of video games worldwide."
  );
  check(
    "each element's tooltip is its own original text",
    ctx.el("p").title === "The association was founded in 1988\nIts headquarters are in San Jose"
  );
  check(
    "...not the whole section's",
    ctx.all("p")[1].title === "The association represents developers of video games worldwide."
  );
  check("the <br> is still there", ctx.all("p")[0].querySelectorAll("br").length === 1);
}

// The tally makes the two paths comparable; skipped leaves must count.
async function skippedLeavesAreCounted() {
  const ctx = loadContentScript(
    `<h2>A heading</h2>
     <p>This paragraph is long enough to be sent as a real document of prose.</p>
     <p>Short.</p>
     <button>Go</button>
     <pre>const x = 1;</pre>`,
    () => "This paragraph is simple enough to read now."
  );
  await ctx.simplifyPage();

  const stats = ctx.stats();
  check("the section that was sent is one item", stats && stats.processedItems === 1);
  check("the heading is counted as skipped", stats && stats.skippedItems >= 1);
  check(
    "so are the too-short paragraph and the orphan button",
    stats && stats.skippedItems === stats.totalItems - 1 && stats.skippedItems >= 3
  );
  check("the counts add up", stats && stats.totalItems === stats.skippedItems + stats.processedItems);
  check(
    "and the notice says so rather than staying silent about them",
    ctx.notices.some((text) => /items skipped/.test(text))
  );
}

// A node whose text contains a source line break must still anchor in a one-line answer;
// otherwise its text went to the previous node (a <code> label was left empty).
async function aLineBreakInsideANodeStillAnchors() {
  const ctx = loadContentScript(
    `<p>See <code>demo/README.md</code> for setup instructions and an expected-results
checklist for each page.</p>`,
    () => "See demo/README.md for the setup instructions and a checklist for each page."
  );
  await ctx.simplifyPage();

  check("the code element still holds its own label", ctx.el("code").textContent === "demo/README.md");
  check(
    "the sentence reads as the model wrote it",
    ctx.el("p").textContent.replace(/\s+/g, " ").trim() ===
      "See demo/README.md for the setup instructions and a checklist for each page."
  );
}

// Under `white-space: pre-wrap` a blank line inside one text node is a paragraph break;
// the pane must come back as separate paragraphs, not one block.
const PRE_WRAP_PANE = `<div id="pane" style="white-space: pre-wrap;">The movement referred to comments made by its national leader, saying the outcome of the election had already disproved the claim.

The movement argued that attempts to rewrite the political reality of the last election would neither strengthen democracy nor promote national unity.

They instead encourage divisions at a time when readers expect their leaders to focus on the challenges confronting the nation.</div>`;

async function blankLinesStayParagraphs() {
  const ctx = loadContentScript(PRE_WRAP_PANE, () =>
    "The movement pointed to what its leader said about the election result. " +
      "It said rewriting the last election would not help democracy or unity. " +
      "It said this only causes division when leaders should focus on real problems."
  );
  await ctx.simplifyPage();

  const text = ctx.el("#pane").textContent;
  check("the pane's paragraphs go out in one request, not one each", ctx.sent.length === 1);
  check("...as one line, which is what the model was trained on", !ctx.sent[0].includes("\n"));
  check("the answer comes back as three paragraphs, as it went in", text.split(/\n\s*\n/).length === 3);
  check(
    "each paragraph holds its own share of the answer",
    text.split(/\n\s*\n/)[1].trim() === "It said rewriting the last election would not help democracy or unity."
  );
  check(
    "the recorded original keeps the blank lines too",
    (ctx.entries()[0] || {}).input.split(/\n\s*\n/).length === 3
  );

  ctx.run("revertPage()");
  check(
    "reverting restores the pane's own paragraphs",
    ctx.el("#pane").textContent.split(/\n\s*\n/).length === 3 &&
      ctx.el("#pane").textContent.startsWith("The movement referred to comments")
  );
}

// Shape of demo/07's first section: two paragraphs sent as one document, answer covers
// only the first. The second stays as-is (correct), but must be reported rather than
// shown as "1 item · 1 simplified".
async function unreachedParagraphsAreReported() {
  const ctx = loadContentScript(
    `<h1>History</h1>
     <p>The International Game Developers Association is incorporated in the United States as a not-for-profit organization with over twelve thousand members.</p>
     <p>In recognition of the multidisciplinary nature of contemporary game development, everyone who participates in the process may apply for membership.</p>`,
    () => "The IGDA is a not-for-profit group. It has more than twelve thousand members."
  );
  await ctx.simplifyPage();

  check("the section went out as one request", ctx.sent.length === 1);
  check(
    "the paragraph the answer covered was rewritten",
    ctx.all("p")[0].classList.contains("simplified")
  );
  check(
    "the one it didn't is left exactly as the page wrote it",
    !ctx.all("p")[1].classList.contains("simplified") &&
      ctx.all("p")[1].textContent.startsWith("In recognition of the multidisciplinary"),
    ctx.all("p")[1].textContent
  );

  const stats = ctx.stats();
  check(
    "the run counts the paragraph its answer never reached",
    stats && stats.unwrittenUnits === 1,
    JSON.stringify(stats)
  );
  const entry = ctx.entries()[0];
  check(
    "...and the entry says how much of its section was covered",
    entry && entry.coverage && entry.coverage.units === 2 && entry.coverage.written === 1,
    JSON.stringify(entry && entry.coverage)
  );
  check(
    "the notice says so rather than reporting a clean sweep",
    ctx.notices.some((text) => /left as-is/.test(text)),
    JSON.stringify(ctx.notices)
  );
}

// Counterpart: full coverage records nothing extra, so "left as-is" never appears.
async function aFullyCoveredSectionReportsNothingExtra() {
  const ctx = loadContentScript(
    `<p>The association was founded in nineteen eighty-eight by game developers.</p>
     <p>Its headquarters are in San Jose, California, in the United States.</p>`,
    () => "The group started in 1988. Its office is in San Jose."
  );
  await ctx.simplifyPage();

  const stats = ctx.stats();
  check(
    "a fully covered section records no unwritten units",
    stats && stats.unwrittenUnits === undefined,
    JSON.stringify(stats)
  );
  const entry = ctx.entries()[0];
  check("...and its entry carries no coverage field", entry && entry.coverage === undefined);
}

// Allocation properties: order kept, nothing dropped, no unit starved while sentences remain.
function sentencesAreHandedOutInOrder() {
  const ctx = loadContentScript(`<p>placeholder text long enough to be collected</p>`, () => undefined);
  const distribute = (words, output) =>
    ctx.run(
      `distributeSentences(${JSON.stringify(words.map((text) => ({ text })))}, ${JSON.stringify(output)})`
    );

  const four = "One two three. Four five six. Seven eight nine. Ten eleven twelve.";
  const even = distribute(["a b c", "d e f"], four);
  check("two equal units split four sentences evenly", even.join("|") === "One two three. Four five six.|Seven eight nine. Ten eleven twelve.");

  const skewed = distribute(["a b c d e f g h i", "j k l"], four);
  check("a unit that was most of the input gets most of the answer", skewed[0].split(".").length - 1 === 3);
  check("...and the other still gets some of it", skewed[1] === "Ten eleven twelve.");

  const scarce = distribute(["a b c", "d e f", "g h i"], "Only one sentence came back.");
  check("fewer sentences than units fills the front of the section", scarce[0] === "Only one sentence came back.");
  check("...and leaves the rest empty rather than splitting a sentence", scarce.slice(1).join("") === "");

  const leftover = distribute(["a b c", "d e f"], "One. Two. Three. Four. Five.");
  check(
    "no sentence of the answer is dropped",
    leftover.join(" ").match(/\./g).length === 5
  );
  check("...and their order is the answer's order", leftover.join(" ") === "One. Two. Three. Four. Five.");

  // Reported shift: a boundary without a space ("week.Panettiere") read as one sentence
  // under a whitespace-only split, shifting every later unit onto its neighbour's text.
  const glued = distribute(
    ["Her death was announced this week", "She was 16 when drugs arrived", "Someone handed her a pill"],
    "Her death was announced this week.Panettiere was 16 when drugs arrived. Someone handed her a pill."
  );
  check(
    "a boundary the model glued shut still separates two sentences",
    glued[0] === "Her death was announced this week." && glued[1] === "Panettiere was 16 when drugs arrived.",
    JSON.stringify(glued)
  );
  check(
    "...so the units after it are not shifted onto their neighbour's text",
    glued[2] === "Someone handed her a pill.",
    JSON.stringify(glued)
  );

  // lookalikes are not boundaries
  const abbreviated = distribute(["a b c", "d e f"], "The U.S.Congress met. It voted for the bill.");
  check(
    "an abbreviation is not read as a boundary",
    abbreviated[0] === "The U.S.Congress met.",
    JSON.stringify(abbreviated)
  );
  const filename = distribute(["a b c"], "See demo/README.md for the setup instructions.");
  check(
    "...nor is a glued filename",
    filename[0] === "See demo/README.md for the setup instructions.",
    JSON.stringify(filename)
  );
}

(async () => {
  await everyElementOfASectionSurvives();
  await optedOutLabelsAreDocumentContext();
  await revertRestoresTheWholeSection();
  await aLeafDivIsBodyProse();
  await theRecordedOriginalIsThePage();
  await skippedLeavesAreCounted();
  await aLineBreakInsideANodeStillAnchors();
  await blankLinesStayParagraphs();
  await unreachedParagraphsAreReported();
  await aFullyCoveredSectionReportsNothingExtra();
  sentencesAreHandedOutInOrder();

  console.log(failures === 0 ? "\nall checks passed" : `\n${failures} check(s) failed`);
  process.exit(failures === 0 ? 0 : 1);
})();
