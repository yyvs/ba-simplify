// DOM-level checks for what the sentence path leaves standing in the page.
//
//   cd extension/tests && npm install && node inline-markup.test.js
//
// content.js is loaded as in document-watcher.test.js (vm context, jsdom, minimal chrome.*).
//
// Contract (once broken): a simplification replaces the sent text and nothing else.
// Every element in the leaf survives (links, images, subtrees extraction skips), and
// revert restores the page exactly.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { JSDOM } = require("jsdom");

const CONTENT_JS = path.join(__dirname, "..", "content.js");
const source = fs.readFileSync(CONTENT_JS, "utf8");
// manifest.json loads shared/model-labels.js before content.js, which depends on it
// (e.g. sessionSummary).
const LABELS_JS = fs.readFileSync(path.join(__dirname, "..", "shared", "model-labels.js"), "utf8");
// loaded between model-selection.js and content.js in manifest.json: lock key, run kinds, wording
const ACTIVE_RUN_JS = fs.readFileSync(path.join(__dirname, "..", "shared", "active-run.js"), "utf8");

let failures = 0;
function check(label, ok) {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}`);
  if (!ok) failures += 1;
}

// `simplify` maps sent text to the backend's answer; undefined returns it unchanged
// (a real backend response too).
function loadContentScript(html, simplify) {
  const dom = new JSDOM(`<!doctype html><html><body>${html}</body></html>`, {
    pretendToBeVisual: true,
    url: "https://example.com/article",
  });
  const { window } = dom;
  const sent = [];

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
        // only the simplification proxy answers; badge/history are fire-and-forget
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
  };
  context.globalThis = context;
  vm.createContext(context);
  vm.runInContext(LABELS_JS, context);
  vm.runInContext(ACTIVE_RUN_JS, context);
  vm.runInContext(source, context);
  vm.runInContext(`showNotice = function () {};`, context);

  return {
    window,
    sent,
    run: (expression) => vm.runInContext(expression, context),
    el: (selector) => window.document.querySelector(selector),
    simplify: (selector) =>
      vm.runInContext(`simplifyElement(document.querySelector(${JSON.stringify(selector)}))`, context),
    // whole-page path: decides which elements are units of their own (last two checks)
    simplifyPage: () => vm.runInContext(`simplifyPage("sentence")`, context),
  };
}

// Reported case, verbatim: a `translate="no"` link is never sent, and was deleted along
// with the markup when the answer for the remaining ": ..." came back.
async function anOptedOutLinkSurvives() {
  const ctx = loadContentScript(
    `<ul><li><a href="07-document-sections.html" translate="no" class="notranslate">07 — Document sections</a>: heading-delimited sections for the document model.</li></ul>`,
    (text) => (text.trim() === ": heading-delimited sections for the document model."
      ? ": sections split by headings, for the document model."
      : undefined)
  );
  await ctx.simplify("li");

  const link = ctx.el("li a");
  check("the link the request never saw is still in the page", link !== null);
  check("...with its own text untouched", link && link.textContent === "07 — Document sections");
  check("...and its href intact", link && link.getAttribute("href") === "07-document-sections.html");
  check(
    "only the text that was sent came back changed",
    ctx.el("li").textContent === "07 — Document sections: sections split by headings, for the document model."
  );
  check("the item is marked as simplified", ctx.el("li").classList.contains("simplified"));

  ctx.run("revertPage()");
  check(
    "reverting restores the original markup exactly",
    ctx.el("li").innerHTML ===
      `<a href="07-document-sections.html" translate="no" class="notranslate">07 — Document sections</a>: heading-delimited sections for the document model.`
  );
}

// As above, but the model drops the ": " (usual, since a leading separator isn't in its
// training sentences), fusing answer and label: the reported
// `<a>01 — Basic paragraphs</a>Baseline toggle/revert function.`
async function aDroppedSeparatorComesBack() {
  const ctx = loadContentScript(
    `<ul><li><a href="01-basic-paragraphs.html" translate="no" class="notranslate">01 — Basic paragraphs</a>: baseline toggle/revert behaviour.</li></ul>`,
    () => "Baseline toggle/revert function."
  );
  await ctx.simplify("li");

  check(
    "a separator the model dropped is put back in front of the answer",
    ctx.el("li").textContent === "01 — Basic paragraphs: Baseline toggle/revert function.",
    JSON.stringify(ctx.el("li").textContent)
  );
  check(
    "...so the label and the answer are never fused into one word",
    !/paragraphs<\/a>[^\s:;,|–—-]/.test(ctx.el("li").innerHTML),
    ctx.el("li").innerHTML
  );

  // em dash separator the answer kept: restoring it would double it
  const kept = loadContentScript(
    `<ul><li><a translate="no" class="notranslate">05 — Dynamic content</a> — MutationObserver-style insertion of new nodes.</li></ul>`,
    () => "— New nodes are added while you read."
  );
  await kept.simplify("li");
  check(
    "a separator the answer kept is not added a second time",
    kept.el("li").textContent === "05 — Dynamic content — New nodes are added while you read.",
    JSON.stringify(kept.el("li").textContent)
  );

  // plain space separator
  const spaced = loadContentScript(
    `<ul><li><a translate="no" class="notranslate">06 — Line breaks</a> chunking and per-sentence simplification.</li></ul>`,
    () => "Chunking and simplification, one sentence at a time."
  );
  await spaced.simplify("li");
  check(
    "a chunk separated by nothing but a space keeps just its space",
    spaced.el("li").textContent === "06 — Line breaks Chunking and simplification, one sentence at a time.",
    JSON.stringify(spaced.el("li").textContent)
  );
}

// Second reported case, reduced: a card grid whose <div> is the leaf (inline-leaf
// children only), so one request covers every title. Images and links must survive.
async function imagesAndLinksSurvive() {
  const ctx = loadContentScript(
    `<div class="grid"><a class="item" href="/one"><span class="thumb"><img src="one.jpg" alt="One"></span><span class="title">Adeleke loses voice after praise and worship sessions</span></a></div>`,
    () => "Adeleke lost his voice after praise and worship"
  );
  await ctx.simplify("div.grid");

  check("the image is still there", ctx.el("div.grid img") !== null);
  check("...with its source untouched", ctx.el("div.grid img").getAttribute("src") === "one.jpg");
  check("the card is still a link", ctx.el("div.grid a.item") !== null);
  check("...still pointing where it did", ctx.el("div.grid a.item").getAttribute("href") === "/one");
  check(
    "the title's own span still holds the simplified title",
    ctx.el("div.grid .title").textContent === "Adeleke lost his voice after praise and worship"
  );
}

// Unchanged phrases stay in their element, not pulled into the leaf's first text node.
async function untouchedInlineTextStaysPut() {
  const ctx = loadContentScript(
    `<p>The <b>quick brown fox</b> jumps over the exceedingly lazy dog.</p>`,
    () => "The quick brown fox jumps over the lazy dog."
  );
  await ctx.simplify("p");

  check("the bold element is still there", ctx.el("p b") !== null);
  check("...holding exactly its own words", ctx.el("p b").textContent === "quick brown fox");
  check("the paragraph reads as the model wrote it", ctx.el("p").textContent === "The quick brown fox jumps over the lazy dog.");
}

// <br>s are chunk boundaries: one request per side, each answer back on its side.
async function lineBreaksStillSeparateChunks() {
  const ctx = loadContentScript(
    `<p>Founded in 1988 by three former students.<br>Headquarters are in San Jose, California.</p>`,
    (text) =>
      text.startsWith("Founded")
        ? "Started in 1988 by three former students."
        : "The head office is in San Jose, California."
  );
  await ctx.simplify("p");

  check("both sides of the break were sent separately", ctx.sent.length === 2);
  check("the <br> is still there", ctx.el("p br") !== null);
  check(
    "each answer landed on its own side of it",
    ctx.el("p").innerHTML === "Started in 1988 by three former students.<br>The head office is in San Jose, California."
  );
}

// A partly rewritten link keeps the longest contiguous run of its label's words the model
// kept ("Computer Game", with "Developers" cut from the middle); the rest stays in the
// sentence outside the link rather than stitching non-adjacent words together.
async function aHalfRewrittenLinkKeepsWhatSurvived() {
  const ctx = loadContentScript(
    `<p>The conference, later known as the <a href="/cgdc">Computer Game Developers Conference</a>, was started in 1988.</p>`,
    () => "The conference, later called the Computer Game Conference, started in 1988."
  );
  await ctx.simplify("p");

  check("the link keeps the longest surviving run of its label", ctx.el("p a").textContent === "Computer Game");
  check("...and the rest of the label is still in the sentence", ctx.el("p").innerHTML.includes("</a> Conference,"));
  check("...and is still a link to the same place", ctx.el("p a").getAttribute("href") === "/cgdc");
  check("the sentence still reads as the model wrote it", ctx.el("p").textContent === "The conference, later called the Computer Game Conference, started in 1988.");
}

// Known limit: a label the model deleted outright has no anchor. The link stays, empty.
async function aDeletedLabelLeavesTheLinkInPlace() {
  const ctx = loadContentScript(
    `<p>Please consult the <a href="/docs">supplementary documentation</a> before you begin.</p>`,
    () => "Please read this before you begin."
  );
  await ctx.simplify("p");

  check("a link whose label the model deleted is still in the page", ctx.el("p a") !== null);
  check("...and reverting brings its label back", (() => {
    ctx.run("revertPage()");
    return ctx.el("p a").textContent === "supplementary documentation";
  })());
}

// Invented text must not go inside the <strong> whose word it replaced, extending the
// author's emphasis.
async function rewrittenTextDoesNotLandInsideEmphasis() {
  const ctx = loadContentScript(
    `<p>The <strong>fundamental</strong> objective is to <em>substantially</em> ameliorate the <mark>overall comprehensibility</mark> of the material.</p>`,
    () => "The main goal is to greatly improve the overall comprehensibility of the material."
  );
  await ctx.simplify("p");

  check("all three formatting elements are still in the page", ctx.el("p").querySelectorAll("strong, em, mark").length === 3);
  check("the emphasis does not spread over words the author never emphasised", ctx.el("p strong").textContent === "");
  check("the phrase the model kept is still inside its own element", ctx.el("p mark").textContent === "overall comprehensibility");
  check("the sentence itself is what the model wrote", ctx.el("p").textContent === "The main goal is to greatly improve the overall comprehensibility of the material.");
}

// Under `white-space: pre-wrap` a blank line in a text node is a break like <br>. Without
// punctuation, two paragraphs otherwise reached the segmenter as one run.
async function aBlankLineSplitsAChunkLikeABr() {
  const ctx = loadContentScript(
    `<div style="white-space: pre-wrap;">The first paragraph says one whole thing about the subject.

The second paragraph says a different whole thing about it.</div>`,
    (text) =>
      text.trim() === "The first paragraph says one whole thing about the subject."
        ? "The first paragraph says one thing."
        : "The second paragraph says another thing."
  );
  await ctx.simplify("div");

  check("each paragraph was sent on its own", ctx.sent.length === 2);
  check(
    "...and the blank line between them still separates them",
    ctx.el("div").textContent ===
      "The first paragraph says one thing.\n\nThe second paragraph says another thing."
  );
  check("the tooltip records both, separated by the break", ctx.el("div").title.split(/\n\s*\n/).length === 2);

  ctx.run("revertPage()");
  check(
    "reverting restores the pane exactly",
    ctx.el("div").textContent ===
      "The first paragraph says one whole thing about the subject.\n\nThe second paragraph says a different whole thing about it."
  );
}

// Model output is untrusted. Write-back assigns to a text node's `.data`, which can't
// create elements.
async function modelOutputCannotCreateElements() {
  const ctx = loadContentScript(
    `<p>A sentence long enough to be sent to the backend for simplification.</p>`,
    () => `Simple text <img src=x onerror="throw new Error('executed')"> and more.`
  );
  await ctx.simplify("p");

  check("a generation containing markup creates no element", ctx.el("p").querySelectorAll("*").length === 0);
  check("...and no image in the document at all", ctx.window.document.querySelectorAll("img").length === 0);
  check(
    "...it is shown as the visible text it is",
    ctx.el("p").textContent === `Simple text <img src=x onerror="throw new Error('executed')"> and more.`
  );
}

// Nothing was rewritten, so nothing may be rewritten -- including the markup.
async function anUnchangedAnswerTouchesNothing() {
  const ctx = loadContentScript(
    `<p>A sentence <em>already</em> simple enough to leave alone.</p>`,
    (text) => text
  );
  const before = ctx.el("p").innerHTML;
  await ctx.simplify("p");

  check("an unchanged answer leaves the markup as it was", ctx.el("p").innerHTML === before);
  check("...and does not mark the element simplified", !ctx.el("p").classList.contains("simplified"));
}

// A control inside a label is its own unit; its text was once glued to the label's
// sentence ("...in the toolbarClear"), leaving the button empty.
async function aNestedControlKeepsItsOwnLabel() {
  const ctx = loadContentScript(
    `<div class="section"><h2>Quick simplification check</h2><label for="demo-text" style="display: flex">Enter text which should be simplified, when ready toggle the simplification by pressing the browser extension icon in the toolbar<button id="demo-clear" type="button">Clear</button></label></div>`,
    () => "Enter text to simplify, then press the extension icon."
  );
  await ctx.simplifyPage();

  check(
    "the button's own label is not glued to the end of the label's sentence",
    ctx.sent.every((text) => !text.includes("toolbarClear"))
  );
  check("...and is not sent as part of it at all", ctx.sent.every((text) => !text.includes("Clear")));
  check("the button still says what it does", ctx.el("button").textContent === "Clear");
  check("the label itself was simplified", ctx.el("label").classList.contains("simplified"));
}

// Counterpart: a link inside a paragraph is part of the sentence, not its own unit.
async function aLinkInsideAParagraphStaysPartOfTheSentence() {
  const ctx = loadContentScript(
    `<p>The conference was started in 1988 by <a href="/cc">Chris Crawford</a>, a game designer.</p>`,
    (text) => text
  );
  await ctx.simplifyPage();

  check("the paragraph goes out as one whole sentence", ctx.sent.length === 1);
  check(
    "...with the link's words in their place in it",
    ctx.sent[0] === "The conference was started in 1988 by Chris Crawford, a game designer."
  );
}

(async () => {
  await anOptedOutLinkSurvives();
  await aDroppedSeparatorComesBack();
  await imagesAndLinksSurvive();
  await untouchedInlineTextStaysPut();
  await aHalfRewrittenLinkKeepsWhatSurvived();
  await aDeletedLabelLeavesTheLinkInPlace();
  await rewrittenTextDoesNotLandInsideEmphasis();
  await aNestedControlKeepsItsOwnLabel();
  await aLinkInsideAParagraphStaysPartOfTheSentence();
  await aBlankLineSplitsAChunkLikeABr();
  await modelOutputCannotCreateElements();
  await lineBreaksStillSeparateChunks();
  await anUnchangedAnswerTouchesNothing();
  console.log(failures === 0 ? "\nall checks passed" : `\n${failures} check(s) failed`);
  process.exit(failures === 0 ? 0 : 1);
})();
