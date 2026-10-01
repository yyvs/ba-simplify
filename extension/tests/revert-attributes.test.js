// What the revert leaves behind on the elements themselves.
//
//   cd extension/tests && npm install && node revert-attributes.test.js
//
// inline-markup.test.js checks revert via innerHTML, which can't see the element's own
// attributes. Two regressions hid there: revert deleted a page-written `title`, and left
// empty `class`/`style` on elements that had none. Both found by the RQ2 site audit
// (research/audit_sites.mjs), which compares attribute sets before/after on real pages.
//
// A changed attribute value is the feature; a changed attribute set is damage.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { JSDOM } = require("jsdom");

const EXT = path.join(__dirname, "..");
// manifest's list and order, as audit_sites.mjs reads it
const CONTENT_SCRIPTS = JSON.parse(
  fs.readFileSync(path.join(EXT, "manifest.json"), "utf8")
).content_scripts[0].js;

let failures = 0;
function check(label, ok) {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}`);
  if (!ok) failures += 1;
}

function loadContentScript(html, simplify) {
  const dom = new JSDOM(`<!doctype html><html><body>${html}</body></html>`, {
    pretendToBeVisual: true,
    url: "https://example.com/article",
  });
  const { window } = dom;
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
          if (message.cmd === "claimRun") return callback && callback({ ok: true, holder: null });
          if (message.cmd === "runProgress" || message.cmd === "releaseRun") {
            return callback && callback({ ok: true });
          }
          if (message.cmd !== "fetchSimplify") return;
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
  for (const script of CONTENT_SCRIPTS) {
    vm.runInContext(fs.readFileSync(path.join(EXT, script), "utf8"), context);
  }
  vm.runInContext(`showNotice = function () {};`, context);
  return {
    window,
    run: (expression) => vm.runInContext(expression, context),
    el: (selector) => window.document.querySelector(selector),
    simplify: (selector) =>
      vm.runInContext(`simplifyElement(document.querySelector(${JSON.stringify(selector)}))`, context),
    attrs: (selector) =>
      [...window.document.querySelector(selector).attributes].map((a) => a.name).sort(),
  };
}

// While simplified, `title` holds the original text; revert must restore the page's own
// tooltip rather than remove the attribute.
async function aPagesOwnTitleSurvivesTheRoundTrip() {
  const ctx = loadContentScript(
    `<p title="Defined in the glossary">The utilisation of this methodology is contingent upon prior authorisation.</p>`,
    () => "You need permission before using this method."
  );
  await ctx.simplify("p");
  check(
    "while simplified, the tooltip shows the original text",
    ctx.el("p").getAttribute("title") ===
      "The utilisation of this methodology is contingent upon prior authorisation."
  );
  ctx.run("revertPage()");
  check(
    "reverting puts the page's own tooltip back",
    ctx.el("p").getAttribute("title") === "Defined in the glossary"
  );
}

// No title before: none (not an empty one) after.
async function anElementWithNoTitleGetsNoneBack() {
  const ctx = loadContentScript(
    `<p>The utilisation of this methodology is contingent upon prior authorisation.</p>`,
    () => "You need permission before using this method."
  );
  await ctx.simplify("p");
  ctx.run("revertPage()");
  check("an element that had no tooltip has none afterwards", !ctx.el("p").hasAttribute("title"));
}

// classList.remove() of the only class, and clearing an inline background, leave empty attributes.
async function noEmptyAttributesAreLeftBehind() {
  const ctx = loadContentScript(
    `<p>The utilisation of this methodology is contingent upon prior authorisation.</p>`,
    () => "You need permission before using this method."
  );
  const before = ctx.attrs("p");
  await ctx.simplify("p");
  ctx.run("revertPage()");
  check("the element had no attributes to begin with", before.length === 0);
  check(
    `nothing was left behind (got ${JSON.stringify(ctx.attrs("p"))})`,
    ctx.attrs("p").length === 0
  );
}

// Existing classes are kept, so the cleanup can't remove the attribute unconditionally.
async function anExistingClassIsKeptExactly() {
  const ctx = loadContentScript(
    `<p class="lead intro">The utilisation of this methodology is contingent upon prior authorisation.</p>`,
    () => "You need permission before using this method."
  );
  await ctx.simplify("p");
  check("the marker class is added while simplified", ctx.el("p").classList.contains("simplified"));
  ctx.run("revertPage()");
  check(
    `the page's own classes are intact and ours is gone (got ${JSON.stringify(ctx.el("p").getAttribute("class"))})`,
    ctx.el("p").getAttribute("class") === "lead intro"
  );
}

async function theAttributeSetIsUnchanged() {
  const ctx = loadContentScript(
    `<p id="x" class="lead" title="Glossary" data-page="4">The utilisation of this methodology is contingent upon prior authorisation.</p>`,
    () => "You need permission before using this method."
  );
  const before = ctx.attrs("p");
  await ctx.simplify("p");
  ctx.run("revertPage()");
  const after = ctx.attrs("p");
  check(
    `the attribute set is unchanged (${JSON.stringify(before)} -> ${JSON.stringify(after)})`,
    JSON.stringify(before) === JSON.stringify(after)
  );
}

// Pages can ship class="" themselves (nextjs.org's docs navigation, caught by the audit),
// so the cleanup records what it added instead of removing empty attributes.
async function aPagesOwnEmptyClassIsNotRemoved() {
  const ctx = loadContentScript(
    `<p class="">The utilisation of this methodology is contingent upon prior authorisation.</p>`,
    () => "You need permission before using this method."
  );
  await ctx.simplify("p");
  ctx.run("revertPage()");
  check(
    `an empty class the page wrote is still there (got ${JSON.stringify(ctx.el("p").getAttribute("class"))})`,
    ctx.el("p").getAttribute("class") === ""
  );
}

// same for style
async function aPagesOwnEmptyStyleIsNotRemoved() {
  const ctx = loadContentScript(
    `<p style="">The utilisation of this methodology is contingent upon prior authorisation.</p>`,
    () => "You need permission before using this method."
  );
  await ctx.simplify("p");
  ctx.run("revertPage()");
  check("an empty style the page wrote is still there", ctx.el("p").hasAttribute("style"));
}

(async () => {
  await aPagesOwnTitleSurvivesTheRoundTrip();
  await anElementWithNoTitleGetsNoneBack();
  await noEmptyAttributesAreLeftBehind();
  await anExistingClassIsKeptExactly();
  await theAttributeSetIsUnchanged();
  await aPagesOwnEmptyClassIsNotRemoved();
  await aPagesOwnEmptyStyleIsNotRemoved();
  console.log(failures === 0 ? "\nall checks passed" : `\n${failures} check(s) failed`);
  process.exit(failures === 0 ? 0 : 1);
})();
