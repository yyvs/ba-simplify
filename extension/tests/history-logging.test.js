// What a run actually writes to the History log.
//
//   cd extension/tests && npm install && node history-logging.test.js
//
// Counterpart to history-view.test.js; together they pin the log's shape from both sides.
// Nothing in between validates it: content.js posts a plain object, background.js stores it.
//
// content.js is loaded as in inline-markup.test.js, after shared/model-labels.js as in manifest.json.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { JSDOM } = require("jsdom");

const source = fs.readFileSync(path.join(__dirname, "..", "content.js"), "utf8");
const LABELS_JS = fs.readFileSync(path.join(__dirname, "..", "shared", "model-labels.js"), "utf8");
// loaded between model-selection.js and content.js in manifest.json: lock key, run kinds, wording
const ACTIVE_RUN_JS = fs.readFileSync(path.join(__dirname, "..", "shared", "active-run.js"), "utf8");

let failures = 0;
function check(label, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${ok || !detail ? "" : ` — ${detail}`}`);
  if (!ok) failures += 1;
}

// `respond` maps sent text to a backend body (e.g. a rejected generation or a split).
function loadContentScript(html, respond) {
  const dom = new JSDOM(`<!doctype html><html><body>${html}</body></html>`, {
    pretendToBeVisual: true,
    url: "https://example.com/bins",
  });
  const { window } = dom;
  window.document.title = "Bin collections";
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
          const body = respond(message.text) || {};
          callback({
            ok: true,
            model: "finetuned",
            audience: "non_native_speakers",
            data: { simplified: message.text, cached: false, ...body },
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
    recorded,
    run: (expression) => vm.runInContext(expression, context),
    // as the service worker would return it from the preflight
    setSelection: (selection) =>
      vm.runInContext(`session.selection = ${JSON.stringify(selection)}; session.clickedAt = 0;`, context),
    simplifyPage: () => vm.runInContext(`simplifyPage("sentence_by_sentence")`, context),
    // toggle off/on: the next run re-sends every unit to a backend that has cached them
    revertPage: () => vm.runInContext(`revertPage()`, context),
    // SPA route change as in navigation.test.js (jsdom refuses real location assignment)
    navigateTo: (url) => {
      vm.runInContext(`history.pushState({}, "", ${JSON.stringify(url)});`, context);
      vm.runInContext(`checkForNavigation()`, context);
    },
  };
}

const SEQ2SEQ_SELECTION = {
  model: "finetuned",
  modelId: "yunvs/bart-base-wikilarge-simplification",
  granularity: "sentence_by_sentence",
  audience: "non_native_speakers",
  audienceApplies: false,
};

const LLM_SELECTION = {
  model: "llm_7b",
  modelId: "qwen2.5:7b-instruct",
  granularity: "sentence_by_sentence",
  audience: "children",
  audienceApplies: true,
};

const KERBSIDE = "Residents are requested to position their receptacles at the kerbside.";
const KERBSIDE_SIMPLE = "Residents are asked to put their bins at the kerb.";
const COLLECTIONS = "Collections will commence at approximately 6 a.m. on Tuesdays.";
const COLLECTIONS_SIMPLE = "Collections start at about 6 a.m. on Tuesdays.";

(async () => {
  // --- one paragraph simplified, one rejected
  const ctx = loadContentScript(
    `<p id="a">${KERBSIDE}</p><p id="b">${COLLECTIONS}</p>`,
    (text) => {
      if (text.trim() === KERBSIDE) {
        return { simplified: KERBSIDE_SIMPLE, model_result: KERBSIDE_SIMPLE };
      }
      // only a full stop dropped: the change guard served the input back
      return {
        simplified: COLLECTIONS,
        model_result: COLLECTIONS.replace(/\.$/, ""),
        fallback_reason: "no_meaningful_change",
      };
    }
  );
  ctx.setSelection(SEQ2SEQ_SELECTION);
  await ctx.simplifyPage();

  check("the run posts one recordHistory message", ctx.recorded.length === 1, `${ctx.recorded.length} messages`);
  const message = ctx.recorded[0];

  // --- session level
  const s = message.session;
  check("the page it came from is identified", message.page.sessionId === "test-page-session" &&
    message.page.url === "https://example.com/bins" && message.page.title === "Bin collections",
    JSON.stringify(message.page));
  check("the method is recorded as a stable value, not a label", s.method === "fine_tuned_seq2seq", s.method);
  check("the exact model id is recorded", s.modelId === SEQ2SEQ_SELECTION.modelId, s.modelId);
  check("the model key is kept too, so a re-run can address the same model", s.modelKey === "finetuned", s.modelKey);
  check("the granularity says what was done to the page", s.granularity === "sentence_by_sentence", s.granularity);
  check(
    "a seq2seq run records no audience, because it acted on none",
    s.audience === undefined,
    JSON.stringify(s)
  );
  check("the run's elapsed time is on the session, not the entries", typeof s.totalTimeMs === "number", String(s.totalTimeMs));
  check(
    "a rejected generation makes the run's status a warning",
    s.status === "completed_with_warnings",
    s.status
  );
  check(
    "the counts are aggregated once per run",
    s.stats.processedItems === 2 && s.stats.simplifiedItems === 1 && s.stats.unchangedItems === 1 &&
      s.stats.totalItems === 2 && s.stats.cachedResponses === 0,
    JSON.stringify(s.stats)
  );

  // --- entry level
  const entries = message.entries;
  check("one entry per backend round trip", entries.length === 2, `${entries.length} entries`);

  const simplified = entries.find((e) => e.input === KERBSIDE);
  check("an entry names its input, not its 'original'", !!simplified && simplified.original === undefined);
  check("...and its output, not its 'simplified'", !!simplified && simplified.output === KERBSIDE_SIMPLE &&
    simplified.simplified === undefined, JSON.stringify(simplified));
  check("...with the model result alongside it", !!simplified && simplified.modelResult === KERBSIDE_SIMPLE);
  check("...and its own request time", !!simplified && typeof simplified.requestTimeMs === "number");
  check("...marked as changed, uncached, with no fallback", !!simplified && simplified.changed === true &&
    simplified.cached === false && simplified.fallbackReason === null, JSON.stringify(simplified));
  check(
    "the model and audience are no longer repeated on every entry",
    !!simplified && simplified.model === undefined && simplified.audience === undefined,
    JSON.stringify(simplified)
  );

  const rejected = entries.find((e) => e.input === COLLECTIONS);
  check(
    "a rejected generation is kept as the model result while the output is the fallback",
    !!rejected && rejected.modelResult === COLLECTIONS.replace(/\.$/, "") && rejected.output === COLLECTIONS,
    JSON.stringify(rejected)
  );
  check("...with the reason it was rejected", !!rejected && rejected.fallbackReason === "no_meaningful_change");
  check("...and marked unchanged, since the page's text did not move", !!rejected && rejected.changed === false);
  check(
    "an entry that wasn't split carries no sentenceSplit at all",
    !!rejected && rejected.sentenceSplit === undefined && simplified.sentenceSplit === undefined
  );

  // --- a split, and an audience that applied
  const split = loadContentScript(
    `<p id="a">${KERBSIDE} ${COLLECTIONS}</p>`,
    () => ({
      simplified: `${KERBSIDE_SIMPLE} Collections start at 6 a.m. on Tuesdays.`,
      model_result: `${KERBSIDE_SIMPLE} Collections start at 6 a.m. on Tuesdays.`,
      sentence_split: {
        count: 2,
        parts: [KERBSIDE_SIMPLE, "Collections start at 6 a.m. on Tuesdays."],
      },
    })
  );
  split.setSelection(LLM_SELECTION);
  await split.simplifyPage();

  const splitMessage = split.recorded[0];
  const splitEntry = splitMessage.entries[0];
  check(
    "a split entry records how many parts were combined, and which",
    !!splitEntry.sentenceSplit && splitEntry.sentenceSplit.count === 2 &&
      splitEntry.sentenceSplit.parts.length === 2 &&
      splitEntry.sentenceSplit.parts[0] === KERBSIDE_SIMPLE,
    JSON.stringify(splitEntry.sentenceSplit)
  );
  check(
    "a prompted run records the audience it was written for",
    splitMessage.session.audience === "children",
    JSON.stringify(splitMessage.session)
  );
  check("...under the prompted-LLM method", splitMessage.session.method === "prompted_llm", splitMessage.session.method);
  check(
    "a clean run's status is a plain completion",
    splitMessage.session.status === "completed",
    splitMessage.session.status
  );

  // --- re-run of an already recorded page
  // Regression: pageSessionId belongs to the page load, not the run, so each toggle
  // off/on appended duplicate entries (8 toggles of a two-section page logged 16).
  {
    let cached = false;
    const repeat = loadContentScript(
      `<p id="a">${KERBSIDE}</p><p id="b">${COLLECTIONS}</p>`,
      (text) => ({
        simplified: text.trim() === KERBSIDE ? KERBSIDE_SIMPLE : COLLECTIONS_SIMPLE,
        cached,
      })
    );
    repeat.setSelection(SEQ2SEQ_SELECTION);
    await repeat.simplifyPage();
    check(
      "the first run of a page is recorded",
      repeat.recorded.length === 1 && repeat.recorded[0].entries.length === 2,
      `${repeat.recorded.length} messages`
    );

    // same page, same session id, all answers now cached
    cached = true;
    repeat.revertPage();
    await repeat.simplifyPage();
    check(
      "a re-run served entirely from cache is not appended to it",
      repeat.recorded.length === 1,
      `${repeat.recorded.length} messages, entries: ${JSON.stringify(
        repeat.recorded.map((m) => m.entries.length)
      )}`
    );

    repeat.revertPage();
    await repeat.simplifyPage();
    check(
      "...however many times it is repeated",
      repeat.recorded.length === 1,
      `${repeat.recorded.length} messages`
    );
  }

  // --- mixed re-run keeps its cache hits
  // "Same headline on every subpage": the cached entry records which unit the model
  // wasn't asked about again.
  {
    let firstRun = true;
    const mixed = loadContentScript(
      `<p id="a">${KERBSIDE}</p><p id="b">${COLLECTIONS}</p>`,
      (text) => ({
        simplified: text.trim() === KERBSIDE ? KERBSIDE_SIMPLE : COLLECTIONS_SIMPLE,
        // the shared headline is cached on the second run; the body is not
        cached: !firstRun && text.trim() === KERBSIDE,
      })
    );
    mixed.setSelection(SEQ2SEQ_SELECTION);
    await mixed.simplifyPage();
    firstRun = false;
    mixed.revertPage();
    await mixed.simplifyPage();

    check(
      "a re-run with one fresh answer is recorded",
      mixed.recorded.length === 2,
      `${mixed.recorded.length} messages`
    );
    const second = mixed.recorded[1] || { entries: [] };
    check(
      "...in full, cached entries included",
      second.entries.length === 2 &&
        second.entries.filter((e) => e.cached).length === 1 &&
        second.entries.filter((e) => !e.cached).length === 1,
      JSON.stringify(second.entries.map((e) => e.cached))
    );
  }

  // --- first run served entirely from cache
  // e.g. after a route change; there is no earlier record to duplicate, so it is kept.
  {
    const fresh = loadContentScript(
      `<p id="a">${KERBSIDE}</p>`,
      () => ({ simplified: KERBSIDE_SIMPLE, cached: true })
    );
    fresh.setSelection(SEQ2SEQ_SELECTION);
    await fresh.simplifyPage();
    check(
      "a page's first run is recorded even when every answer came from cache",
      fresh.recorded.length === 1 && fresh.recorded[0].entries.length === 1,
      `${fresh.recorded.length} messages`
    );

    // a route change makes the next page a first run again
    fresh.navigateTo("https://example.com/bins/tuesday");
    fresh.run(`document.body.innerHTML = '<p id="c">${KERBSIDE}</p>';`);
    await fresh.simplifyPage();
    check(
      "...and so is the first run of the page a route change leads to",
      fresh.recorded.length === 2,
      `${fresh.recorded.length} messages`
    );
  }

  console.log(failures === 0 ? "\nall history logging checks passed" : `\n${failures} check(s) failed`);
  process.exit(failures === 0 ? 0 : 1);
})();
