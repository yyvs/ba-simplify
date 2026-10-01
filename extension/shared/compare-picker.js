// shared/compare-picker.js - panel that sets up a multi-model comparison: pick the
// cut(s) the page is split into, then any number of models that can read that unit.
//
// Granularity is multiple choice. Sentence scope sends one request per leaf element;
// document scope sends heading-delimited sections whole (see content.js's two
// collectors). Ticking both gives one comparison whose entries are the coarsest cut's
// units: each section row carries its whole-section answer and the joined sentence
// answers for the units inside it (collectComparisonPlan in content.js), so both answer
// the same text.
//
// Which cuts a key can run is a property of the model. A sentence-scope key runs in
// both (in the sections cut the backend splits, simplifies per sentence and rejoins).
// A whole-sections key runs only in the sections cut; a document checkpoint given one
// leaf element isn't a document-scope run. See compareCutCandidates and cutReadingLabel
// in shared/model-labels.js.
//
// Kept separate from shared/picker.js: that panel writes the stored selection; this one
// starts a run and leaves the selection untouched.
//
// Reuses picker.js's PICKER_STYLES, pickerEl() and mark constants (globals in the same
// content-script scope, see manifest.json). Adds checkbox rows and a run footer.
const COMPARE_HOST_ID = "simplify-compare-picker";

// ☑ / ☐ rather than picker.js's radio ● / ○: both controls here are multi-select.
const COMPARE_MARK_ON = "☑";
const COMPARE_MARK_OFF = "☐";

const COMPARE_STYLES = `
  /* the run button and what it says about itself, which the radio panel has no
     equivalent of: that panel commits on every click, this one commits once. */
  .footer {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 20px;
    padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.14);
  }
  .footer-note { flex: 1; font-size: 12px; color: rgba(255,255,255,0.55); }
  .run {
    flex: none;
    padding: 10px 16px;
    border: 0;
    border-radius: 8px;
    background: #4688f1;
    color: #fff;
    font: inherit;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
  }
  .run:hover:not([disabled]) { background: #5a97f5; }
  .run[disabled] { opacity: 0.4; cursor: default; }
  .run:focus-visible { outline: 2px solid #fff; outline-offset: 2px; }
`;

const compareState = {
  host: null,
  root: null,
  card: null,
  // chosen cuts, as granularity values
  granularities: new Set(),
  // model keys, not families: one family can contribute both its whole-sections and its
  // sentence checkpoint to a comparison over sections (see compareUnitCandidates).
  selected: new Set(),
  health: null,
  healthError: null,
};

// --- rendering --------------------------------------------------------------

function compareIsOpen() {
  return !!compareState.host && compareState.host.isConnected;
}

function compareModelsLoaded() {
  return compareState.health && Array.isArray(compareState.health.models_loaded)
    ? compareState.health.models_loaded
    : null;
}

// Every key that can read the chosen cuts, in panel order. Combinations that don't
// exist (e.g. the comparison checkpoint has no document-scope counterpart) are omitted,
// not greyed out.
function compareCandidates() {
  return compareCutCandidates(compareCuts(), compareModelsLoaded());
}

// Chosen cuts, coarsest first: run order, and the first defines the entries' unit.
function compareCuts() {
  return comparisonCutOrder(Array.from(compareState.granularities));
}

function compareGranularitySection() {
  const segments = pickerEl("div", "segments");
  segments.setAttribute("role", "group");
  segments.setAttribute("aria-label", "Granularity");
  GRANULARITY_CHOICES.forEach((choice) => {
    const selected = compareState.granularities.has(choice.value);
    const btn = pickerEl("button", "segment");
    btn.type = "button";
    btn.setAttribute("role", "checkbox");
    btn.setAttribute("aria-checked", String(selected));
    btn.dataset.compareGranularity = choice.value;
    btn.append(
      pickerEl("span", "mark", selected ? COMPARE_MARK_ON : COMPARE_MARK_OFF),
      pickerEl("span", null, choice.label)
    );
    segments.appendChild(btn);
  });
  const section = pickerSection("Granularity", segments);
  section.appendChild(
    pickerEl(
      "div",
      "hint",
      compareState.granularities.size > 1
        ? "Both: the page is cut into sections, and each section is also sent a leaf " +
            "element at a time. One row per section, holding both answers, so the two " +
            "ways of reading a page can be compared rather than only each model in one."
        : "What every model is given: whole leaf elements, or whole sections. Tick both " +
            "to compare the two ways of reading the same page."
    )
  );
  return section;
}

function compareModelSection() {
  const group = pickerEl("div", "options");
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", "Models to compare");
  const candidates = compareCandidates();
  const names = compareState.health && compareState.health.model_names;

  candidates.forEach(({ familyId, key, keyGranularity, cuts, available, reason }) => {
    const checked = compareState.selected.has(key);
    const row = pickerEl("button", "option");
    row.type = "button";
    row.setAttribute("role", "checkbox");
    row.setAttribute("aria-checked", String(checked));
    if (!available) {
      row.disabled = true;
      row.title = reason;
    }
    row.dataset.compareModel = key;
    row.append(
      pickerEl("span", "mark", checked ? COMPARE_MARK_ON : COMPARE_MARK_OFF),
      pickerEl("span", "option-title", modelFamilyTitle(familyId, names))
    );
    // method, then what this key is given in each chosen cut; distinguishes two rows of
    // the same family, and shows when one model runs in both cuts
    const reading = cuts.map((cut) => cutReadingLabel(cut, keyGranularity)).join(", then ");
    const detail = [modelFamilyMethod(familyId), reading]
      .concat(available ? [] : [reason])
      .filter(Boolean)
      .join(" · ");
    row.appendChild(pickerEl("span", "option-detail", detail));
    group.appendChild(row);
  });

  const section = pickerSection("Models", group);
  if (candidates.length === 0) {
    section.appendChild(
      pickerEl(
        "div",
        "hint",
        compareState.granularities.size === 0
          ? "Pick at least one unit above."
          : "No model reads this unit of text."
      )
    );
  }
  return section;
}

function renderComparePicker(focusSelector) {
  const { root } = compareState;
  if (!root) return;

  root.textContent = "";
  const style = document.createElement("style");
  // picker.js's card, rows and type scale, plus this panel's own footer
  style.textContent = `${PICKER_STYLES}\n${COMPARE_STYLES}`;

  const overlay = pickerEl("div", "overlay");
  const panel = pickerEl("div", "panel");
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-modal", "true");
  panel.setAttribute("aria-label", "Compare multiple models");

  const head = pickerEl("div", "head");
  head.append(pickerEl("div", "title", "Compare models"));
  const close = pickerEl("button", "close", "×");
  close.type = "button";
  close.title = "Close";
  close.setAttribute("aria-label", "Close");
  close.addEventListener("click", hideComparePicker);
  head.appendChild(close);
  panel.appendChild(head);

  panel.appendChild(compareGranularitySection());
  panel.appendChild(pickerEl("hr", "rule"));
  panel.appendChild(compareModelSection());

  if (compareState.healthError) {
    panel.appendChild(
      pickerEl("div", "hint warn", "Backend not reachable — showing every model, availability unknown.")
    );
  }

  const footer = pickerEl("div", "footer");
  const count = compareState.selected.size;
  const cuts = compareCuts();
  // arm-runs, not ticked models: a model that can read both chosen cuts runs twice
  const runs = compareCandidates().filter((candidate) => compareState.selected.has(candidate.key))
    .reduce((total, candidate) => total + candidate.cuts.length, 0);
  footer.appendChild(
    pickerEl(
      "div",
      "footer-note",
      cuts.length === 0
        ? "Pick at least one unit."
        : count === 0
          ? "Pick at least one model."
          : `This page is prepared once and put through ${runs === 1 ? "one run" : `${runs} runs`} in turn. ` +
              `Nothing on the page changes; they are combined into one report on the Reports page.`
    )
  );
  // counts models, not runs; the note above gives the run count
  const run = pickerEl(
    "button",
    "run",
    count === 0
      ? "Run comparison"
      : cuts.length > 1
        ? `Compare ${count} ${count === 1 ? "model" : "models"}, both ways`
        : count > 1
          ? `Compare ${count} models`
          : "Run comparison"
  );
  run.type = "button";
  run.id = "simplify-compare-run";
  if (count === 0 || cuts.length === 0) run.disabled = true;
  run.addEventListener("click", startComparisonFromPicker);
  footer.appendChild(run);
  panel.appendChild(footer);

  panel.addEventListener("click", onCompareClick);
  overlay.appendChild(panel);
  root.append(style, overlay);
  compareState.card = panel;

  if (focusSelector) {
    const target = root.querySelector(focusSelector);
    if (target) target.focus();
  }
}

// --- interaction ------------------------------------------------------------

function onCompareClick(event) {
  if (!event.target.closest) return;

  const granularity = event.target.closest(".segment[data-compare-granularity]");
  if (granularity && !granularity.disabled) {
    const cut = granularity.dataset.compareGranularity;
    if (compareState.granularities.has(cut)) compareState.granularities.delete(cut);
    else compareState.granularities.add(cut);
    // drop only keys no chosen cut can run; the rest of the selection stays valid
    const runnable = new Set(compareCandidates().map((candidate) => candidate.key));
    Array.from(compareState.selected).forEach((key) => {
      if (!runnable.has(key)) compareState.selected.delete(key);
    });
    renderComparePicker(`[data-compare-granularity="${cut}"]`);
    return;
  }

  const model = event.target.closest(".option[data-compare-model]");
  if (model && !model.disabled) {
    const key = model.dataset.compareModel;
    if (compareState.selected.has(key)) compareState.selected.delete(key);
    else compareState.selected.add(key);
    renderComparePicker(`[data-compare-model="${key}"]`);
  }
}

function startComparisonFromPicker() {
  const models = compareCandidates()
    // panel order, not click order, so run order and arm letters follow the rows
    .filter((candidate) => compareState.selected.has(candidate.key))
    .map((candidate) => candidate.key);
  const cuts = compareCuts();
  hideComparePicker();
  if (models.length === 0 || cuts.length === 0) return;
  // content.js owns the run
  if (typeof onComparisonRequested === "function") onComparisonRequested(cuts, models);
}

function onCompareKeydown(event) {
  if (event.key === "Escape") hideComparePicker();
}

function onComparePointerDown(event) {
  if (!compareState.card) return;
  const path = typeof event.composedPath === "function" ? event.composedPath() : [];
  if (path.includes(compareState.card)) return;
  hideComparePicker();
}

// --- open / close -----------------------------------------------------------

let comparePriorOverflow = "";

function showComparePicker() {
  if (compareIsOpen()) {
    hideComparePicker();
    return;
  }
  const host = document.createElement("div");
  host.id = COMPARE_HOST_ID;
  // the observer skips this host by id (OWN_UI_IDS in content.js); translate="no" as
  // on the notice box
  host.setAttribute("translate", "no");
  compareState.host = host;
  compareState.root = host.attachShadow({ mode: "open" });
  compareState.selected = new Set();
  compareState.granularities = new Set();
  document.body.appendChild(host);

  document.addEventListener("keydown", onCompareKeydown, true);
  document.addEventListener("pointerdown", onComparePointerDown, true);
  comparePriorOverflow = document.documentElement.style.overflow;
  document.documentElement.style.overflow = "hidden";

  // preselect only the current granularity, not both: both cuts cost minutes per model
  readSelection().then((selection) => {
    if (!compareIsOpen()) return;
    if (compareState.granularities.size === 0) compareState.granularities.add(selection.granularity);
    renderComparePicker();
  });
  chrome.runtime.sendMessage({ cmd: "health" }, (resp) => {
    if (chrome.runtime.lastError) {
      compareState.healthError = chrome.runtime.lastError.message;
    } else if (resp && resp.ok) {
      compareState.health = resp.health;
      compareState.healthError = null;
    } else {
      compareState.healthError = (resp && resp.error) || "unknown error";
    }
    if (compareIsOpen()) renderComparePicker();
  });
}

function hideComparePicker() {
  document.removeEventListener("keydown", onCompareKeydown, true);
  document.removeEventListener("pointerdown", onComparePointerDown, true);
  document.documentElement.style.overflow = comparePriorOverflow;
  if (compareState.host) compareState.host.remove();
  compareState.host = null;
  compareState.root = null;
  compareState.card = null;
  compareState.health = null;
  compareState.healthError = null;
}
