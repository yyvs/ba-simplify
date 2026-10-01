// shared/picker.js - model picker panel (model, granularity, audience). On web pages
// the content script injects it (manifest.json's content_scripts) and the toolbar
// icon's right-click menu opens it; on extension pages the shared toolbar button does.
// One file and one storage for both surfaces.
//
// In-page DOM, not a browser-action popup: chrome.action.onClicked only fires without
// a default_popup, and left-click is the primary toggle action.
//
// Rendered in a shadow root so page CSS can't reach in and the extension's own
// MutationObserver never sees the subtree as text to simplify (only the host is
// filtered).
//
// Requires shared/model-labels.js and shared/model-selection.js (readSelection/
// writeSelection) to be loaded first.
const PICKER_HOST_ID = "simplify-model-picker";

const pickerState = {
  host: null,
  root: null,
  selection: null,
  // pointerdown outside the card (i.e. on the backdrop) dismisses
  card: null,
  // /health response: loaded keys, their ids/tags, the authoritative audience list.
  // null = not yet asked or unreachable.
  health: null,
  healthError: null,
  // whether an output-affecting setting changed while open; see hideModelPicker()
  changed: false,
};

const PICKER_STYLES = `
  /* A shadow root stops the page's *selectors* reaching in, but not its *inheritance*:
     inherited properties (text-transform, letter-spacing, line-height, direction, ...)
     pass from the host element into the shadow tree, and a page that sets them with
     !important on a selector matching the host -- "* { letter-spacing: 2px !important }"
     is a real pattern -- wins over a normal declaration here. Hence !important on the
     reset, which as a :host rule also outranks the page's element/universal selectors
     at equal importance. Verified against a page that fights injected UI on all of
     these. */
  :host { all: initial !important; }
  * { box-sizing: border-box; }
  /* Full screen: the picker takes over the viewport rather than perching in a corner.
     One consequence worth stating -- everything outside the card is backdrop, so a
     pointerdown there dismisses (see onPickerPointerDown), and the page underneath
     can't be clicked by accident while a setting is being changed. */
  .overlay {
    position: fixed;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 32px 20px;
    overflow-y: auto;
    background: rgba(8,8,8,0.82);
    /* backdrop-filter is a nicety, not load-bearing: where it isn't supported the
       backdrop is still an 82% wash over the page. */
    backdrop-filter: blur(3px);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 14px;
    line-height: 1.45;
    text-align: left;
    /* Top of the layering scale, above every other floating surface the extension
       has -- the same value shared/layers.css publishes as --layer-picker, written
       out as a literal here because these styles live in a shadow root injected
       into arbitrary pages, which never load that file. */
    z-index: 2147483647;
  }
  .panel {
    width: 100%;
    max-width: 520px;
    /* the card grows with its content and scrolls the overlay, not itself, so a long
       list never ends up with two nested scrollbars */
    padding: 22px 24px 24px;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    /* opaque, not translucent: this is a settings surface that sits over arbitrary page
       content, and even 3% transparency ghosts the text underneath through the rows. */
    background: #141414;
    color: #fff;
    box-shadow: 0 24px 64px rgba(0,0,0,0.5);
  }
  .head { display: flex; align-items: baseline; gap: 8px; margin-bottom: 18px; }
  .title { flex: 1; font-size: 20px; font-weight: 600; letter-spacing: -0.01em; }
  .close {
    flex: none;
    width: 30px; height: 30px;
    padding: 0;
    border: 0;
    border-radius: 6px;
    background: transparent;
    color: rgba(255,255,255,0.6);
    font: inherit;
    font-size: 19px;
    line-height: 1;
    cursor: pointer;
  }
  .close:hover { background: rgba(255,255,255,0.12); color: #fff; }
  /* the three groups are named rather than merely spaced apart: everything is on one
     surface now, and without headings a reader has to infer where a control's scope
     ends. Small and quiet, since they are signposts, not content. */
  .section-label {
    margin-bottom: 8px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.5);
  }
  .section + .section { margin-top: 20px; }
  /* every choice in the panel is a radio: a filled dot for the selection, a ring for
     the rest, in one column so the eye can scan down it. */
  .option {
    display: grid;
    grid-template-columns: 18px 1fr;
    gap: 1px 10px;
    width: 100%;
    margin: 0;
    padding: 9px 10px;
    border: 0;
    border-radius: 8px;
    background: transparent;
    color: inherit;
    font: inherit;
    text-align: left;
    cursor: pointer;
  }
  .option:hover:not([disabled]) { background: rgba(255,255,255,0.07); }
  /* "subtle background rather than a heavy border": the selected row reads as filled,
     not outlined, so the list keeps one vertical rhythm. */
  .option[aria-checked="true"] { background: rgba(70,136,241,0.16); }
  .option[disabled] { cursor: default; opacity: 0.45; }
  .option:focus-visible { outline: 2px solid #4688f1; outline-offset: 1px; }
  .mark { align-self: start; color: #4688f1; font-size: 11px; line-height: 1.6; }
  .option[aria-checked="false"] .mark { color: rgba(255,255,255,0.4); }
  .option-title { font-size: 14px; font-weight: 600; }
  /* method and distinguishing detail on one line, so a model is two lines and the
     list of four stays scannable in a single glance. */
  .option-detail { grid-column: 2; font-size: 12.5px; color: rgba(255,255,255,0.58); }
  .option-detail.is-id {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    overflow-wrap: anywhere;
  }
  /* the audience rows carry a label and nothing else, so they sit tighter than a model */
  .option.compact { padding: 7px 10px; }
  .option.compact .option-title { font-weight: 400; }
  .rule { margin: 20px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.14); }
  /* two options, so a segmented control rather than a select: both labels visible,
     one click to change, and it reads as one control because it is one box. */
  .segments {
    display: flex;
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 8px;
    overflow: hidden;
  }
  .segment {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    flex: 1;
    padding: 10px 12px;
    border: 0;
    background: transparent;
    color: #eee;
    font: inherit;
    font-size: 13px;
    cursor: pointer;
  }
  .segment + .segment { border-left: 1px solid rgba(255,255,255,0.22); }
  .segment:hover:not([disabled]) { background: rgba(255,255,255,0.07); }
  .segment[aria-checked="true"] { background: rgba(70,136,241,0.16); }
  .segment[disabled] { cursor: default; opacity: 0.4; }
  .segment:focus-visible { outline: 2px solid #4688f1; outline-offset: -2px; }
  .segment .mark { font-size: 11px; }
  .hint { margin-top: 8px; font-size: 12px; color: rgba(255,255,255,0.5); }
  .hint.warn { color: #f3b562; }
`;

// radio marks, used by all three groups
const PICKER_MARK_ON = "●";
const PICKER_MARK_OFF = "○";

// --- rendering --------------------------------------------------------------

function pickerEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

// heading plus its control
function pickerSection(label, ...children) {
  const section = pickerEl("div", "section");
  section.append(pickerEl("div", "section-label", label), ...children);
  return section;
}

// One radio row: mark, title, optional detail line. `detailClass` applies code styling
// for the comparison model, whose detail is a repo id.
function pickerOption({ selected, title, detail, detailClass, disabled, dataset, compact }) {
  const row = pickerEl("button", compact ? "option compact" : "option");
  row.type = "button";
  row.setAttribute("role", "radio");
  row.setAttribute("aria-checked", String(selected));
  if (disabled) row.disabled = true;
  Object.keys(dataset || {}).forEach((key) => (row.dataset[key] = dataset[key]));
  row.append(
    pickerEl("span", "mark", selected ? PICKER_MARK_ON : PICKER_MARK_OFF),
    pickerEl("span", "option-title", title)
  );
  if (detail !== undefined) {
    row.appendChild(pickerEl("span", detailClass ? `option-detail ${detailClass}` : "option-detail", detail));
  }
  return row;
}

// A family with nothing loaded is shown disabled, not hidden, so it reads as "not
// started" rather than nonexistent.
function pickerModelRow(familyId, selection, health, modelsLoaded) {
  const names = health && health.model_names;
  const available = modelFamilyAvailable(familyId, modelsLoaded);
  const isId = modelFamilyDetailIsId(familyId);
  // "Fine-tuned seq2seq · Custom-trained for simplification"; an id detail stands alone
  // in code style
  const detail = available
    ? isId
      ? modelFamilyDetail(familyId, names)
      : `${modelFamilyMethod(familyId)} · ${modelFamilyDetail(familyId, names)}`
    : `${modelFamilyMethod(familyId)} · not loaded on the backend`;

  return pickerOption({
    selected: familyId === selection.family,
    title: modelFamilyTitle(familyId, names),
    detail,
    detailClass: available && isId ? "is-id" : null,
    disabled: !available,
    dataset: { family: familyId },
  });
}

// Two segments, each disabled with a reason where the combination can't run. Marks the
// effective granularity, not the stored intent, in case the intent fell back.
function pickerGranularitySection(selection, modelsLoaded) {
  const segments = pickerEl("div", "segments");
  segments.setAttribute("role", "radiogroup");
  segments.setAttribute("aria-label", "Granularity");

  const reasons = [];
  GRANULARITY_CHOICES.forEach((choice) => {
    const status = familyGranularityStatus(selection.family, choice.value, modelsLoaded);
    const selected = choice.value === selection.granularity;
    const btn = pickerEl("button", "segment");
    btn.type = "button";
    btn.setAttribute("role", "radio");
    btn.setAttribute("aria-checked", String(selected));
    btn.dataset.granularity = choice.value;
    btn.append(
      pickerEl("span", "mark", selected ? PICKER_MARK_ON : PICKER_MARK_OFF),
      pickerEl("span", null, choice.label)
    );
    if (!status.available) {
      btn.disabled = true;
      btn.title = `${choice.label} — ${status.reason}`;
      reasons.push(`${choice.label}: ${status.reason}`);
    }
    segments.appendChild(btn);
  });

  const section = pickerSection("Granularity", segments);
  // reasons shown below the control, not only in tooltips
  reasons.forEach((reason) => section.appendChild(pickerEl("div", "hint", reason)));
  return section;
}

// inline list rather than a select, so all options are visible
function pickerAudienceSection(selection, health) {
  const audiences =
    health && Array.isArray(health.audiences) && health.audiences.length ? health.audiences : FALLBACK_AUDIENCES;
  const group = pickerEl("div", "options");
  group.setAttribute("role", "radiogroup");
  group.setAttribute("aria-label", "Audience");
  audiences.forEach((audience) => {
    group.appendChild(
      pickerOption({
        selected: audience.value === selection.audience,
        title: audience.label,
        dataset: { audience: audience.value },
        compact: true,
      })
    );
  });
  return pickerSection("Audience", group);
}

function renderPicker(focusSelector) {
  const { root, selection, health, healthError } = pickerState;
  if (!root || !selection) return;
  const modelsLoaded = health && Array.isArray(health.models_loaded) ? health.models_loaded : null;

  root.textContent = "";
  const style = document.createElement("style");
  style.textContent = PICKER_STYLES;

  const overlay = pickerEl("div", "overlay");
  const panel = pickerEl("div", "panel");
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-modal", "true");
  panel.setAttribute("aria-label", "Simplification settings");

  const head = pickerEl("div", "head");
  head.append(pickerEl("div", "title", "Simplification"));
  const close = pickerEl("button", "close", "×");
  close.type = "button";
  close.title = "Close";
  close.setAttribute("aria-label", "Close");
  close.addEventListener("click", hideModelPicker);
  head.appendChild(close);
  panel.appendChild(head);

  const models = pickerEl("div", "options");
  models.setAttribute("role", "radiogroup");
  models.setAttribute("aria-label", "Model");
  MODEL_FAMILY_ORDER.forEach((familyId) => {
    models.appendChild(pickerModelRow(familyId, selection, health, modelsLoaded));
  });
  panel.appendChild(pickerSection("Model", models));

  panel.appendChild(pickerEl("hr", "rule"));
  panel.appendChild(pickerGranularitySection(selection, modelsLoaded));
  // Audience only for prompted families; the backend drops it for seq2seq checkpoints.
  if (modelFamilyHonoursAudience(selection.family, health && health.audience_models)) {
    panel.appendChild(pickerAudienceSection(selection, health));
  }
  // still usable without /health, but availability is unknown
  if (healthError) {
    panel.appendChild(
      pickerEl("div", "hint warn", "Backend not reachable — showing every model, availability unknown.")
    );
  }

  panel.addEventListener("click", onPickerClick);
  overlay.appendChild(panel);
  root.append(style, overlay);
  pickerState.card = panel;

  if (focusSelector) {
    const target = root.querySelector(focusSelector);
    if (target) target.focus();
  }
}

// --- interaction ------------------------------------------------------------
// Each click persists immediately (no Save/apply step); the panel stays open.

// Each option carries its dimension as a data attribute, also used to refocus after redraw.
const PICKER_DIMENSIONS = [
  { selector: ".option[data-family]", key: "family", attr: "family" },
  { selector: ".segment[data-granularity]", key: "granularity", attr: "granularity" },
  { selector: ".option[data-audience]", key: "audience", attr: "audience" },
];

function onPickerClick(event) {
  if (!event.target.closest) return;
  for (const { selector, key, attr } of PICKER_DIMENSIONS) {
    const control = event.target.closest(selector);
    if (!control || control.disabled) continue;
    const value = control.dataset[attr];
    selectPickerValues({ [key]: value }, `[data-${attr}="${value}"]`);
    return;
  }
}

function selectPickerValues(next, focusSelector) {
  const modelsLoaded =
    pickerState.health && Array.isArray(pickerState.health.models_loaded) ? pickerState.health.models_loaded : null;
  const before = pickerState.selection;
  writeSelection(next, modelsLoaded).then(() => {
    // re-read: writeSelection may resolve the family's remembered audience or fall back
    // to another granularity
    readSelection().then((selection) => {
      // compare resolved values: re-picking the current model or an unserved
      // granularity resolves to the same key and isn't a change
      if (before && (before.model !== selection.model || before.audience !== selection.audience)) {
        pickerState.changed = true;
      }
      pickerState.selection = selection;
      renderPicker(focusSelector);
    });
  });
}

function onPickerKeydown(event) {
  if (event.key === "Escape") {
    hideModelPicker();
  }
}

// Pointerdown outside the card (on the backdrop, inside the same shadow tree) dismisses,
// so test against the card, not the host. composedPath() because event.target is
// retargeted to the host outside the shadow tree.
function onPickerPointerDown(event) {
  if (!pickerState.card) return;
  const path = typeof event.composedPath === "function" ? event.composedPath() : [];
  if (path.includes(pickerState.card)) return;
  hideModelPicker();
}

// --- open / close -----------------------------------------------------------

// page's <html> overflow, restored on close
let pickerPriorOverflow = "";

function modelPickerIsOpen() {
  return !!pickerState.host && pickerState.host.isConnected;
}

// toggles: calling while open closes it
function showModelPicker() {
  if (modelPickerIsOpen()) {
    hideModelPicker();
    return;
  }
  const host = document.createElement("div");
  host.id = PICKER_HOST_ID;
  // the observer skips this host by id (OWN_UI_IDS in content.js); translate="no" as
  // on the notice box
  host.setAttribute("translate", "no");
  pickerState.host = host;
  pickerState.root = host.attachShadow({ mode: "open" });
  document.body.appendChild(host);

  document.addEventListener("keydown", onPickerKeydown, true);
  document.addEventListener("pointerdown", onPickerPointerDown, true);
  // block scrolling the hidden page; saved and restored in case the page sets its own
  pickerPriorOverflow = document.documentElement.style.overflow;
  document.documentElement.style.overflow = "hidden";

  readSelection().then((selection) => {
    pickerState.selection = selection;
    renderPicker();
  });
  // draw from storage first, redraw when /health answers
  chrome.runtime.sendMessage({ cmd: "health" }, (resp) => {
    if (chrome.runtime.lastError) {
      pickerState.healthError = chrome.runtime.lastError.message;
    } else if (resp && resp.ok) {
      pickerState.health = resp.health;
      pickerState.healthError = null;
    } else {
      pickerState.healthError = (resp && resp.error) || "unknown error";
    }
    if (modelPickerIsOpen()) renderPicker();
  });
}

function hideModelPicker() {
  document.removeEventListener("keydown", onPickerKeydown, true);
  document.removeEventListener("pointerdown", onPickerPointerDown, true);
  document.documentElement.style.overflow = pickerPriorOverflow;
  if (pickerState.host) pickerState.host.remove();
  pickerState.host = null;
  pickerState.root = null;
  pickerState.card = null;
  pickerState.health = null;
  pickerState.healthError = null;

  // A change leaves existing page text from the old model. The page owner handles it
  // (content.js's onPickerSelectionCommitted reloads) on close rather than per click, so
  // picking a model and then its audience is one adjustment.
  const changed = pickerState.changed;
  pickerState.changed = false;
  if (changed && typeof onPickerSelectionCommitted === "function") {
    onPickerSelectionCommitted();
  }
}
