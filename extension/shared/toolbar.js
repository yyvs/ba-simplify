// shared/toolbar.js - nav bar at the top of every extension page (home, history,
// instructions, reports, error): page links, reload, and the model selection.
// Plain <script> (no build step); requires shared/model-labels.js,
// shared/model-selection.js and shared/demo-pages.js first. Pages live flat at the
// extension root, so paths are bare filenames.
//
// The selection button shows the active configuration ("Ollama Qwen2.5 · 7B · Sections ·
// Children · 8–12") and opens the same panel (shared/picker.js) as the icon's menu.
//
// "Clear cache" acts on the backend, so it lives in the bar rather than on one page.
const TOOLBAR_BACKEND_ORIGIN = "http://127.0.0.1:8000"; // matches background.js's BACKEND_ORIGIN
// appended by background.js to the homepage URL when the panel can't be injected;
// must match PICKER_FALLBACK_HASH there.
const TOOLBAR_PICKER_HASH = "#picker";

const TOOLBAR_PAGES = [
  { path: "instructions.html", label: "Instructions" },
  { path: "history.html", label: "History" },
  { path: "reports.html", label: "Reports" },
];

const TOOLBAR_CLEAR_CACHE_LABEL = "Clear cache";
const TOOLBAR_CLEAR_CACHE_TITLE =
  "Empties the backend's result cache, so text simplified before is sent to the model " +
  "again. Leaves the History log alone.";
const TOOLBAR_FEEDBACK_MS = 2500;
const TOOLBAR_FEEDBACK_ERROR_MS = 5000;

function resetClearCacheButton(button, afterMs) {
  setTimeout(() => {
    button.textContent = TOOLBAR_CLEAR_CACHE_LABEL;
    button.title = TOOLBAR_CLEAR_CACHE_TITLE;
    button.disabled = false;
  }, afterMs);
}

async function clearBackendCache(button) {
  if (
    !confirm(
      "Clear the simplification cache? Text simplified before will be sent to the " +
        "model again instead of being served from the cache. The History log is not " +
        "affected.",
    )
  )
    return;
  button.disabled = true;
  button.textContent = "Clearing…";
  try {
    const res = await fetch(`${TOOLBAR_BACKEND_ORIGIN}/cache/clear`, {
      method: "POST",
    });
    if (!res.ok) {
      button.textContent = "Not cleared";
      button.title = `The backend answered ${res.status}. Nothing was cleared.`;
      resetClearCacheButton(button, TOOLBAR_FEEDBACK_ERROR_MS);
      return;
    }
    const data = await res.json();
    const cleared =
      data && typeof data.cleared === "number" ? data.cleared : null;
    button.textContent = "Cache cleared";
    button.title =
      cleared === null
        ? "The backend's cache was cleared."
        : `${cleared} cached ${cleared === 1 ? "simplification" : "simplifications"} dropped.`;
    resetClearCacheButton(button, TOOLBAR_FEEDBACK_MS);
  } catch (e) {
    button.textContent = "Not cleared";
    button.title =
      "The backend isn't reachable, so there was nothing to clear.";
    resetClearCacheButton(button, TOOLBAR_FEEDBACK_ERROR_MS);
  }
}

function buildToolbar() {
  const bar = document.createElement("div");
  bar.id = "ext-toolbar";

  const title = document.createElement("a");
  title.className = "ext-toolbar-title";
  title.href = chrome.runtime.getURL("home.html");
  // icon32 for an 18px slot: 16 is soft on 2x displays, 48 is needless bytes
  const logo = document.createElement("img");
  logo.className = "ext-toolbar-logo";
  logo.src = chrome.runtime.getURL("icons/icon32.png");
  // decorative: the text beside it names the link
  logo.alt = "";
  title.append(logo, document.createTextNode("Simplify"));

  const nav = document.createElement("nav");
  const currentPath = location.pathname;
  TOOLBAR_PAGES.forEach(({ path, label }) => {
    const a = document.createElement("a");
    a.href = chrome.runtime.getURL(path);
    a.textContent = label;
    if (currentPath === `/${path}`) a.classList.add("active");
    nav.appendChild(a);
  });

  const demoBtn = document.createElement("button");
  demoBtn.type = "button";
  demoBtn.textContent = "Demo";
  demoBtn.addEventListener("click", () => {
    chrome.runtime.sendMessage({ cmd: "openDemoPage", file: "index.html" });
  });
  nav.appendChild(demoBtn);

  // /health, if reachable, for the served model name (e.g. "Ollama Qwen2.5 · 7B")
  // instead of the generic fallback
  let toolbarHealth = null;

  const pickerLabel = document.createElement("span");
  pickerLabel.className = "ext-toolbar-model-label";
  pickerLabel.textContent = "Simplifies with:";

  const pickerBtn = document.createElement("button");
  pickerBtn.type = "button";
  pickerBtn.id = "ext-toolbar-picker";
  pickerBtn.className = "ext-toolbar-picker";
  pickerBtn.title = "Change model, granularity and audience";
  pickerBtn.setAttribute("aria-haspopup", "dialog");
  pickerBtn.textContent = "Reading your selection…";
  pickerBtn.addEventListener("click", (event) => {
    // keep the picker's outside-click dismissal from seeing the opening click
    event.stopPropagation();
    showModelPicker();
  });

  // re-rendered on any selection change (panel, other pages, Error page switcher)
  function refreshPickerButton() {
    return readSelection().then((selection) => {
      pickerBtn.textContent = selectionSummary(
        selection.family,
        selection.granularity,
        selection.audience,
        toolbarHealth,
      );
      return selection;
    });
  }

  // Red: discards the backend result cache irrecoverably; regenerating on the
  // prompted-LLM path takes minutes.
  // Feedback on the button itself (not every page has a notice box); the count goes in
  // the title so the label doesn't change width.
  const clearCacheBtn = document.createElement("button");
  clearCacheBtn.type = "button";
  clearCacheBtn.id = "ext-toolbar-clear-cache";
  clearCacheBtn.className = "ext-toolbar-danger";
  clearCacheBtn.textContent = TOOLBAR_CLEAR_CACHE_LABEL;
  clearCacheBtn.title = TOOLBAR_CLEAR_CACHE_TITLE;
  clearCacheBtn.addEventListener("click", () =>
    clearBackendCache(clearCacheBtn),
  );

  const reloadBtn = document.createElement("button");
  reloadBtn.textContent = "↻";
  reloadBtn.title = "Reload this page";
  reloadBtn.setAttribute("aria-label", "Reload this page");
  reloadBtn.addEventListener("click", () => location.reload());

  // keep label and button together when the bar wraps
  const pickerField = document.createElement("span");
  pickerField.className = "ext-toolbar-field";
  pickerField.append(pickerLabel, pickerBtn);

  bar.append(title, nav, clearCacheBtn, pickerField, reloadBtn);
  document.body.prepend(bar);

  refreshPickerButton();

  // best-effort: static labels if the backend is unreachable
  fetch(`${TOOLBAR_BACKEND_ORIGIN}/health`)
    .then((r) => r.json())
    .then((health) => {
      toolbarHealth = health;
      refreshPickerButton();
    })
    .catch(() => {});

  // panel, icon menu and Error page all write this storage
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local") return;
    const watched = [
      MODEL_STORAGE_KEY,
      GRANULARITY_STORAGE_KEY,
      AUDIENCE_STORAGE_KEY,
    ];
    if (watched.some((key) => key in changes)) refreshPickerButton();
  });

  // Icon menu used where the panel can't be injected (chrome:// or extension pages):
  // background.js opens the homepage with this hash instead.
  function openPickerFromHash() {
    if (location.hash !== TOOLBAR_PICKER_HASH) return;
    // drop the hash so a reload doesn't reopen it
    history.replaceState(null, "", location.pathname);
    pickerBtn.click();
  }
  openPickerFromHash();
  // an already-open tab refocused with the hash navigates without reloading
  window.addEventListener("hashchange", openPickerFromHash);
}

buildToolbar();
