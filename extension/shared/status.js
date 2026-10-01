// shared/status.js - live backend-status dashboard, rendered into the "Backend status"
// section of home.html and instructions.html. The failure-guidance page for a failed
// health check is shared/error.js / error.html.
const BACKEND_ORIGIN = "http://127.0.0.1:8000";
// /health poll interval and backoff ceiling. Bounded because unconditional 3s polling
// (~1200 req/h per open tab) dominated the backend access log:
//  - only while the page is visible (see syncPollingToVisibility); re-checks on return.
//  - each repeat of the same reading doubles the interval up to POLL_INTERVAL_MAX_MS;
//    any change resets it to POLL_INTERVAL_MS.
const POLL_INTERVAL_MS = 3000;
const POLL_INTERVAL_MAX_MS = 30000;
// fetch to a port nothing listens on can hang rather than refuse. Shorter than one poll
// interval, so a stalled backend reports on the same tick and checks don't pile up.
const HEALTH_TIMEOUT_MS = 2500;
// MODEL_STORAGE_KEY / DEFAULT_MODEL / getSelectedModel() come from
// shared/model-selection.js, loaded before this script by the pages that use it.

// Sole writer of the reachability box. `text` is error.html's sentence; `shortText` the
// dashboard's terse label (falls back to `text`).
function setLiveStatus(state, text, shortText) {
  // backoff keys off state + full text: short labels collide (e.g. two models both
  // read "Model not configured")
  recordReading(`${state}:${text}`);

  const statusBox = document.getElementById("dash-reachable");
  if (statusBox) {
    statusBox.classList.remove("status-checking", "status-down", "status-up");
    statusBox.classList.add(`status-${state}`);
    statusBox.textContent = shortText || text;
    return;
  }

  const box = document.getElementById("live-status");
  if (box) {
    box.classList.remove("status-checking", "status-down", "status-up");
    box.classList.add(`status-${state}`);
  }
  const textEl = document.getElementById("live-status-text");
  if (textEl) textEl.textContent = text;
}

// runs continuously; no "resolved, stop checking" state as on error.html
async function pollHealth() {
  let health = null;
  try {
    const res = await fetch(`${BACKEND_ORIGIN}/health`, {
      signal: healthTimeoutSignal(),
    });
    if (!res.ok) {
      setLiveStatus(
        "down",
        `Backend responded with HTTP ${res.status}`,
        `HTTP ${res.status}`,
      );
    } else {
      health = await res.json();
      if (!health.model_loaded) {
        setLiveStatus(
          "down",
          "Backend is up, waiting for the model to finish loading…",
          "Loading model…",
        );
      } else {
        const selectedModel = await getSelectedModel();
        const selectedAvailable =
          !Array.isArray(health.models_loaded) ||
          health.models_loaded.includes(selectedModel);
        if (!selectedAvailable) {
          setLiveStatus(
            "down",
            `Backend is up, but '${modelDisplayNameWithScope(selectedModel, health.model_names, health.granularities)}' isn't configured on it`,
            "Model not configured",
          );
        } else {
          setLiveStatus("up", "Backend is up and ready.", "Backend ready");
        }
      }
    }
  } catch (e) {
    const timedOut = e && e.name === "AbortError";
    setLiveStatus(
      "down",
      timedOut
        ? `Backend did not answer within ${Math.round(HEALTH_TIMEOUT_MS / 1000)}s`
        : "Backend not reachable",
      timedOut ? "No answer" : "Not reachable",
    );
  }

  updateDashboardFacts(health);
}

// plain timer: AbortSignal.timeout() isn't available everywhere, including the tests' jsdom
function healthTimeoutSignal() {
  const controller = new AbortController();
  setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS);
  return controller.signal;
}

function updateDashboardFacts(health) {
  const models = document.getElementById("dash-models");
  if (models) renderModelsLoaded(health);
  const selected = document.getElementById("dash-selected");
  if (!selected && !document.getElementById("dash-method")) return;
  getSelectedModel().then((model) => {
    const label = modelDisplayName(model, health && health.model_names);
    const method = document.getElementById("dash-method");
    const methodText = modelMethodLabel(
      model,
      health && health.methods,
      health && health.granularities,
    );
    const capable =
      health &&
      Array.isArray(health.audience_models) &&
      health.audience_models.includes(model);

    if (method) {
      if (capable) {
        getSelectedAudience().then((audience) => {
          const audienceName = audienceDisplayName(
            audience,
            health && health.audiences,
          );
          method.textContent = `${methodText} · for ${audienceName}`;
        });
      } else {
        method.textContent = methodText;
      }
    }

    if (!selected) return;
    renderSelectedModel(model, label, health);
  });
}

// Selected model row: readable name first, exact id on its own line (ids are long and
// would wrap awkwardly beside the name).
function renderSelectedModel(model, text, health) {
  const container = document.getElementById("dash-selected");
  container.textContent = "";

  const line = document.createElement("div");
  line.className = "model-inline";

  const label = document.createElement("span");
  label.className = "model-name-inline";
  label.textContent = text;
  line.appendChild(label);

  // scope, as in the loaded list: the two fine-tuned checkpoints share a name and differ
  // only in scope
  line.appendChild(modelScopeElement(model, health));

  const idEl = modelIdElement(model, health);
  if (idEl && idEl.textContent !== text) {
    const idWrap = document.createElement("span");
    idWrap.className = "model-id-inline";
    idWrap.appendChild(idEl);
    line.appendChild(idWrap);
  }

  container.appendChild(line);
}

// Muted scope span beside a model name, shared by both rows. `keys` may be several,
// for a row covering multiple backend keys served by one artifact.
function modelScopeElement(key, health, keys = null) {
  const granularities = health && health.granularities;
  const span = document.createElement("span");
  span.className = "model-scope";
  span.textContent = (keys || [key])
    .map((k) => modelScopeText(k, granularities))
    .join(" · ");
  return span;
}

// Exact id as reported by the backend: a link to ollama.com / huggingface.co for a
// published model, plain code for a local checkpoint directory. This page is where the
// verbatim ids are shown; pickers elsewhere use readable names.
function modelIdElement(key, health) {
  const id = health && health.model_names && health.model_names[key];
  if (!id) return null;
  const url = modelSourceUrl(id);
  const el = document.createElement(url ? "a" : "code");
  el.className = "model-id";
  el.textContent = id;
  if (url) {
    el.href = url;
    el.target = "_blank";
    // no window.opener, no referrer leaking the extension origin
    el.rel = "noopener noreferrer";
    el.title = `Open ${id} on ${new URL(url).host}`;
  }
  return el;
}

// One row per distinct loaded artifact, not per key. E.g. `llm_7b` and `llm_doc_7b` are
// one Ollama tag at two granularities (granularity selects a prompt template, see
// backend/main.py's MODEL_ENV_CONFIG); same for the 3B pair.
//
// Grouped by reported id, not picker family: the fine-tuned family's two granularities
// are different checkpoints (WikiLarge, D-Wikipedia) and stay two rows.
function groupLoadedModels(models, modelNames) {
  const groups = [];
  const byId = new Map();
  models.forEach((key) => {
    const id = (modelNames && modelNames[key]) || null;
    // keys without a reported id each get their own row
    const existing = id ? byId.get(id) : null;
    if (existing) {
      existing.keys.push(key);
      return;
    }
    const group = { id, keys: [key] };
    groups.push(group);
    if (id) byId.set(id, group);
  });
  return groups;
}

// Label the row with the sentence-scope key if present: its label names no scope
// ("LLM (Qwen2.5 7B)"), and the row's scope span already states it.
function groupLabelKey(group, granularities) {
  return (
    group.keys.find(
      (key) =>
        modelGranularity(key, granularities) ===
        GRANULARITY_SENTENCE_BY_SENTENCE,
    ) || group.keys[0]
  );
}

// One model per line: readable name plus exact id, unless the name already is the id.
function renderModelsLoaded(health) {
  const container = document.getElementById("dash-models");
  container.textContent = "";
  const models =
    health && Array.isArray(health.models_loaded) ? health.models_loaded : [];
  if (models.length === 0) {
    container.textContent = "none loaded";
    return;
  }

  const granularities = health && health.granularities;
  groupLoadedModels(models, health && health.model_names).forEach((group) => {
    const labelKey = groupLabelKey(group, granularities);
    const label = modelDisplayName(labelKey, health && health.model_names);
    if (!label) return;

    const row = document.createElement("div");
    row.className = "loaded-model-item";

    const name = document.createElement("span");
    name.className = "loaded-model-name";
    name.textContent = label;
    row.appendChild(name);

    // on every row: labels no longer state scope, and it distinguishes the two
    // fine-tuned checkpoints, which share a name
    row.appendChild(modelScopeElement(labelKey, health, group.keys));

    const idEl = modelIdElement(labelKey, health);
    if (idEl && idEl.textContent !== label) {
      const idWrap = document.createElement("span");
      idWrap.className = "model-id-subtle";
      idWrap.appendChild(idEl);
      row.appendChild(idWrap);
    }

    container.appendChild(row);
  });
}

// End-to-end check: a real, timed /simplify request (like background.js's
// checkBackendHealth stage 3) confirming the selected model produces output. Not run on
// the poll cadence, since it is a real inference call.
async function runVitalityCheck() {
  const vitalityEl = document.getElementById("dash-vitality");
  vitalityEl.textContent = "Checking…";
  const model = await getSelectedModel();
  const start = performance.now();
  try {
    const res = await fetch(`${BACKEND_ORIGIN}/simplify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: "This is a short connectivity test sentence.",
        model,
      }),
    });
    const elapsedMs = Math.round(performance.now() - start);
    if (!res.ok) {
      vitalityEl.textContent = `Failed (HTTP ${res.status})`;
      return;
    }
    const data = await res.json();
    if (
      !data ||
      typeof data.simplified !== "string" ||
      !data.simplified.trim()
    ) {
      vitalityEl.textContent = "Failed (no usable output)";
      return;
    }
    vitalityEl.textContent = `Healthy — responded in ${elapsedMs}ms`;
  } catch (e) {
    vitalityEl.textContent = `Failed (${e.message || e})`;
  }
}

const refreshVitalityButton = document.getElementById("dashboard-refresh");
if (refreshVitalityButton) {
  refreshVitalityButton.addEventListener("click", () => {
    runVitalityCheck();
  });
}

// --- when to poll ----------------------------------------------------------
// explicit `polling` flag: a stubbed setInterval can return timer id 0
let pollTimer = null;
let polling = false;
let pollDelayMs = POLL_INTERVAL_MS;
let lastReading = null;

function armPolling(delayMs) {
  pollDelayMs = delayMs;
  clearInterval(pollTimer);
  pollTimer = setInterval(pollHealth, delayMs);
  polling = true;
}

function stopPolling() {
  clearInterval(pollTimer);
  pollTimer = null;
  polling = false;
}

// Same reading widens the interval, a new one resets it. No-op while hidden, so a check
// in flight when the tab was hidden can't re-arm the stopped timer.
function recordReading(reading) {
  if (!polling) return;
  if (reading !== lastReading) {
    lastReading = reading;
    if (pollDelayMs !== POLL_INTERVAL_MS) armPolling(POLL_INTERVAL_MS);
    return;
  }
  const widened = Math.min(pollDelayMs * 2, POLL_INTERVAL_MAX_MS);
  if (widened !== pollDelayMs) armPolling(widened);
}

// Also called once at load: a tab that opens hidden (restored session, background tab)
// never fires visibilitychange.
function syncPollingToVisibility() {
  if (document.visibilityState === "hidden") {
    stopPolling();
    return;
  }
  if (polling) return;
  // back on screen: reset backoff and re-check immediately
  lastReading = null;
  armPolling(POLL_INTERVAL_MS);
  pollHealth();
}

document.addEventListener("visibilitychange", syncPollingToVisibility);
syncPollingToVisibility();
if (document.getElementById("dash-vitality")) {
  runVitalityCheck();
}
