// shared/error.js - troubleshooting page (error.html) for a failed backend health check,
// a toolbar click on an unsupported page, or a tab that predates the extension (re)load.
// Content depends on the failure stage (see background.js's checkBackendHealth /
// action.onClicked / openDemoPage); shared/status.js is the general dashboard.
const BACKEND_ORIGIN = "http://127.0.0.1:8000";
const POLL_INTERVAL_MS = 3000;
const STORAGE_KEY = "error"; // matches the key background.js's openInfoTab writes to
// getSelectedModel() / selectModelKey() come from shared/model-selection.js.

const STAGES = [
  "reachable",
  "model_loaded",
  "model_unavailable",
  "model_response",
  "unsupported_page",
  "content_script_unreachable",
  "demo_unreachable",
];
// stages unrelated to backend health: no live status box or polling
const NON_BACKEND_STAGES = new Set(["unsupported_page", "content_script_unreachable", "demo_unreachable"]);

let pollTimer = null;
let currentStage = null;
let elapsedTimer = null;
let elapsedStartMs = null;

// Elapsed-time counter while the model loads; the backend reports no load progress.
function startElapsedTimer() {
  clearInterval(elapsedTimer);
  elapsedStartMs = performance.now();
  const el = document.getElementById("model-loaded-elapsed");
  const tick = () => {
    el.textContent = `Waiting ${Math.round((performance.now() - elapsedStartMs) / 1000)}s since this page opened…`;
  };
  tick();
  elapsedTimer = setInterval(tick, 1000);
}

function stopElapsedTimer() {
  clearInterval(elapsedTimer);
}

function showStage(stage, error) {
  const resolvedStage = STAGES.includes(stage) ? stage : "reachable";
  currentStage = resolvedStage;
  STAGES.forEach((s) => {
    document.getElementById(`stage-${s}`).classList.toggle("active", s === resolvedStage);
  });
  if (error) {
    document.getElementById("raw-error-details").style.display = "";
    document.getElementById("raw-error").textContent = error;
  } else {
    document.getElementById("raw-error-details").style.display = "none";
  }

  if (resolvedStage === "model_loaded") {
    startElapsedTimer();
  } else {
    stopElapsedTimer();
  }
  if (resolvedStage !== "model_unavailable") {
    document.getElementById("model-unavailable-actions").textContent = "";
  }

  const liveStatus = document.getElementById("live-status");
  clearInterval(pollTimer);
  if (NON_BACKEND_STAGES.has(resolvedStage)) {
    liveStatus.style.display = "none";
  } else {
    liveStatus.style.display = "";
    document.getElementById("subtitle").textContent = "Checking what's wrong…";
    pollTimer = setInterval(pollHealth, POLL_INTERVAL_MS);
    pollHealth();
  }
}

function setLiveStatus(state, text) {
  const box = document.getElementById("live-status");
  box.classList.remove("status-checking", "status-down", "status-up");
  box.classList.add(`status-${state}`);
  document.getElementById("live-status-text").textContent = text;
}

// Polls /health until reachable, loaded and the selected model available, then stops
// (unlike shared/status.js, which keeps polling).
async function pollHealth() {
  let health = null;
  try {
    const res = await fetch(`${BACKEND_ORIGIN}/health`);
    if (!res.ok) {
      setLiveStatus("down", `Backend responded with HTTP ${res.status}`);
    } else {
      health = await res.json();
      if (!health.model_loaded) {
        setLiveStatus("down", "Backend is up, waiting for the model to finish loading…");
      } else {
        const selectedModel = await getSelectedModel();
        const selectedAvailable =
          !Array.isArray(health.models_loaded) || health.models_loaded.includes(selectedModel);
        if (!selectedAvailable) {
          setLiveStatus(
            "down",
            `Backend is up, but '${modelDisplayNameWithScope(selectedModel, health.model_names, health.granularities)}' isn't configured on it`
          );
          if (currentStage === "model_unavailable") updateModelUnavailableActions(health, selectedModel);
        } else {
          setLiveStatus("up", "Backend is up and ready — you can close this tab and try again.");
          document.getElementById("subtitle").textContent = "All good — the extension should work now.";
          clearInterval(pollTimer);
          stopElapsedTimer();
        }
      }
    }
  } catch (e) {
    setLiveStatus("down", "Backend not reachable yet");
  }
}

// "model_unavailable" stage: one-click switch to a model the backend has loaded.
function updateModelUnavailableActions(health, selectedModel) {
  const container = document.getElementById("model-unavailable-actions");
  container.textContent = "";
  const available = Array.isArray(health.models_loaded) ? health.models_loaded.filter((k) => k !== selectedModel) : [];
  if (available.length === 0) return; // this stage only fires when at least one other model is loaded
  const target = available[0];
  const btn = document.createElement("button");
  btn.id = "model-unavailable-switch";
  btn.className = "status-btn";
  btn.textContent = `Switch to ${modelDisplayNameWithScope(target, health.model_names, health.granularities)} now`;
  btn.addEventListener("click", () => {
    btn.disabled = true;
    btn.textContent = "Switching…";
    // selectModelKey, not a bare storage write, so the granularity intent stays in sync
    selectModelKey(target).then(pollHealth);
  });
  container.appendChild(btn);
}

async function render() {
  const result = await chrome.storage.local.get(STORAGE_KEY);
  const info = result[STORAGE_KEY] || {};
  showStage(info.stage, info.error || "");
}

// An open error tab is refocused, not reloaded (background.js's openInfoTab), so
// watch storage for new info.
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes[STORAGE_KEY]) {
    showStage(changes[STORAGE_KEY].newValue?.stage, changes[STORAGE_KEY].newValue?.error || "");
  }
});

render();
