// shared/active-run.js - one simplification run at a time, and the record of which run
// holds the lock.
//
// Concurrent runs share one batch queue per model on the backend, so both slow down and
// report timings that describe neither; the project measures run cost, so they are blocked.
//
// The lock lives in chrome.storage.local, not worker memory: MV3 kills an idle service
// worker after ~30 s, which would release it mid-run. Only the service worker writes it
// (background.js's claimActiveRun / releaseActiveRun / updateActiveRunProgress, serialized
// through one promise chain so two simultaneous claims can't both win).
//
// Loaded via importScripts() in background.js, as a content script alongside content.js
// (see manifest.json), and via <script> on extension pages; plain globals, not a module.
const ACTIVE_RUN_STORAGE_KEY = "simplifierActiveRun";

// Run kinds; both hold the same lock and differ only in how they are reported.
const ACTIVE_RUN_PAGE = "page";
const ACTIVE_RUN_COMPARE = "compare";

// The record holds:
//
//   kind        one of the two above
//   tabId       set by the service worker, not the claimant, so it can be trusted;
//               a record whose tab is gone is not a live run
//   sessionId   the content script's per-page session id (matches the History log entry)
//   url, title, host   what the run is *of*, for a reader looking at a different tab
//   startedAt   when the lock was taken
//   updatedAt   last progress report; see ACTIVE_RUN_STALE_MS below
//   done, total the run's own progress counts, or 0 before it has any
//   phase       the content script's session phase ("preflight" / "simplifying")
//   models      compare runs only: how many models the run is working through
//   modelIndex  compare runs only: which of them is running now, 1-based

// A record with no progress report for this long is treated as abandoned. A live run
// reports at least once a second (content.js re-renders its notice on a timer), so this
// only catches cases no release message covers: tab reloaded mid-run, content script
// threw, browser restarted. Without it a stuck lock could only be cleared by restarting
// the browser.
const ACTIVE_RUN_STALE_MS = 15000;

function activeRunIsStale(run, now) {
  if (!run) return true;
  const last = run.updatedAt || run.startedAt || 0;
  return (now || Date.now()) - last > ACTIVE_RUN_STALE_MS;
}

// Name of the run in the busy notice, toolbar menu and Home status line: page title,
// else host (a full URL is hard to recognise from another tab).
function activeRunLabel(run) {
  if (!run) return "";
  return run.title || run.host || run.url || "another page";
}

// Shared one-line summary so every surface words the state the same way.
function activeRunSummary(run) {
  if (!run) return null;
  const where = activeRunLabel(run);
  if (run.kind === ACTIVE_RUN_COMPARE) {
    const count = run.models || 0;
    // Progress counts runs, not models: a model compared at both granularities runs
    // twice. The wording still names models.
    const total = run.runs || count;
    const index = run.modelIndex || 0;
    const models = `${count} ${count === 1 ? "model" : "models"}`;
    return index > 0 && total > 0
      ? `Comparing ${models} on ${where} (${index} of ${total})`
      : `Comparing ${models} on ${where}`;
  }
  return `Simplifying ${where}`;
}
