// shared/model-selection.js - the stored simplification selection and the rules for
// resolving it. Loaded via importScripts() in background.js, <script> on extension pages,
// and as a content script alongside content.js (see manifest.json); plain globals, not a
// module. Requires shared/model-labels.js to be loaded first.
//
// Written by three UIs (in-page picker shared/picker.js, the extension pages' toolbar,
// the Error page's "switch to an available model" button). Stored keys:
//
//   simplifierModel          resolved backend key ("llm_doc_7b"). Authoritative: sent to
//                            /simplify, checked against /health, recorded in history;
//                            family and granularity shown in UIs are derived from it.
//   simplifierGranularity    granularity *intent* (one of GRANULARITY_CHOICES). Needed
//                            because a family without a (loaded) document-scope model,
//                            e.g. the comparison checkpoint, resolves to its sentence key;
//                            the intent restores document scope when switching back.
//   simplifierAudienceByModel  per-family audience, restored when returning to a
//                            prompted model.
//   simplifierAudience       audience the next request carries. Separate key because
//                            every request path reads it and the backend echoes it back.
const MODEL_STORAGE_KEY = "simplifierModel";
const DEFAULT_MODEL = "online";
const GRANULARITY_STORAGE_KEY = "simplifierGranularity";
const AUDIENCE_BY_FAMILY_STORAGE_KEY = "simplifierAudienceByModel";

// Renamed model keys: "llm" predates the "llm_7b"/"llm_3b" split, "local" was renamed to
// "finetuned" (see model-labels.js). Stored preferences survive extension updates, so an
// old key would otherwise get a 400 on every request. Keep indefinitely.
const RENAMED_MODEL_KEYS = { llm: "llm_7b", local: "finetuned" };

// Old granularity intent values ("sentence"/"document"), mapped to the current vocabulary
// (shared/model-labels.js); otherwise the stored intent would be dropped.
const RENAMED_GRANULARITIES = {
  sentence: GRANULARITY_SENTENCE_BY_SENTENCE,
  document: GRANULARITY_WHOLE_SECTIONS,
};

// --- reading ----------------------------------------------------------------

// Stored model key, migrated and validated; an unknown (removed) key falls back to the
// default instead of 400ing every request.
async function getSelectedModel() {
  const result = await chrome.storage.local.get(MODEL_STORAGE_KEY);
  const stored = result[MODEL_STORAGE_KEY];
  if (!stored) return DEFAULT_MODEL;

  const renamed = RENAMED_MODEL_KEYS[stored];
  if (renamed) {
    // persist the migration so every UI reading this key agrees
    await chrome.storage.local.set({ [MODEL_STORAGE_KEY]: renamed });
    return renamed;
  }
  return MODEL_KEYS.includes(stored) ? stored : DEFAULT_MODEL;
}

// Sent on every request regardless of model; the backend ignores it where unsupported
// and echoes back the one it used.
async function getSelectedAudience() {
  const result = await chrome.storage.local.get(AUDIENCE_STORAGE_KEY);
  return result[AUDIENCE_STORAGE_KEY] || DEFAULT_AUDIENCE;
}

// Everything a picker needs in one read. granularityIntent can differ from granularity
// (see header).
async function readSelection() {
  const model = await getSelectedModel();
  const stored = await chrome.storage.local.get([
    GRANULARITY_STORAGE_KEY,
    AUDIENCE_STORAGE_KEY,
    AUDIENCE_BY_FAMILY_STORAGE_KEY,
  ]);
  const family = modelFamilyOf(model) || modelFamilyOf(DEFAULT_MODEL);
  const granularity = modelFamilyGranularityOf(model);
  const storedIntent = stored[GRANULARITY_STORAGE_KEY];
  const intent = RENAMED_GRANULARITIES[storedIntent] || storedIntent;
  // persist the migration, like the model key above
  if (intent !== storedIntent) {
    await chrome.storage.local.set({ [GRANULARITY_STORAGE_KEY]: intent });
  }
  const audienceByFamily = stored[AUDIENCE_BY_FAMILY_STORAGE_KEY] || {};
  return {
    model,
    family,
    granularity,
    // unrecognized or missing intent defers to the resolved key
    granularityIntent: GRANULARITY_CHOICES.some((g) => g.value === intent) ? intent : granularity,
    audience: stored[AUDIENCE_STORAGE_KEY] || DEFAULT_AUDIENCE,
    audienceByFamily,
  };
}

// --- writing ----------------------------------------------------------------

// Applies a picker choice. Every field is optional and defaults to the current
// selection, so a UI can write one dimension at a time.
//
// With `modelsLoaded` (from /health), an unavailable combination resolves to the
// family's sentence key; the intent is still stored as asked so it recovers once the
// other model is loaded.
async function writeSelection(next, modelsLoaded) {
  const current = await readSelection();
  const family = next.family || current.family;
  const intent = next.granularity || current.granularityIntent;
  const audience =
    next.audience ||
    // switching family restores its last audience, else the global one
    (next.family ? current.audienceByFamily[family] : null) ||
    current.audience;

  const status = familyGranularityStatus(family, intent, modelsLoaded);
  const model =
    status.available || !modelsLoaded
      ? status.key || familyModelKey(family, GRANULARITY_SENTENCE_BY_SENTENCE)
      : familyModelKey(family, GRANULARITY_SENTENCE_BY_SENTENCE);

  await chrome.storage.local.set({
    [MODEL_STORAGE_KEY]: model,
    [GRANULARITY_STORAGE_KEY]: intent,
    [AUDIENCE_STORAGE_KEY]: audience,
    [AUDIENCE_BY_FAMILY_STORAGE_KEY]: { ...current.audienceByFamily, [family]: audience },
  });
  return { model, family, granularity: modelFamilyGranularityOf(model), granularityIntent: intent, audience };
}

// Selects a raw backend key (caller: the Error page's "switch to an available model"
// button). Syncs the granularity intent so the picker doesn't contradict the key.
async function selectModelKey(model) {
  await chrome.storage.local.set({
    [MODEL_STORAGE_KEY]: model,
    [GRANULARITY_STORAGE_KEY]: modelFamilyGranularityOf(model),
  });
}
