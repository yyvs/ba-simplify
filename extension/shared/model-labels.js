// shared/model-labels.js - display names for backend model keys, the method/granularity
// vocabulary, model families for the picker, and target-audience constants. Loaded via
// <script> on extension pages and importScripts() in background.js; plain globals, no
// bundler.
//
// "finetuned" is the checkpoint fine-tuned for this project; "online" is the ready-made
// third-party fine-tune from the Hub (its repo id comes from /health model_names.online,
// overridable via SIMPLIFIER_MODEL_ONLINE). "finetuned" gets a static label because its
// configured value can be a filesystem path.
//
// "llm_*" labels are built from the Ollama tag ("qwen2.5:7b-instruct-q4_K_M" ->
// "LLM (Qwen2.5 7B)"), not hardcoded, so overriding the env var renames the entry. The
// verbatim id is shown in the Backend status section (linked via modelSourceUrl) and in
// the on-page notice, not in pickers.
//
// Labels never include scope: modelScopeText() gives it separately for a muted span
// beside the name (see "Models loaded" on the status dashboard);
// modelDisplayNameWithScope() joins both for text-only contexts (<option>, status line).
const DEFAULT_ONLINE_MODEL_NAME = "Comparison fine-tuned model";
// Shared by both of this project's checkpoints ("finetuned", "document"); scope is
// shown beside the name (modelScopeText).
const FINETUNED_MODEL_LABEL = "Fine-tuned model";
// The two prompted-LLM sizes are separate keys, not a size setting: different output,
// separate caches, ~2x speed difference. `size` is a fallback for a tag that doesn't state
// one. Unlike the document checkpoint, the prompted document models have no 512-token
// training ceiling, only a context window (hence per-model /health.document_max_tokens).
const LLM_KEY_TRAITS = {
  llm_7b: { size: "7B" },
  llm_3b: { size: "3B" },
  llm_doc_7b: { size: "7B" },
  llm_doc_3b: { size: "3B" },
};
// modelFamilyTitle fallback before /health reports a tag, or for an unparseable tag.
// Deliberately generic rather than guessing a model name.
const GENERIC_LLM_FAMILY = "Open LLM";

// "qwen2.5:7b-instruct-q4_K_M" -> "Qwen2.5". The part before ":" is Ollama's model name;
// "mistral-nemo" -> "Mistral-Nemo".
function ollamaFamilyName(tag) {
  const base = String(tag).split(":")[0].split("/").pop();
  if (!base) return null;
  return base
    .split("-")
    .map((part) => (part ? part[0].toUpperCase() + part.slice(1) : part))
    .join("-");
}

// Size stated by the tag, preferred over the key's (an "llm_7b" key pointed at an 8B tag
// reads "8B"). Matches "7b" / "3.8b".
function ollamaParamSize(tag) {
  const match = /(?:^|[:._-])(\d+(?:\.\d+)?)b(?:[._-]|$)/i.exec(String(tag));
  return match ? `${match[1]}B` : null;
}

// "LLM (Qwen2.5 7B)": family and size from the served tag where it states them. No
// speed/quality adjectives. The "LLM (...)" wrapper marks the method next to the
// fine-tuned labels; it is the short form of "Prompted LLM" to fit menu rows.
//
// Picker cards don't use this (see modelFamilyTitle): they show the method on their own
// line, so the wrapper would repeat it.
function llmDisplayName(key, tag) {
  const { size } = LLM_KEY_TRAITS[key];
  const family = tag && ollamaFamilyName(tag);
  const sizeText = (tag && ollamaParamSize(tag)) || size;
  // no tag yet: size only, avoiding "LLM (Open LLM 7B)"
  const inner = family ? `${family} ${sizeText}` : sizeText;
  return `LLM (${inner})`;
}

// Every model key the backend can expose, in picker order; usable ones come from
// /health.models_loaded.
const MODEL_KEYS = [
  "online",
  "finetuned",
  "llm_7b",
  "llm_3b",
  "llm_doc_7b",
  "llm_doc_3b",
  "document",
];

function modelDisplayName(key, modelNames) {
  if (key === "online") return DEFAULT_ONLINE_MODEL_NAME;
  // "document" is the second project checkpoint; shares the label, differs in scope
  if (key === "finetuned" || key === "document") return FINETUNED_MODEL_LABEL;
  if (LLM_KEY_TRAITS[key])
    return llmDisplayName(key, modelNames && modelNames[key]);
  return key;
}

// --- scope beside a model name ---
// Lowercased GRANULARITY_LABELS entry ("Fine-tuned model" · "whole sections").
function modelScopeText(key, granularities) {
  return granularityLabel(modelGranularity(key, granularities)).toLowerCase();
}

// For text-only contexts (<option>, status line). With a DOM, prefer a separate muted
// span for modelScopeText() -- see renderModelsLoaded in shared/status.js.
function modelDisplayNameWithScope(key, modelNames, granularities) {
  return `${modelDisplayName(key, modelNames)} · ${modelScopeText(key, granularities)}`;
}

// --- exact model id and its source page ---
// Classified by syntax, since any env var accepts any shape: Ollama tag ("name:variant",
// no slash) -> ollama.com, HF repo id ("owner/name") -> huggingface.co, filesystem path
// or anything unrecognized -> null (no dead links).
function modelSourceUrl(modelId) {
  if (!modelId || typeof modelId !== "string") return null;
  const id = modelId.trim();
  if (!id) return null;
  // paths first: a Windows path like "C:\\models\\best" also contains a colon
  if (/^[.~/]/.test(id) || /^[A-Za-z]:[\\/]/.test(id)) return null;
  if (id.includes(":") && !id.includes("/"))
    return `https://ollama.com/library/${id}`;
  if (/^[\w.-]+\/[\w.-]+$/.test(id)) return `https://huggingface.co/${id}`;
  return null;
}

// --- run vocabulary: method and granularity ---
// Same values on the wire (/health), in storage, the History log and here.
// backend/vocabulary.py mirrors them; backend/tests/test_vocabulary.py fails if they
// disagree.
//
// - method: fine-tuned seq2seq checkpoint vs. prompted instruction-tuned LLM (the
//   project's central comparison). The specific checkpoint/tag is the model id.
// - granularity: unit of text per request; also changes how content.js collects text:
//   - sentence_by_sentence: one request per leaf element (split further at <br>),
//     replaced in place. Inline formatting is lost, structure isn't.
//   - whole_sections: one request per heading-delimited section of the main content.
//     The model may delete, merge and reorder sentences, so output can't be mapped back
//     to paragraphs; a section's paragraphs collapse into one.
//
// snake_case machine values; the label maps below are the only place they become prose,
// so wording can change without stored values going stale.
const METHOD_FINE_TUNED_SEQ2SEQ = "fine_tuned_seq2seq";
const METHOD_PROMPTED_LLM = "prompted_llm";

const METHOD_LABELS = {
  [METHOD_FINE_TUNED_SEQ2SEQ]: "Fine-tuned seq2seq",
  [METHOD_PROMPTED_LLM]: "Prompted LLM",
};

const GRANULARITY_SENTENCE_BY_SENTENCE = "sentence_by_sentence";
const GRANULARITY_WHOLE_SECTIONS = "whole_sections";

const GRANULARITY_LABELS = {
  [GRANULARITY_SENTENCE_BY_SENTENCE]: "Sentence by sentence",
  [GRANULARITY_WHOLE_SECTIONS]: "Whole sections",
};

// Unknown values are shown raw, not dropped (e.g. written by a later build).
function methodLabel(method) {
  return METHOD_LABELS[method] || method || "unknown method";
}

function granularityLabel(granularity) {
  return (
    GRANULARITY_LABELS[granularity] || granularity || "unknown granularity"
  );
}

// --- method and granularity per model key ---
// /health's `methods` / `granularities` are authoritative. Fallbacks cover the time before
// it answers (context menus built at install, the notice shown on click); being wrong
// costs one label flicker.
const FALLBACK_MODEL_METHODS = {
  online: METHOD_FINE_TUNED_SEQ2SEQ,
  finetuned: METHOD_FINE_TUNED_SEQ2SEQ,
  document: METHOD_FINE_TUNED_SEQ2SEQ,
  llm_7b: METHOD_PROMPTED_LLM,
  llm_3b: METHOD_PROMPTED_LLM,
  llm_doc_7b: METHOD_PROMPTED_LLM,
  llm_doc_3b: METHOD_PROMPTED_LLM,
};

const FALLBACK_GRANULARITIES = {
  online: GRANULARITY_SENTENCE_BY_SENTENCE,
  finetuned: GRANULARITY_SENTENCE_BY_SENTENCE,
  llm_7b: GRANULARITY_SENTENCE_BY_SENTENCE,
  llm_3b: GRANULARITY_SENTENCE_BY_SENTENCE,
  llm_doc_7b: GRANULARITY_WHOLE_SECTIONS,
  llm_doc_3b: GRANULARITY_WHOLE_SECTIONS,
  document: GRANULARITY_WHOLE_SECTIONS,
};

// An unknown key degrades to the key itself rather than claiming the wrong method.
function modelMethod(key, methods) {
  if (methods && methods[key]) return methods[key];
  return FALLBACK_MODEL_METHODS[key] || key;
}

function modelGranularity(key, granularities) {
  if (granularities && granularities[key]) return granularities[key];
  return FALLBACK_GRANULARITIES[key] || GRANULARITY_SENTENCE_BY_SENTENCE;
}

// "Method · granularity", as every "what is simplifying this page" read-out shows it.
function modelMethodLabel(key, methods, granularities) {
  return `${methodLabel(modelMethod(key, methods))} · ${granularityLabel(modelGranularity(key, granularities))}`;
}

// --- target audience ---
// Only prompted LLMs use it; the backend normalizes it away for seq2seq checkpoints.
// Stored regardless of model so switching doesn't lose it.
const AUDIENCE_STORAGE_KEY = "simplifierAudience";
const DEFAULT_AUDIENCE = "non_native_speakers"; // matches prompting.DEFAULT_AUDIENCE

// Fallback until /health answers with `audience_models`.
const FALLBACK_AUDIENCE_MODELS = [
  "llm_7b",
  "llm_3b",
  "llm_doc_7b",
  "llm_doc_3b",
];

// Fallback copy of the backend's audience list (prompting.Audience / AUDIENCE_LABELS),
// e.g. for context menus built at install time. Prefer /health's list.
const FALLBACK_AUDIENCES = [
  { value: "non_native_speakers", label: "Non-native speakers" },
  { value: "children", label: "Children · 8–12" },
  { value: "low_literacy", label: "Low literacy" },
  { value: "cognitive_disability", label: "Cognitive disabilities" },
  { value: "general_adult", label: "General adult" },
];

function audienceDisplayName(value, audiences) {
  const list =
    Array.isArray(audiences) && audiences.length
      ? audiences
      : FALLBACK_AUDIENCES;
  const found = list.find((a) => a.value === value);
  return found ? found.label : value;
}

// --- model families: the picker's view of the keys ---
// Backend keys are (weights, granularity): "finetuned"/"document" are two separately
// trained checkpoints, "llm_7b"/"llm_doc_7b" two prompt templates over one served tag.
// Each key loads, caches and fails independently. The picker instead shows four
// families with granularity as a setting; (family, granularity) resolves to exactly one
// key (familyModelKey), which is what gets stored and sent, so nothing downstream knows
// about families.
//
// Family ids equal their sentence-scope key, so modelFamilyOf on an old stored key is a
// lookup, not a migration.
const MODEL_FAMILY_ORDER = ["finetuned", "llm_7b", "llm_3b", "online"];
// `method` and `keys` use the canonical vocabulary constants, so renaming a value can't
// leave a lookup here resolving to undefined.
//
// `keys[whole_sections]: null` on "online" is deliberate: the off-the-shelf checkpoint is
// trained on single sentences with a 64-token ceiling (backend/main.py's
// SEQ2SEQ_MAX_LENGTH) and would truncate each section. familyGranularityStatus() reports
// the absence.
const MODEL_FAMILIES = {
  finetuned: {
    method: METHOD_FINE_TUNED_SEQ2SEQ,
    // contrast with the off-the-shelf comparison checkpoint; kept short to fit one line
    detail: "Custom-trained",
    keys: {
      [GRANULARITY_SENTENCE_BY_SENTENCE]: "finetuned",
      [GRANULARITY_WHOLE_SECTIONS]: "document",
    },
  },
  llm_7b: {
    method: METHOD_PROMPTED_LLM,
    detail: "Highest quality · slower",
    keys: {
      [GRANULARITY_SENTENCE_BY_SENTENCE]: "llm_7b",
      [GRANULARITY_WHOLE_SECTIONS]: "llm_doc_7b",
    },
  },
  llm_3b: {
    method: METHOD_PROMPTED_LLM,
    detail: "Good quality · faster",
    keys: {
      [GRANULARITY_SENTENCE_BY_SENTENCE]: "llm_3b",
      [GRANULARITY_WHOLE_SECTIONS]: "llm_doc_3b",
    },
  },
  online: {
    method: METHOD_FINE_TUNED_SEQ2SEQ,
    // detail is the exact repo id (the third-party fine-tune compared against), rendered
    // in code style by the picker
    detail: null,
    detailIsId: true,
    keys: {
      [GRANULARITY_SENTENCE_BY_SENTENCE]: "online",
      [GRANULARITY_WHOLE_SECTIONS]: null,
    },
  },
};

// Inverse of familyModelKey(). null for a key no family claims, so callers can fall back.
function modelFamilyOf(modelKey) {
  return (
    MODEL_FAMILY_ORDER.find((id) =>
      Object.values(MODEL_FAMILIES[id].keys).includes(modelKey),
    ) || null
  );
}

function modelFamilyGranularityOf(modelKey) {
  const family = modelFamilyOf(modelKey);
  if (!family) return GRANULARITY_SENTENCE_BY_SENTENCE;
  return MODEL_FAMILIES[family].keys[GRANULARITY_WHOLE_SECTIONS] === modelKey
    ? GRANULARITY_WHOLE_SECTIONS
    : GRANULARITY_SENTENCE_BY_SENTENCE;
}

// null where the combination has no model.
function familyModelKey(familyId, granularity) {
  const family = MODEL_FAMILIES[familyId];
  if (!family) return null;
  return family.keys[granularity] || null;
}

// Card title. Prompted families derive it from the served tag (configurable, e.g.
// SIMPLIFIER_MODEL_LLM_7B); `modelNames` is /health.model_names, without it the size
// comes from the family id.
function modelFamilyTitle(familyId, modelNames) {
  if (familyId === "finetuned") return FINETUNED_MODEL_LABEL;
  if (familyId === "online") return DEFAULT_ONLINE_MODEL_NAME;
  if (!LLM_KEY_TRAITS[familyId]) return familyId;
  // sentence key's tag is served by default; the document key uses the same model
  const family = MODEL_FAMILIES[familyId];
  const tag =
    modelNames &&
    (modelNames[family.keys[GRANULARITY_SENTENCE_BY_SENTENCE]] ||
      modelNames[family.keys[GRANULARITY_WHOLE_SECTIONS]]);
  const size = (tag && ollamaParamSize(tag)) || LLM_KEY_TRAITS[familyId].size;
  return `Ollama ${(tag && ollamaFamilyName(tag)) || GENERIC_LLM_FAMILY} · ${size}`;
}

function modelFamilyMethod(familyId) {
  const family = MODEL_FAMILIES[familyId];
  return family ? methodLabel(family.method) : familyId;
}

// Card's third line: static detail, or the verbatim repo id for "online".
function modelFamilyDetail(familyId, modelNames) {
  const family = MODEL_FAMILIES[familyId];
  if (!family) return "";
  if (!family.detailIsId) return family.detail;
  return (
    (modelNames && modelNames[family.keys[GRANULARITY_SENTENCE_BY_SENTENCE]]) ||
    DEFAULT_ONLINE_MODEL_NAME
  );
}

function modelFamilyDetailIsId(familyId) {
  return !!(MODEL_FAMILIES[familyId] && MODEL_FAMILIES[familyId].detailIsId);
}

// --- granularity as a setting ---
// The picker's two segments, labelled from the shared vocabulary.
const GRANULARITY_CHOICES = [
  {
    value: GRANULARITY_SENTENCE_BY_SENTENCE,
    label: GRANULARITY_LABELS[GRANULARITY_SENTENCE_BY_SENTENCE],
  },
  {
    value: GRANULARITY_WHOLE_SECTIONS,
    label: GRANULARITY_LABELS[GRANULARITY_WHOLE_SECTIONS],
  },
];

// Whether a (family, granularity) cell is selectable, and why not: no such model vs.
// not loaded. Without `modelsLoaded` (/health.models_loaded) only nonexistent
// combinations are unavailable.
function familyGranularityStatus(familyId, granularity, modelsLoaded) {
  const key = familyModelKey(familyId, granularity);
  if (!key) {
    return {
      key: null,
      available: false,
      reason: "no whole-sections model for this one",
    };
  }
  if (!Array.isArray(modelsLoaded))
    return { key, available: true, reason: null };
  if (!modelsLoaded.includes(key)) {
    return { key, available: false, reason: "not loaded on the backend" };
  }
  return { key, available: true, reason: null };
}

// --- comparing models that read different units ---
// A comparison fixes the input: the page is cut once and every arm gets the same strings.
// A model's reading granularity need not match the cut: a section sent to a
// sentence_by_sentence key is split, simplified per sentence and rejoined by the backend
// (backend/main.py's simplify_text). "Sections read sentence by sentence" vs. "sections
// read whole" is the project's document-scope question.
//
// The panel's granularity control picks the page cut; this returns every key that can
// read that unit:
//   - whole sections: every key of every family
//   - sentence by sentence: sentence_by_sentence keys only (a document checkpoint given
//     one leaf element answers no document-scope question)
// Within each family, the reading matching the cut comes first.
function compareUnitCandidates(unitGranularity, modelsLoaded) {
  const readings =
    unitGranularity === GRANULARITY_WHOLE_SECTIONS
      ? [GRANULARITY_WHOLE_SECTIONS, GRANULARITY_SENTENCE_BY_SENTENCE]
      : [GRANULARITY_SENTENCE_BY_SENTENCE];
  const rows = [];
  MODEL_FAMILY_ORDER.forEach((familyId) => {
    readings.forEach((keyGranularity) => {
      const status = familyGranularityStatus(
        familyId,
        keyGranularity,
        modelsLoaded,
      );
      if (!status.key) return;
      rows.push({ familyId, keyGranularity, ...status });
    });
  });
  return rows;
}

// --- comparing over more than one cut ---
// A comparison can use both cuts in one table. Entries are the coarsest cut's units: a
// section's row carries the whole-sections answer and the joined sentence answers for the
// units inside it, so both answer the same text (see content.js's collectComparisonPlan).
// Coarsest first, so the entry-defining cut runs first and ordering is consistent.
function comparisonCutOrder(cuts) {
  return [GRANULARITY_WHOLE_SECTIONS, GRANULARITY_SENTENCE_BY_SENTENCE].filter(
    (cut) => (cuts || []).includes(cut),
  );
}

// Every selectable key across the chosen cuts, once each, with the cuts it can run in
// (sentence-scope keys: both; whole-sections keys: sections only). Ordered by the
// coarsest cut first so rows don't reshuffle as cuts are toggled.
function compareCutCandidates(cuts, modelsLoaded) {
  const rows = [];
  const byKey = new Map();
  comparisonCutOrder(cuts).forEach((cut) => {
    compareUnitCandidates(cut, modelsLoaded).forEach((candidate) => {
      const seen = byKey.get(candidate.key);
      if (seen) {
        seen.cuts.push(cut);
        return;
      }
      const row = { ...candidate, cuts: [cut] };
      byKey.set(candidate.key, row);
      rows.push(row);
    });
  });
  return rows;
}

// Short cut names for column headings and picker rows ("A · sections").
const CUT_SHORT_LABELS = {
  [GRANULARITY_WHOLE_SECTIONS]: "sections",
  [GRANULARITY_SENTENCE_BY_SENTENCE]: "sentences",
};

function cutShortLabel(cut) {
  return CUT_SHORT_LABELS[cut] || cut;
}

// What one model does with one cut, for the picker's row and the report's arm line.
function cutReadingLabel(cut, keyGranularity) {
  if (cut === GRANULARITY_WHOLE_SECTIONS) {
    return keyGranularity === GRANULARITY_WHOLE_SECTIONS
      ? "each section whole"
      : "each section split into sentences";
  }
  return "one leaf element at a time";
}

// Arm labels A, B, C... used in the panel, run records, combined report and its
// evaluation columns. Letters, not model names, because two arms can come from one family
// (same checkpoint at both scopes). Past Z: "#27".
const COMPARISON_ARM_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

function comparisonArmLabel(index) {
  return COMPARISON_ARM_LABELS[index] || `#${index + 1}`;
}

function modelFamilyAvailable(familyId, modelsLoaded) {
  return GRANULARITY_CHOICES.some(
    (g) => familyGranularityStatus(familyId, g.value, modelsLoaded).available,
  );
}

// Derived from /health's per-key audience_models.
function modelFamilyHonoursAudience(familyId, audienceModels) {
  const capable =
    Array.isArray(audienceModels) && audienceModels.length
      ? audienceModels
      : FALLBACK_AUDIENCE_MODELS;
  const family = MODEL_FAMILIES[familyId];
  if (!family) return false;
  return Object.values(family.keys).some((key) => key && capable.includes(key));
}

// Active configuration in one line (context-menu item, toolbar summary). Audience only
// for families that use it.
function selectionSummary(familyId, granularity, audience, health) {
  const names = health && health.model_names;
  const parts = [
    modelFamilyTitle(familyId, names),
    granularityLabel(granularity),
  ];
  if (modelFamilyHonoursAudience(familyId, health && health.audience_models)) {
    parts.push(audienceDisplayName(audience, health && health.audiences));
  }
  return parts.join(" · ");
}
