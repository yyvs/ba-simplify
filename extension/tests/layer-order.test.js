// Stacking order over extension pages: sticky toolbar, burger dropdowns, compare/analyze
// modals, transient notice, model picker.
//
//   cd extension/tests && npm install && node layer-order.test.js
//
// CSS-only. Regression: per-surface z-indexes (toolbar 10000, notice 200, modal 100,
// menu 10) put the toolbar above everything, hiding notices (drawn where the bar is) and
// showing the nav through modals. Values now come from shared/layers.css; this fails on a
// hard-coded value, a drifted fallback, or a page not linking the file.
//
// Also: .page-block must not have `overflow: hidden`, which clipped dropdowns regardless
// of z-index.
const fs = require("fs");
const path = require("path");

const EXT = path.join(__dirname, "..");
const read = (p) => fs.readFileSync(path.join(EXT, p), "utf8");

let failures = 0;
function check(label, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${ok || !detail ? "" : ` — ${detail}`}`);
  if (!ok) failures += 1;
}

// --- the scale itself
const LAYERS_CSS = read("shared/layers.css");
const tokens = {};
for (const [, name, value] of LAYERS_CSS.matchAll(/--layer-([a-z]+):\s*(\d+);/g)) {
  tokens[name] = Number(value);
}

const ORDER = ["toolbar", "modal", "menu", "notice", "picker"];
check(
  "shared/layers.css defines every layer",
  ORDER.every((name) => Number.isInteger(tokens[name])),
  JSON.stringify(tokens)
);
for (let i = 1; i < ORDER.length; i += 1) {
  const [below, above] = [ORDER[i - 1], ORDER[i]];
  check(
    `${above} outranks ${below}`,
    tokens[above] > tokens[below],
    `${above}=${tokens[above]} ${below}=${tokens[below]}`
  );
}
check("the picker is the top of the scale", tokens.picker === 2147483647, String(tokens.picker));

// --- the rules that use it
// selector -> the layer it must be on. One entry per surface that floats.
const USES = [
  ["shared/toolbar.css", "#ext-toolbar", "toolbar"],
  ["shared/history.css", ".compare-modal", "modal"],
  ["shared/history.css", ".page-menu", "menu"],
  ["shared/history.css", ".page-notice", "notice"],
];

for (const [file, selector, layer] of USES) {
  const css = read(file);
  // exact selector, so ".page-menu button" can't match ".page-menu" (nor .page-notice-hidden)
  const rule = new RegExp(`\\${selector}\\s*\\{([^}]*)\\}`).exec(css);
  const declared = rule && /z-index:\s*([^;]+);/.exec(rule[1]);
  check(`${selector} declares a z-index`, !!declared, `in ${file}`);
  if (!declared) continue;
  const used = /var\(\s*--layer-([a-z]+)\s*,\s*(\d+)\s*\)/.exec(declared[1]);
  check(
    `${selector} is on the ${layer} layer`,
    !!used && used[1] === layer,
    declared[1].trim()
  );
  // a stale fallback only shows on a page missing the layers.css link
  if (used) {
    check(
      `${selector}'s fallback matches --layer-${layer}`,
      Number(used[2]) === tokens[layer],
      `${used[2]} vs ${tokens[layer]}`
    );
  }
}

// --- nothing clips the dropdowns
const HISTORY_CSS = read("shared/history.css");
const pageBlock = /\.page-block\s*\{([^}]*)\}/.exec(HISTORY_CSS);
check("the .page-block rule is still there", !!pageBlock);
check(
  ".page-block doesn't clip its burger dropdowns",
  !!pageBlock && !/overflow:\s*hidden/.test(pageBlock[1]),
  pageBlock && pageBlock[1].trim()
);

// --- every page that stacks anything loads the scale
for (const page of fs.readdirSync(EXT).filter((f) => f.endsWith(".html"))) {
  const html = read(page);
  if (!html.includes("shared/toolbar.css")) continue; // index.html: a redirect, no UI
  check(`${page} links shared/layers.css`, html.includes("shared/layers.css"));
}

// --- picker (can't read the variables)
// Injected into arbitrary pages in a shadow root, so it repeats the value as a literal.
const PICKER_JS = read("shared/picker.js");
const overlay = /\.overlay\s*\{([^}]*)\}/.exec(PICKER_JS);
const overlayZ = overlay && /z-index:\s*(\d+);/.exec(overlay[1]);
check("the picker overlay states its z-index", !!overlayZ);
check(
  "the picker overlay matches --layer-picker",
  !!overlayZ && Number(overlayZ[1]) === tokens.picker,
  overlayZ && overlayZ[1]
);

// The content script's notice is on a web page: it only has to stay below the picker.
const CONTENT_JS = read("content.js");
const noticeZ = /#simplify-notice\s*\{[\s\S]*?z-index:\s*(\d+);/.exec(CONTENT_JS);
check("the injected notice states its z-index", !!noticeZ);
check(
  "the injected notice stays under the picker",
  !!noticeZ && Number(noticeZ[1]) < tokens.picker,
  noticeZ && noticeZ[1]
);

console.log(failures === 0 ? "\nall checks passed" : `\n${failures} check(s) failed`);
process.exit(failures === 0 ? 0 : 1);
