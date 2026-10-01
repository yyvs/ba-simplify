// Everything manifest.json names exists, and the icons are declared.
//
//   cd extension/tests && npm install && node manifest-assets.test.js
//
// Regression: icons/ held four PNGs the manifest never declared. Chrome silently shows its
// grey puzzle piece instead (toolbar, management page, permission prompts).
//
// "icons" (browser UI) and "action.default_icon" (toolbar button) are independent; the
// action stays the placeholder without the latter.
//
// Sizes are checked against the PNG's IHDR dimensions, so a misnamed file (a 48 saved as
// icon32.png) fails here.
const fs = require("fs");
const path = require("path");

const EXT = path.join(__dirname, "..");
const manifest = JSON.parse(
  fs.readFileSync(path.join(EXT, "manifest.json"), "utf8"),
);

let failures = 0;
function check(label, ok, detail = "") {
  console.log(
    `${ok ? "PASS" : "FAIL"}  ${label}${ok || !detail ? "" : ` — ${detail}`}`,
  );
  if (!ok) failures += 1;
}

const exists = (rel) => fs.existsSync(path.join(EXT, rel));

// 8-byte signature, then IHDR: width/height are big-endian uint32 at bytes 16 and 20.
function pngSize(rel) {
  const buf = fs.readFileSync(path.join(EXT, rel));
  if (buf.length < 24 || buf.toString("ascii", 12, 16) !== "IHDR") return null;
  return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
}

// --- every file the manifest names exists
const referenced = [
  manifest.background?.service_worker,
  ...(manifest.content_scripts ?? []).flatMap((entry) => [
    ...(entry.js ?? []),
    ...(entry.css ?? []),
  ]),
  ...Object.values(manifest.icons ?? {}),
  ...Object.values(manifest.action?.default_icon ?? {}),
  manifest.action?.default_popup,
].filter(Boolean);

for (const rel of referenced) {
  check(`manifest references a file that exists: ${rel}`, exists(rel));
}

// --- the two icon keys
for (const [label, declared] of [
  ["icons", manifest.icons],
  ["action.default_icon", manifest.action?.default_icon],
]) {
  check(`${label} is declared`, !!declared && Object.keys(declared).length > 0);
  if (!declared) continue;
  for (const [size, rel] of Object.entries(declared)) {
    const dims = pngSize(rel);
    check(
      `${label}["${size}"] is a ${size}x${size} PNG`,
      !!dims && dims.width === Number(size) && dims.height === Number(size),
      dims ? `${dims.width}x${dims.height}` : "not a readable PNG",
    );
  }
}

// Chrome rescales the nearest declared size, so a missing 16 is a blurry toolbar, not an
// error. These are the sizes the browser asks for.
for (const size of ["16", "32", "48", "128"]) {
  check(
    `toolbar icon declares ${size}px`,
    !!manifest.action?.default_icon?.[size],
  );
}

// --- nothing in icons/ is left unreferenced
const onDisk = fs
  .readdirSync(path.join(EXT, "icons"))
  .filter((f) => f.endsWith(".png"))
  .map((f) => `icons/${f}`);
const used = new Set([
  ...Object.values(manifest.icons ?? {}),
  ...Object.values(manifest.action?.default_icon ?? {}),
]);
for (const rel of onDisk) {
  check(`${rel} is declared in the manifest`, used.has(rel));
}

// --- in-page toolbar icon
// shared/toolbar.js runs only on chrome-extension:// pages, so no web_accessible_resources
// entry is needed. A rename in icons/ would otherwise silently break the <img> on five pages.
const TOOLBAR_JS = fs.readFileSync(
  path.join(EXT, "shared", "toolbar.js"),
  "utf8",
);
const toolbarIcon = /getURL\(\s*["'`](icons\/[^"'`]+)["'`]\s*\)/.exec(
  TOOLBAR_JS,
);
check("the extension toolbar loads an icon", !!toolbarIcon);
check(
  "the toolbar's icon exists on disk",
  !!toolbarIcon && exists(toolbarIcon[1]),
  toolbarIcon && toolbarIcon[1],
);
check(
  "the toolbar's icon is one of the declared sizes",
  !!toolbarIcon && used.has(toolbarIcon[1]),
  toolbarIcon && toolbarIcon[1],
);

console.log(
  failures === 0 ? "\nall checks passed" : `\n${failures} check(s) failed`,
);
process.exit(failures === 0 ? 0 : 1);
