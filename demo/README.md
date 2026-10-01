# Extension demo pages

Static HTML fixtures for manually verifying `extension/` end to end — above all the
DOM targeting, which decides what on a page counts as prose. The table below is the
checklist: each page states what it should do, and the `.meta` notes on the page itself
say what to expect case by case. The design behind it lives in `extension/content.js`'s
own comments, which are the authority here.

## Setup

1. Start the backend — `backend/run_dev.sh` also serves this `demo/` folder over
   plain HTTP (so you don't need `file://` + "Allow access to file URLs"), and
   stops both together after a period of inactivity:

   ```sh
   cd backend
   ./run_dev.sh                # backend on :8000, demo pages on :8001
   ```

   (`DEMO_PORT=...` to change the demo port, `NO_DEMO=1` to skip serving it, `-idle
900` to change the idle timeout or `-noidle` to remove it, same `PORT` override as
   before — see the script's header comment.)

   Alternatively, run them separately:

   ```sh
   cd backend && source venv/bin/activate && uvicorn main:app --reload --port 8000
   cd demo && python3 -m http.server 8001
   ```

2. Load `extension/` as an unpacked extension in `chrome://extensions` (enable
   Developer mode → "Load unpacked").
3. Open `http://127.0.0.1:8001/` (or `demo/index.html` directly via `file://`, in
   which case enable "Allow access to file URLs" for the extension).
4. Click the toolbar icon to toggle simplification on each page; click again to revert.

## Pages

| Page                                | What it checks                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `index.html`                        | The overview these pages are reached from, and what the extension's Demo menu entry opens. Carries a scratch box of its own: type or paste text, toggle the extension, and compare. It also prefills from the URL — `index.html#text=...` or `?text=...` — for pasting a case straight in.                                                                                                                                                       |
| `01-basic-paragraphs.html`          | Baseline toggle/revert, hover-for-original tooltip, the removed 30-char minimum-length filter, the code-pattern skip heuristic.                                                                                                                                                                                                                                                                                                                  |
| `02-dom-targeting.html`             | Headings, list items, table cells, semantic sectioning tags (`header`/`main`/`section`/`article`/`aside`/`footer`), `blockquote`, `figcaption`/`summary`, inline formatting survival, and — importantly — that nested `<div>`s wrapping a `<p>` only trigger **one** `/simplify` request, not one per wrapper.                                                                                                                                   |
| `03-interactive-and-forms.html`     | `<a>`/`<button>` staying inline within a cohesive sentence vs. standing alone when orphaned, `<label>`, `<option>`, and `placeholder` attribute translation (with exact revert).                                                                                                                                                                                                                                                                 |
| `04-exclusions.html`                | `<pre>`/`<code>`/`<script>`/`<style>` skip, and the `translate="no"`/`.notranslate` opt-out — including a `.notranslate` span nested _inside_ an otherwise-translated sentence.                                                                                                                                                                                                                                                                  |
| `05-dynamic-content.html`           | `MutationObserver` handling of lazily-inserted content (headings/paragraphs/list items), and that the on-page progress count is cumulative across insertions rather than resetting.                                                                                                                                                                                                                                                              |
| `06-line-breaks-and-sentences.html` | The two-stage split: `<br>` cut into separate chunks client-side (including the no-punctuation case that would otherwise merge, `<br><br>` gaps, and the old word-fusing bug), then per-sentence splitting in the backend (including abbreviations like "Dr."/"U.S." that must _not_ split), plus exact revert of `<br>`/link/bold markup.                                                                                                       |
| `07-document-sections.html`         | Document-granularity collection (**set the scope to "Whole sections" in the picker first** — toolbar icon → right-click → "Change simplification method"): content-scope detection, headings left untouched as section dividers, each section's paragraphs all still standing with the answer divided back across them, content outside `<main>` left alone, and full revert. Also works as an A/B against the sentence models on the same page. |
| `08-html-checker.html`              | Ad-hoc markup checking: a textarea whose HTML is rendered twice at equal width in one row — the left copy pinned via `translate="no"`/`.notranslate` as the reference, the right copy a normal candidate — so before/after sit side by side. Both re-render (debounced) as you type.                                                                                                                                                             |

Each test case is annotated with a small `.meta` note (styled gray, marked
`class="notranslate"`) stating what's expected — those notes doubling as another live
check that `.notranslate` actually holds up across every page.

No build step or dependencies — these are plain static HTML/CSS/JS files.
