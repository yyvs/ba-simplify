// shared/home.js - the extension homepage (home.html; chrome-extension://<id>/ redirects
// there via the root index.html shim). DEMO_PAGES comes from shared/demo-pages.js.
const container = document.getElementById("demo-links");
const demoCard = document.querySelector('[data-demo="index.html"]');
if (demoCard) {
  demoCard.addEventListener("click", () => {
    chrome.runtime.sendMessage({ cmd: "openDemoPage", file: "index.html" });
  });
}
DEMO_PAGES.forEach(({ label, file }) => {
  const btn = document.createElement("button");
  btn.textContent = label;
  btn.addEventListener("click", () => {
    chrome.runtime.sendMessage({ cmd: "openDemoPage", file });
  });
  container.appendChild(btn);
});
