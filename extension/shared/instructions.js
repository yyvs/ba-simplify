// shared/instructions.js - the setup page (instructions.html), also opened by
// background.js on install and update. Content is the same for both.
//
// Demo pages are served by the local static server, not the extension, so the button asks
// the service worker to open them (same message as home.js) instead of linking a URL.
const openDemoButton = document.getElementById("open-demo");
if (openDemoButton) {
  openDemoButton.addEventListener("click", () => {
    chrome.runtime.sendMessage({ cmd: "openDemoPage", file: "index.html" });
  });
}
