// Content scripts cannot be ES modules; this loads the real one as a module
// from the extension's own files (listed as web accessible resources).
(async () => {
  try {
    await import(chrome.runtime.getURL("dist/content.js"));
  } catch (e) {
    console.warn("lipread: could not load", e);
  }
})();
