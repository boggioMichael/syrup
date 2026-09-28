import { loadSettings, saveSettings } from "./shared.js";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

async function main(): Promise<void> {
  const s = await loadSettings();
  $<HTMLInputElement>("apiUrl").value = s.apiUrl;
  $<HTMLInputElement>("apiToken").value = s.apiToken;
  $<HTMLSelectElement>("source").value = s.source;
  $<HTMLInputElement>("showBoxes").checked = s.showBoxes;
  $<HTMLFormElement>("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const settings = await loadSettings();
    settings.apiUrl = $<HTMLInputElement>("apiUrl").value.trim() || "http://127.0.0.1:8765";
    settings.apiToken = $<HTMLInputElement>("apiToken").value.trim();
    settings.source = $<HTMLSelectElement>("source").value as "url" | "capture";
    settings.showBoxes = $<HTMLInputElement>("showBoxes").checked;
    // A service elsewhere than localhost needs permission for its origin.
    const origin = new URL(settings.apiUrl).origin + "/*";
    if (!/^(http:\/\/(127\.0\.0\.1|localhost))/.test(settings.apiUrl)) {
      const granted = await chrome.permissions.request({ origins: [origin] }).catch(() => false);
      if (!granted) {
        $("saved").textContent = `permission for ${origin} was not granted`;
        return;
      }
    }
    await saveSettings(settings);
    $("saved").textContent = "saved";
  });
}

void main();
