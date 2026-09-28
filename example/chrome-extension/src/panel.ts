/**
 * The side panel: controls, the transcript with speaker, time, language
 * and confidence, click-to-seek, export. It reflects the worker's state
 * for the tab it was opened on.
 */
import { loadSettings, saveSettings, trackLabel, type Message, type Mode, type TabState } from "./shared.js";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
let tabId: number | null = null;
let state: TabState | null = null;

async function currentTabId(): Promise<number | null> {
  const fromQuery = new URLSearchParams(location.search).get("tabId");
  if (fromQuery) return Number(fromQuery);
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab?.id ?? null;
}

function send(message: Message | { type: "export"; tabId: number; format: string; tracks: number[] | null }): Promise<any> {
  return chrome.runtime.sendMessage(message).catch(() => ({ ok: false }));
}

function fmt(t: number): string {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

function render(): void {
  const status = $("status");
  const list = $<HTMLDivElement>("transcript");
  const start = $<HTMLButtonElement>("start");
  const pause = $<HTMLButtonElement>("pause");
  const stop = $<HTMLButtonElement>("stop");
  const exportBox = $("export");
  const s = state;
  if (!s || s.phase === "idle") {
    status.textContent = "Open a video and press Lip Read.";
    status.dataset.phase = "idle";
    list.replaceChildren();
    start.disabled = false;
    pause.disabled = true;
    stop.disabled = true;
    exportBox.hidden = true;
    return;
  }
  status.dataset.phase = s.phase;
  const where = s.processing === "server" ? " · processed on the server" : s.processing === "local" ? " · processed on this machine" : "";
  switch (s.phase) {
    case "starting":
      status.textContent = "Starting…";
      break;
    case "running":
      status.textContent = `Reading lips… ${Math.round(s.progress * 100)}%${where}`;
      break;
    case "paused":
      status.textContent = "Paused.";
      break;
    case "failed":
      status.textContent = `Failed: ${s.error}`;
      break;
    case "done": {
      const r = s.result!;
      const lang = r.language.used ?? "none";
      const how = r.language.detection === "assumed" ? " (assumed: no visual language identification exists)" : r.language.detection === "audio-detected" ? " (from the audio)" : "";
      status.textContent = `${r.tracks.length} ${r.tracks.length === 1 ? "person" : "people"}, ${r.segments.length} segments · language ${lang}${how} · mode ${r.mode}${where}`;
      break;
    }
  }
  start.disabled = s.phase === "running" || s.phase === "starting";
  pause.disabled = !(s.phase === "running" || s.phase === "starting" || s.phase === "paused");
  pause.textContent = s.phase === "paused" ? "Resume" : "Pause";
  stop.disabled = false;
  exportBox.hidden = s.phase !== "done";
  const segments = s.result?.segments ?? s.partialSegments ?? [];
  const warnings = s.result?.warnings ?? [];
  const rows = segments
    .filter((seg) => s.followTrackId == null || seg.trackId === s.followTrackId)
    .map((seg) => {
      const row = document.createElement("div");
      row.className = "seg";
      row.dataset.start = String(seg.start);
      row.dataset.track = String(seg.trackId);
      row.innerHTML = `<div class="meta"><span class="who">${trackLabel(s.result, seg.trackId)}</span><span class="time">${fmt(seg.start)}–${fmt(seg.end)}</span><span class="lang">${seg.language}</span><span class="conf" title="confidence">${Math.round(seg.confidence * 100)}%</span><span class="mode">${seg.mode}</span></div>`;
      const text = document.createElement("div");
      text.className = "text";
      for (const w of seg.words) {
        const span = document.createElement("span");
        span.className = w.uncertain ? "word uncertain" : "word";
        span.title = `${Math.round(w.confidence * 100)}%${w.raw ? ` (network said "${w.raw}")` : ""}`;
        span.textContent = w.uncertain ? `[${w.text}?]` : w.text;
        text.append(span, " ");
      }
      row.appendChild(text);
      row.addEventListener("click", () => {
        if (tabId != null) void send({ type: "seek", tabId, time: seg.start });
      });
      return row;
    });
  const people = (s.result?.tracks ?? []).map((t) => {
    const chip = document.createElement("button");
    chip.className = "chip" + (s.followTrackId === t.trackId ? " on" : "");
    chip.textContent = `${t.label} (${fmt(t.firstSeen)}–${fmt(t.lastSeen)})`;
    chip.title = "show only this person";
    chip.addEventListener("click", () => {
      if (tabId != null) void send({ type: "follow", tabId, trackId: s.followTrackId === t.trackId ? null : t.trackId });
    });
    return chip;
  });
  const warn = warnings.map((w) => {
    const el = document.createElement("div");
    el.className = "warning";
    el.textContent = w;
    return el;
  });
  const peopleBox = document.createElement("div");
  peopleBox.className = "people";
  peopleBox.replaceChildren(...people);
  list.replaceChildren(peopleBox, ...warn, ...rows);
  if (rows.length === 0 && s.phase === "done") {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "Nothing could be read.";
    list.appendChild(empty);
  }
}

async function exportAs(format: string): Promise<void> {
  if (tabId == null || !state?.result) return;
  const tracks = state.followTrackId != null ? [state.followTrackId] : null;
  const reply = await send({ type: "export", tabId, format, tracks });
  if (!reply?.ok) {
    $("status").textContent = `Export failed: ${reply?.error ?? "unknown"}`;
    return;
  }
  const blob = new Blob([reply.text], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `lipread.${format}`;
  a.dataset.export = format;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(a.href);
    a.remove();
  }, 1000);
}

async function main(): Promise<void> {
  tabId = await currentTabId();
  const settings = await loadSettings();
  $<HTMLSelectElement>("mode").value = settings.mode;
  $<HTMLSelectElement>("language").value = settings.language;
  $<HTMLSelectElement>("speakers").value = settings.speakers;
  $("notice").textContent =
    "Lip-reading output is probabilistic. Many sounds look identical on the lips, so this transcript is a best guess, not a verbatim record; words marked [like this?] are uncertain.";
  for (const id of ["mode", "language", "speakers"]) {
    $<HTMLSelectElement>(id).addEventListener("change", async () => {
      const s = await loadSettings();
      s.mode = $<HTMLSelectElement>("mode").value as Mode;
      s.language = $<HTMLSelectElement>("language").value;
      s.speakers = $<HTMLSelectElement>("speakers").value as "all" | "current";
      await saveSettings(s);
    });
  }
  $("start").addEventListener("click", () => {
    if (tabId != null) void send({ type: "start", tabId });
  });
  $("pause").addEventListener("click", () => {
    if (tabId == null) return;
    void send({ type: state?.phase === "paused" ? "resume" : "pause", tabId });
  });
  $("stop").addEventListener("click", () => {
    if (tabId != null) void send({ type: "stop", tabId });
  });
  for (const format of ["txt", "json", "srt", "vtt"]) {
    $(`export-${format}`).addEventListener("click", () => void exportAs(format));
  }
  $("options").addEventListener("click", () => chrome.runtime.openOptionsPage());
  chrome.runtime.onMessage.addListener((message: Message) => {
    if (message.type === "state" && message.state.tabId === tabId) {
      state = message.state;
      render();
    }
  });
  if (tabId != null) {
    const reply = await send({ type: "get-state", tabId });
    state = reply?.state ?? null;
  }
  render();
}

void main();
