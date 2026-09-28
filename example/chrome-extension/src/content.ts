/**
 * The content script: a Lip Read button in the player, an overlay that
 * draws the subtitles (and, optionally, a box and label per person) in
 * sync with the video's clock, and click-to-follow one person. All
 * network work happens in the service worker; this file only asks it to
 * start/stop and receives state.
 */
import { boxAt, segmentsAt, type Message, type TabState, type TranscriptSegment } from "./shared.js";
import { contentRect, directMediaUrl, findPlayer, makeButton, onNavigate, pageUrl, type Player } from "./youtube.js";

let player: Player | null = null;
let overlay: HTMLDivElement | null = null;
let state: TabState | null = null;
let showBoxes = true;
let raf = 0;

function send(message: Message): void {
  chrome.runtime.sendMessage(message).catch(() => undefined);
}

function ensureOverlay(p: Player): HTMLDivElement {
  if (overlay && overlay.isConnected) return overlay;
  overlay = document.createElement("div");
  overlay.className = "lipread-overlay";
  overlay.dataset.lipread = "overlay";
  const boxes = document.createElement("div");
  boxes.className = "lipread-boxes";
  const subtitles = document.createElement("div");
  subtitles.className = "lipread-subtitles";
  const status = document.createElement("div");
  status.className = "lipread-status";
  overlay.append(boxes, subtitles, status);
  if (getComputedStyle(p.container).position === "static") p.container.style.position = "relative";
  p.container.appendChild(overlay);
  overlay.addEventListener("click", (e) => {
    const target = (e.target as HTMLElement).closest<HTMLElement>(".lipread-box");
    if (!target) return;
    e.stopPropagation();
    const id = Number(target.dataset.track);
    const follow = state?.followTrackId === id ? null : id;
    send({ type: "follow", trackId: follow });
  });
  return overlay;
}

function ensureButton(p: Player): void {
  if (!p.controls || p.controls.querySelector(".lipread-button")) return;
  const button = makeButton(() => {
    if (!player) return;
    if (state && (state.phase === "running" || state.phase === "starting")) {
      send({ type: "pause" });
    } else if (state && state.phase === "paused") {
      send({ type: "resume" });
    } else {
      send({
        type: "start",
        pageUrl: pageUrl() ?? location.href,
        mediaUrl: directMediaUrl(player.video),
        width: player.video.videoWidth,
        height: player.video.videoHeight,
        currentTime: player.video.currentTime,
      });
    }
  });
  p.controls.prepend(button);
}

function render(): void {
  raf = requestAnimationFrame(render);
  if (!player || !overlay) return;
  const video = player.video;
  const t = video.currentTime;
  const boxesEl = overlay.querySelector<HTMLDivElement>(".lipread-boxes")!;
  const subsEl = overlay.querySelector<HTMLDivElement>(".lipread-subtitles")!;
  const statusEl = overlay.querySelector<HTMLDivElement>(".lipread-status")!;
  const button = player.controls?.querySelector<HTMLButtonElement>(".lipread-button");
  if (!state || state.phase === "idle") {
    boxesEl.replaceChildren();
    subsEl.replaceChildren();
    statusEl.textContent = "";
    if (button) button.classList.remove("lipread-active");
    return;
  }
  if (button) button.classList.toggle("lipread-active", state.phase === "running" || state.phase === "starting");
  const segments: TranscriptSegment[] = state.result?.segments ?? state.partialSegments ?? [];
  const tracks = state.result?.tracks ?? [];
  // Status line: what is happening, and where.
  if (state.phase === "starting" || state.phase === "running") {
    statusEl.textContent = `lip reading… ${Math.round(state.progress * 100)}%${state.processing === "server" ? " (on the server)" : ""}`;
  } else if (state.phase === "failed") {
    statusEl.textContent = `lip reading failed: ${state.error ?? "unknown error"}`;
  } else if (state.phase === "paused") {
    statusEl.textContent = "lip reading paused";
  } else {
    statusEl.textContent = state.followTrackId != null ? `following Person ${state.followTrackId} (click the box to release)` : "";
  }
  // Subtitles for this instant, one line per person.
  const current = segmentsAt(segments, t, state.followTrackId);
  const lines = current.map((s) => {
    const line = document.createElement("div");
    line.className = "lipread-line";
    line.dataset.track = String(s.trackId);
    const who = document.createElement("span");
    who.className = "lipread-who";
    who.textContent = `Person ${s.trackId}`;
    const text = document.createElement("span");
    text.className = "lipread-text";
    const words = s.words.filter((w) => w.start <= t + 0.05);
    text.textContent = (words.length ? words : s.words).map((w) => (w.uncertain ? `[${w.text}?]` : w.text)).join(" ");
    const conf = document.createElement("span");
    conf.className = "lipread-conf";
    conf.textContent = `${Math.round(s.confidence * 100)}%`;
    line.append(who, text, conf);
    return line;
  });
  const key = lines.map((l) => l.textContent).join("\n");
  if (subsEl.dataset.key !== key) {
    subsEl.dataset.key = key;
    subsEl.replaceChildren(...lines);
  }
  // Boxes and labels that follow each person: one element per track, moved
  // in place (rebuilding them every frame would break clicks on them).
  const wanted = new Set<string>();
  const rect = contentRect(video);
  if (showBoxes) {
    for (const track of tracks) {
      if (t < track.firstSeen - 0.2 || t > track.lastSeen + 0.2) continue;
      const box = boxAt(track, t);
      if (!box) continue;
      const id = String(track.trackId);
      wanted.add(id);
      let el = boxesEl.querySelector<HTMLDivElement>(`.lipread-box[data-track="${id}"]`);
      if (!el) {
        el = document.createElement("div");
        el.className = "lipread-box";
        el.dataset.track = id;
        const label = document.createElement("span");
        label.className = "lipread-label";
        label.textContent = track.label;
        el.appendChild(label);
        boxesEl.appendChild(el);
      }
      el.classList.toggle("lipread-followed", state.followTrackId === track.trackId);
      el.classList.toggle("lipread-speaking", current.some((s) => s.trackId === track.trackId));
      el.style.left = `${rect.left + box.x * rect.scale}px`;
      el.style.top = `${rect.top + box.y * rect.scale}px`;
      el.style.width = `${box.w * rect.scale}px`;
      el.style.height = `${box.h * rect.scale}px`;
    }
  }
  for (const el of Array.from(boxesEl.children)) {
    if (!wanted.has((el as HTMLElement).dataset.track ?? "")) el.remove();
  }
}

function attach(): void {
  const found = findPlayer();
  if (!found) return;
  player = found;
  ensureOverlay(found);
  ensureButton(found);
  send({
    type: "page-ready",
    pageUrl: pageUrl() ?? location.href,
    mediaUrl: directMediaUrl(found.video),
    width: found.video.videoWidth,
    height: found.video.videoHeight,
  });
  if (!raf) raf = requestAnimationFrame(render);
}

chrome.runtime.onMessage.addListener((message: Message, _sender, respond) => {
  if (message.type === "state") {
    state = message.state;
  } else if (message.type === "seek" && player) {
    player.video.currentTime = message.time;
    if (player.video.paused) player.video.play().catch(() => undefined);
  } else if (message.type === "current-time" && player) {
    respond({ time: player.video.currentTime, paused: player.video.paused });
    return true;
  }
  respond?.({ ok: true });
  return false;
});

chrome.storage.sync.get("settings").then((v) => {
  showBoxes = (v.settings?.showBoxes ?? true) as boolean;
});
chrome.storage.onChanged.addListener((changes) => {
  if (changes.settings?.newValue) showBoxes = changes.settings.newValue.showBoxes ?? true;
});

// YouTube builds the player after the page loads and swaps videos in place.
attach();
onNavigate(() => {
  state = null;
  send({ type: "stop" });
  setTimeout(attach, 500);
});
const observer = new MutationObserver(() => {
  if (!player || !player.video.isConnected) attach();
  else if (player.controls && !player.controls.querySelector(".lipread-button")) ensureButton(player);
});
observer.observe(document.documentElement, { childList: true, subtree: true });
send({ type: "get-state" });
