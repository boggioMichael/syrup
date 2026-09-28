/**
 * The service worker: the only part that talks to the lipreader service.
 * It keeps one analysis state per tab, mirrors it to the content script
 * (overlay) and the side panel (transcript), and survives being unloaded by
 * keeping the state in session storage.
 */
import {
  loadSettings,
  type AnalysisResult,
  type Job,
  type Message,
  type Options,
  type Settings,
  type TabState,
  type TranscriptSegment,
} from "./shared.js";

const states = new Map<number, TabState>();
const polling = new Set<number>();

chrome.runtime.onInstalled.addListener(() => {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => undefined);
});

// --- persistence -----------------------------------------------------------------

async function restore(): Promise<void> {
  const stored = (await chrome.storage.session.get("tabs")).tabs as Record<string, TabState> | undefined;
  if (stored) for (const [id, s] of Object.entries(stored)) states.set(Number(id), s);
}
const ready = restore();

async function persist(): Promise<void> {
  const tabs: Record<string, TabState> = {};
  for (const [id, s] of states) tabs[id] = s;
  await chrome.storage.session.set({ tabs });
}

function getState(tabId: number): TabState {
  let s = states.get(tabId);
  if (!s) {
    s = { tabId, phase: "idle", progress: 0 };
    states.set(tabId, s);
  }
  return s;
}

async function update(tabId: number, patch: Partial<TabState>): Promise<TabState> {
  const s = Object.assign(getState(tabId), patch);
  await persist();
  const message: Message = { type: "state", state: s };
  chrome.tabs.sendMessage(tabId, message).catch(() => undefined);
  chrome.runtime.sendMessage(message).catch(() => undefined);
  return s;
}

// --- the service -------------------------------------------------------------------

class Api {
  constructor(private settings: Settings) {}

  private headers(extra: Record<string, string> = {}): Record<string, string> {
    const h: Record<string, string> = { ...extra };
    if (this.settings.apiToken) h.Authorization = `Bearer ${this.settings.apiToken}`;
    return h;
  }

  private url(path: string): string {
    return this.settings.apiUrl.replace(/\/$/, "") + path;
  }

  async health(): Promise<{ ok: boolean; processing: "local" | "server"; fetchers: string[]; languages: unknown[] }> {
    const r = await fetch(this.url("/health"), { headers: this.headers() });
    if (!r.ok) throw new Error(`service answered ${r.status}`);
    return r.json();
  }

  async createJob(url: string, options: Options): Promise<Job> {
    const r = await fetch(this.url("/jobs"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({ url, options }),
    });
    const body = await r.json();
    if (!r.ok) throw new Error(body.error ?? `service answered ${r.status}`);
    return body as Job;
  }

  async job(id: string, wait = 20): Promise<Job> {
    const r = await fetch(this.url(`/jobs/${id}?wait=${wait}`), { headers: this.headers() });
    const body = await r.json();
    if (!r.ok) throw new Error(body.error ?? `service answered ${r.status}`);
    return body as Job;
  }

  async deleteJob(id: string): Promise<void> {
    await fetch(this.url(`/jobs/${id}`), { method: "DELETE", headers: this.headers() }).catch(() => undefined);
  }

  async exportJob(id: string, format: string, tracks: number[] | null): Promise<string> {
    const params = new URLSearchParams({ format });
    for (const t of tracks ?? []) params.append("track", String(t));
    const r = await fetch(this.url(`/jobs/${id}/export?${params}`), { headers: this.headers() });
    if (!r.ok) throw new Error(`export failed (${r.status})`);
    return r.text();
  }

  async createSession(width: number, height: number, fps: number, options: Options, source: string): Promise<{ id: string }> {
    const r = await fetch(this.url("/sessions"), {
      method: "POST",
      headers: this.headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({ width, height, fps, options, source }),
    });
    const body = await r.json();
    if (!r.ok) throw new Error(body.error ?? `service answered ${r.status}`);
    return body;
  }

  async pushFrames(sessionId: string, frames: Blob[], timestamps: number[]): Promise<{ segments: TranscriptSegment[]; tracks: unknown[] }> {
    const form = new FormData();
    frames.forEach((f, i) => form.append("frames", f, `f${i}.jpg`));
    form.append("timestamps", JSON.stringify(timestamps));
    const r = await fetch(this.url(`/sessions/${sessionId}/frames`), { method: "POST", headers: this.headers(), body: form });
    const body = await r.json();
    if (!r.ok) throw new Error(body.error ?? `service answered ${r.status}`);
    return body;
  }

  async finishSession(sessionId: string): Promise<{ result: AnalysisResult }> {
    const r = await fetch(this.url(`/sessions/${sessionId}/finish`), { method: "POST", headers: this.headers() });
    const body = await r.json();
    if (!r.ok) throw new Error(body.error ?? `service answered ${r.status}`);
    return body;
  }

  async deleteSession(sessionId: string): Promise<void> {
    await fetch(this.url(`/sessions/${sessionId}`), { method: "DELETE", headers: this.headers() }).catch(() => undefined);
  }
}

function optionsFrom(settings: Settings, currentTime?: number): Options {
  return {
    mode: settings.mode,
    language: settings.language,
    speakers: settings.speakers,
    ...(settings.speakers === "current" && currentTime != null ? { startTime: Math.max(0, currentTime) } : {}),
  };
}

// --- URL path: the service fetches the video -------------------------------------------

async function start(tabId: number, pageUrl: string, mediaUrl: string | null, currentTime?: number): Promise<void> {
  const settings = await loadSettings();
  const api = new Api(settings);
  const state = getState(tabId);
  if (state.jobId) await api.deleteJob(state.jobId);
  await update(tabId, { phase: "starting", progress: 0, error: undefined, result: undefined, partialSegments: undefined, jobId: undefined, pageUrl, mediaUrl: mediaUrl ?? undefined });
  try {
    const health = await api.health();
    const options = optionsFrom(settings, currentTime);
    const target = mediaUrl ?? pageUrl;
    if (!mediaUrl && /youtube\.com|youtu\.be/.test(pageUrl) && !health.fetchers.includes("youtube")) {
      throw new Error("the service cannot fetch YouTube videos (install yt-dlp on that machine), or switch the source to tab capture in the options");
    }
    const job = await api.createJob(target, options);
    await update(tabId, { phase: "running", jobId: job.id, processing: job.processing, options });
    void poll(tabId, api);
  } catch (e) {
    await update(tabId, { phase: "failed", error: (e as Error).message });
  }
}

async function poll(tabId: number, api: Api): Promise<void> {
  if (polling.has(tabId)) return;
  polling.add(tabId);
  try {
    for (;;) {
      const state = states.get(tabId);
      if (!state || !state.jobId || state.phase === "paused" || state.phase === "idle") return;
      const job = await api.job(state.jobId, 20);
      if (job.status === "done" && job.result) {
        await update(tabId, { phase: "done", progress: 1, result: job.result, processing: job.processing });
        return;
      }
      if (job.status === "failed" || job.status === "deleted") {
        await update(tabId, { phase: "failed", error: job.error ?? job.status });
        return;
      }
      await update(tabId, { progress: job.progress, processing: job.processing });
    }
  } catch (e) {
    await update(tabId, { phase: "failed", error: (e as Error).message });
  } finally {
    polling.delete(tabId);
  }
}

async function stop(tabId: number, forget = true): Promise<void> {
  const settings = await loadSettings();
  const api = new Api(settings);
  const state = states.get(tabId);
  if (!state) return;
  if (state.jobId) await api.deleteJob(state.jobId);
  if (state.sessionId) {
    await stopCapture();
    await api.deleteSession(state.sessionId);
  }
  if (forget) {
    // Back to idle, but the page stays known so the panel can start again.
    await update(tabId, { phase: "idle", progress: 0, jobId: undefined, sessionId: undefined, result: undefined,
                          partialSegments: undefined, error: undefined, followTrackId: null, options: undefined });
  }
}

function forget(tabId: number): void {
  states.delete(tabId);
  void persist();
}

// --- capture path: the tab is captured and streamed --------------------------------------
// Not exercised in the build environment (no YouTube, no capturable tab): see docs.

let capture: { tabId: number; sessionId: string; api: Api; offset: number | null } | null = null;

async function startCapture(tabId: number, width: number, height: number, currentTime?: number): Promise<void> {
  const settings = await loadSettings();
  const api = new Api(settings);
  await update(tabId, { phase: "starting", progress: 0, error: undefined, result: undefined, partialSegments: [] });
  try {
    const fps = 12.5;
    const session = await api.createSession(width, height, fps, optionsFrom(settings, currentTime), "tab-capture");
    const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tabId });
    await chrome.offscreen.createDocument({
      url: "capture.html",
      reasons: [chrome.offscreen.Reason.USER_MEDIA],
      justification: "Capture the tab's video frames for lip reading on the local service",
    });
    capture = { tabId, sessionId: session.id, api, offset: null };
    await update(tabId, { phase: "running", sessionId: session.id, processing: "local" });
    chrome.runtime.sendMessage({ type: "capture-start", tabId, streamId, fps, sessionId: session.id } satisfies Message).catch(() => undefined);
  } catch (e) {
    await update(tabId, { phase: "failed", error: (e as Error).message });
  }
}

async function onFrames(sessionId: string, frames: string[], timestamps: number[]): Promise<void> {
  if (!capture || capture.sessionId !== sessionId) return;
  const { tabId, api } = capture;
  try {
    // Map capture time to the player's clock once, then assume continuous playback.
    if (capture.offset == null) {
      const reply = (await chrome.tabs.sendMessage(tabId, { type: "current-time" } satisfies Message)) as { time: number };
      capture.offset = reply.time - timestamps[0];
    }
    const blobs = await Promise.all(frames.map(async (dataUrl) => (await fetch(dataUrl)).blob()));
    const snapshot = await api.pushFrames(sessionId, blobs, timestamps.map((t) => t + (capture!.offset ?? 0)));
    await update(tabId, { partialSegments: snapshot.segments, progress: 0.5 });
  } catch (e) {
    await update(tabId, { phase: "failed", error: (e as Error).message });
    await stopCapture();
  }
}

async function finishCapture(): Promise<void> {
  if (!capture) return;
  const { tabId, api, sessionId } = capture;
  await stopCapture();
  try {
    const { result } = await api.finishSession(sessionId);
    await update(tabId, { phase: "done", progress: 1, result });
  } catch (e) {
    await update(tabId, { phase: "failed", error: (e as Error).message });
  }
}

async function stopCapture(): Promise<void> {
  capture = null;
  chrome.runtime.sendMessage({ type: "capture-stop" } satisfies Message).catch(() => undefined);
  await chrome.offscreen.closeDocument().catch(() => undefined);
}

// --- messages --------------------------------------------------------------------------------

type ExportRequest = { type: "export"; tabId: number; format?: string; tracks?: number[] | null };

chrome.runtime.onMessage.addListener((message: Message | ExportRequest, sender, respond) => {
  (async () => {
    await ready;
    if (message.type === "export") {
      const s = getState(message.tabId);
      if (!s.jobId) throw new Error("nothing to export yet");
      const api = new Api(await loadSettings());
      const text = await api.exportJob(s.jobId, message.format ?? "srt", message.tracks ?? null);
      respond({ ok: true, text });
      return;
    }
    const tabId = ("tabId" in message ? message.tabId : undefined) ?? sender.tab?.id;
    switch (message.type) {
      case "get-state":
        if (tabId != null) respond({ state: getState(tabId) });
        else respond({ state: null });
        return;
      case "page-ready":
        if (tabId != null) {
          const s = getState(tabId);
          if (s.pageUrl && s.pageUrl !== message.pageUrl) await stop(tabId);
          await update(tabId, { pageUrl: message.pageUrl, mediaUrl: message.mediaUrl ?? undefined });
        }
        break;
      case "start": {
        if (tabId == null) break;
        const s = getState(tabId);
        const settings = await loadSettings();
        const pageUrl = message.pageUrl ?? s.pageUrl ?? "";
        const mediaUrl = message.mediaUrl ?? s.mediaUrl ?? null;
        if (settings.source === "capture" && !mediaUrl) await startCapture(tabId, message.width ?? 0, message.height ?? 0, message.currentTime);
        else await start(tabId, pageUrl, mediaUrl, message.currentTime);
        break;
      }
      case "pause":
        if (tabId != null) {
          const s = getState(tabId);
          if (s.phase === "running" || s.phase === "starting") await update(tabId, { phase: "paused" });
        }
        break;
      case "resume":
        if (tabId != null) {
          const s = getState(tabId);
          if (s.phase === "paused" && s.jobId) {
            await update(tabId, { phase: "running" });
            void poll(tabId, new Api(await loadSettings()));
          } else if (s.phase === "paused" && s.sessionId) {
            await finishCapture();
          }
        }
        break;
      case "stop":
        if (tabId != null) await stop(tabId);
        break;
      case "follow":
        if (tabId != null) await update(tabId, { followTrackId: message.trackId });
        break;
      case "seek":
        if (tabId != null) chrome.tabs.sendMessage(tabId, message).catch(() => undefined);
        break;
      case "frames":
        await onFrames(message.sessionId, message.frames, message.timestamps);
        break;
      default:
        break;
    }
    respond({ ok: true });
  })().catch((e) => respond({ ok: false, error: (e as Error).message }));
  return true;
});

chrome.tabs.onRemoved.addListener((tabId) => {
  void stop(tabId, false).then(() => forget(tabId));
});
