/**
 * What the three parts of the extension say to each other (content script,
 * service worker, side panel) and the little runtime shared between them.
 * Result types come from the shared contract.
 */
import type { AnalysisResult, Box, Job, Mode, Options, PersonTrack, Speakers, TranscriptSegment } from "../../shared-types/dist/index";

export type { AnalysisResult, Box, Job, Mode, Options, PersonTrack, Speakers, TranscriptSegment };

export interface Settings {
  apiUrl: string;
  apiToken: string;
  mode: Mode;
  language: string;
  speakers: "all" | "current";
  showBoxes: boolean;
  /** "url": the service fetches the video (yt-dlp); "capture": the tab is captured and streamed */
  source: "url" | "capture";
}

export const DEFAULT_SETTINGS: Settings = {
  apiUrl: "http://127.0.0.1:8765",
  apiToken: "",
  mode: "visual",
  language: "auto",
  speakers: "all",
  showBoxes: true,
  source: "url",
};

export async function loadSettings(): Promise<Settings> {
  const stored = (await chrome.storage.sync.get("settings")).settings as Partial<Settings> | undefined;
  return { ...DEFAULT_SETTINGS, ...(stored ?? {}) };
}

export async function saveSettings(settings: Settings): Promise<void> {
  await chrome.storage.sync.set({ settings });
}

/** Per-tab analysis state, owned by the worker, mirrored to the UI. */
export interface TabState {
  tabId: number;
  phase: "idle" | "starting" | "running" | "paused" | "done" | "failed";
  jobId?: string;
  sessionId?: string;
  progress: number;
  processing?: "local" | "server";
  result?: AnalysisResult;
  /** partial results while a captured session is running */
  partialSegments?: TranscriptSegment[];
  error?: string;
  followTrackId?: number | null;
  pageUrl?: string;
  mediaUrl?: string;
  options?: Options;
}

export type Message =
  | { type: "page-ready"; pageUrl: string; mediaUrl: string | null; width: number; height: number }
  | { type: "start"; tabId?: number; pageUrl?: string; mediaUrl?: string | null; width?: number; height?: number; currentTime?: number }
  | { type: "pause"; tabId?: number }
  | { type: "resume"; tabId?: number }
  | { type: "stop"; tabId?: number }
  | { type: "follow"; tabId?: number; trackId: number | null }
  | { type: "seek"; tabId?: number; time: number }
  | { type: "get-state"; tabId?: number }
  | { type: "state"; state: TabState }
  | { type: "current-time"; tabId?: number }
  | { type: "frames"; sessionId: string; frames: string[]; timestamps: number[] }
  | { type: "capture-start"; tabId: number; streamId: string; fps: number; sessionId: string }
  | { type: "capture-stop" };

export function boxAt(track: PersonTrack, t: number): Box | null {
  const k = track.keyframes;
  if (k.length === 0) return null;
  if (t <= k[0].t) return k[0].box;
  if (t >= k[k.length - 1].t) return k[k.length - 1].box;
  let lo = 0;
  let hi = k.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (k[mid].t <= t) lo = mid;
    else hi = mid;
  }
  const a = k[lo];
  const b = k[hi];
  const f = b.t === a.t ? 0 : (t - a.t) / (b.t - a.t);
  return {
    x: a.box.x + (b.box.x - a.box.x) * f,
    y: a.box.y + (b.box.y - a.box.y) * f,
    w: a.box.w + (b.box.w - a.box.w) * f,
    h: a.box.h + (b.box.h - a.box.h) * f,
  };
}

export function segmentsAt(segments: TranscriptSegment[], t: number, follow: number | null | undefined): TranscriptSegment[] {
  return segments.filter((s) => s.start <= t && t <= s.end && (follow == null || s.trackId === follow));
}

export function trackLabel(result: AnalysisResult | undefined, trackId: number): string {
  const track = result?.tracks.find((t) => t.trackId === trackId);
  return track?.label ?? (trackId < 0 ? "Unattributed" : `Person ${trackId}`);
}

export const NOTICE =
  "Lip-reading output is probabilistic. Many sounds look identical on the lips, so this transcript is a best guess, not a verbatim record; words marked [like this?] are uncertain.";
