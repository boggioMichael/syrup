/**
 * Types shared by the inference API, the Chrome extension and the web app.
 * Mirrors ../schema/*.schema.json; the Python side mirrors it in
 * example/lipreader/lipreader/schema.py. Keep the three in step (the
 * schema tests check the JSON both sides produce).
 */

export type Mode = "visual" | "audiovisual" | "audio-attributed";

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface ModelInfo {
  name: string;
  license: string;
  languages?: string[];
  vocabulary?: string;
  note?: string;
}

export interface Word {
  text: string;
  start: number;
  end: number;
  confidence: number;
  /** below the threshold: render marked, never as certain text */
  uncertain: boolean;
  raw?: string;
}

export interface TranscriptSegment {
  id: string;
  trackId: number;
  start: number;
  end: number;
  /** words joined; uncertain words wrapped as [word?] */
  text: string;
  words: Word[];
  language: string;
  confidence: number;
  mode: Mode;
  model: string;
}

export interface SpeakingSpan {
  start: number;
  end: number;
  probability: number;
}

export interface Keyframe {
  t: number;
  box: Box;
  mouth?: Box;
}

/** One visible person, numbered within this video only. No identity. */
export interface PersonTrack {
  trackId: number;
  label: string;
  firstSeen: number;
  lastSeen: number;
  keyframes: Keyframe[];
  speaking: SpeakingSpan[];
  detectedLanguage?: string | null;
  segmentIds?: string[];
  confidence: number;
}

export type LanguageDetection = "requested" | "assumed" | "audio-detected" | "unavailable";

export interface AnalysisResult {
  version: "1";
  video: {
    source?: string;
    duration: number;
    fps: number;
    width: number;
    height: number;
    sampledFps?: number;
  };
  mode: Mode;
  language: {
    requested: string;
    used: string | null;
    detection: LanguageDetection;
    note?: string;
  };
  tracks: PersonTrack[];
  segments: TranscriptSegment[];
  models: {
    faceDetector?: string;
    vsr?: ModelInfo;
    asr?: ModelInfo;
  };
  timing: {
    processingSeconds: number;
    realtimeFactor: number;
    stages?: Record<string, number>;
  };
  notice: string;
  warnings?: string[];
}

export type Speakers = "all" | "current" | number[];

export interface Options {
  mode?: Mode;
  language?: string;
  speakers?: Speakers;
  sampleFps?: number;
  maxFaces?: number;
  startTime?: number;
  endTime?: number;
  uncertainBelow?: number;
}

export type JobStatus = "queued" | "running" | "done" | "failed" | "deleted";

export interface Job {
  id: string;
  status: JobStatus;
  progress: number;
  createdAt: string;
  finishedAt?: string | null;
  expiresAt?: string | null;
  options: Options;
  result?: AnalysisResult;
  error?: string | null;
  processing?: "local" | "server";
}

export interface LanguageAvailability {
  code: string;
  name: string;
  visual: { available: boolean; model?: string; license?: string; note?: string };
  audio: { available: boolean; model?: string; note?: string };
}

export interface HealthResponse {
  ok: boolean;
  version: string;
  processing: "local" | "server";
  languages: LanguageAvailability[];
  fetchers: string[];
}

/** Keyframe box at time t, interpolated between the two nearest keyframes. */
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

export const NOTICE =
  "Lip-reading output is probabilistic. Many sounds look identical on the lips, so this transcript is a best guess, not a verbatim record; words marked [like this?] are uncertain.";
