/**
 * Offscreen document: turns a tab-capture stream into JPEG frames for the
 * worker. Not exercised in the build environment (no capturable tab there);
 * written against chrome.tabCapture + getUserMedia as documented.
 */
import type { Message } from "./shared.js";

const video = document.getElementById("v") as HTMLVideoElement;
const canvas = document.getElementById("c") as HTMLCanvasElement;
let timer = 0;
let stream: MediaStream | null = null;

async function start(streamId: string, fps: number, sessionId: string): Promise<void> {
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { mandatory: { chromeMediaSource: "tab", chromeMediaSourceId: streamId } } as MediaTrackConstraints,
  });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d")!;
  const t0 = performance.now();
  let frames: string[] = [];
  let times: number[] = [];
  timer = window.setInterval(() => {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    frames.push(canvas.toDataURL("image/jpeg", 0.85));
    times.push((performance.now() - t0) / 1000);
    if (frames.length >= Math.round(fps)) {
      chrome.runtime.sendMessage({ type: "frames", sessionId, frames, timestamps: times } satisfies Message).catch(() => undefined);
      frames = [];
      times = [];
    }
  }, 1000 / fps);
}

function stop(): void {
  clearInterval(timer);
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
}

chrome.runtime.onMessage.addListener((message: Message) => {
  if (message.type === "capture-start") void start(message.streamId, message.fps, message.sessionId);
  if (message.type === "capture-stop") stop();
});
