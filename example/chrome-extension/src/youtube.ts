/**
 * Everything that touches YouTube's page is here, and nowhere else, so a
 * change in their markup is a change in this one file. The rest of the
 * extension sees a `Player`: a video element, a place for a button, a
 * place for an overlay, and navigation events.
 */

export interface Player {
  video: HTMLVideoElement;
  container: HTMLElement;      // positioned ancestor the overlay is placed in
  controls: HTMLElement | null; // where the Lip Read button goes
}

const SELECTORS = {
  player: "#movie_player",
  video: "video.html5-main-video, #movie_player video, video",
  controls: ".ytp-right-controls",
  container: ".html5-video-container",
};

export function findPlayer(root: Document = document): Player | null {
  const playerRoot = (root.querySelector(SELECTORS.player) as HTMLElement | null) ?? root.body;
  const videos = Array.from(playerRoot.querySelectorAll<HTMLVideoElement>(SELECTORS.video));
  const video = videos.sort((a, b) => b.clientWidth * b.clientHeight - a.clientWidth * a.clientHeight)[0];
  if (!video) return null;
  const container =
    (video.closest(SELECTORS.container) as HTMLElement | null) ??
    (video.parentElement as HTMLElement | null) ??
    playerRoot;
  const controls = playerRoot.querySelector(SELECTORS.controls) as HTMLElement | null;
  return { video, container, controls };
}

/** The watch URL without playlist noise, or null when this is not a watch page. */
export function pageUrl(loc: Location = location): string | null {
  const url = new URL(loc.href);
  if (url.hostname.endsWith("youtube.com")) {
    const id = url.searchParams.get("v");
    if (!id) return null;
    return `https://www.youtube.com/watch?v=${id}`;
  }
  return url.href;
}

/**
 * A direct media URL when the video element has one the service could fetch
 * itself (a plain <video src="…mp4">). YouTube's streams are blob: or
 * googlevideo URLs bound to the session, so for YouTube this is null and
 * the service fetches by page URL.
 */
export function directMediaUrl(video: HTMLVideoElement): string | null {
  const src = video.currentSrc || video.src;
  if (!src || src.startsWith("blob:") || /googlevideo\.com/.test(src)) return null;
  return src;
}

/** YouTube is a single-page app: this fires when the watched video changes. */
export function onNavigate(handler: () => void): () => void {
  const listener = () => handler();
  document.addEventListener("yt-navigate-finish", listener);
  window.addEventListener("popstate", listener);
  return () => {
    document.removeEventListener("yt-navigate-finish", listener);
    window.removeEventListener("popstate", listener);
  };
}

/** Where the video's pixels sit inside the element (letterboxing). */
export function contentRect(video: HTMLVideoElement): { left: number; top: number; width: number; height: number; scale: number } {
  const w = video.clientWidth;
  const h = video.clientHeight;
  const vw = video.videoWidth || w;
  const vh = video.videoHeight || h;
  const scale = Math.min(w / vw, h / vh);
  const width = vw * scale;
  const height = vh * scale;
  return { left: (w - width) / 2, top: (h - height) / 2, width, height, scale };
}

export function makeButton(onClick: () => void): HTMLButtonElement {
  const button = document.createElement("button");
  button.className = "ytp-button lipread-button";
  button.title = "Lip Read: subtitles from the lips";
  button.setAttribute("aria-label", "Lip Read");
  button.textContent = "LIP";
  button.addEventListener("click", (e) => {
    e.stopPropagation();
    onClick();
  });
  return button;
}
