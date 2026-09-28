// End-to-end: the built extension in Chromium against the real inference API
// and a fake YouTube page playing a fixture with known words.
//
//   node test/e2e.mjs            (needs: dist/ built, fixtures made, playwright + chromium)
//
// The extension is copied to a temp dir with the fake page's origin added to
// its content-script matches; nothing else differs from what ships.
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const here = path.dirname(fileURLToPath(import.meta.url));
const extensionDir = path.resolve(here, "..");
const fixtures = path.resolve(here, "../../lipreader/tests/fixtures");
const apiDir = path.resolve(here, "../../inference-api");
const API_PORT = 8765;
const PAGE_PORT = 8791;

function log(...args) {
  console.log(new Date().toISOString().slice(11, 19), ...args);
}

async function waitFor(url, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(url);
      if (r.ok) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error(`not reachable: ${url}`);
}

function startProcess(cmd, args, opts) {
  const child = spawn(cmd, args, { stdio: ["ignore", "pipe", "pipe"], ...opts });
  child.stdout.on("data", (d) => process.env.E2E_VERBOSE && process.stdout.write(`[${path.basename(args[0] || cmd)}] ${d}`));
  child.stderr.on("data", (d) => process.env.E2E_VERBOSE && process.stderr.write(`[${path.basename(args[0] || cmd)}] ${d}`));
  return child;
}

function assert(condition, message) {
  if (!condition) throw new Error(`assertion failed: ${message}`);
  log("ok  ", message);
}

async function main() {
  assert(fs.existsSync(path.join(extensionDir, "dist", "worker.js")), "extension is built (tsc -p tsconfig.json)");
  assert(fs.existsSync(path.join(fixtures, "two_alternating.mp4")), "fixtures exist (python3 tests/make_fixtures.py)");

  // The extension, with the fake page's origin allowed for the content script.
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "lipread-ext-"));
  const extCopy = path.join(tmp, "extension");
  fs.cpSync(extensionDir, extCopy, { recursive: true, filter: (src) => !src.includes(`${path.sep}test${path.sep}`) && !src.endsWith(`${path.sep}test`) });
  const manifest = JSON.parse(fs.readFileSync(path.join(extCopy, "manifest.json"), "utf8"));
  const local = `http://127.0.0.1:${PAGE_PORT}/*`;
  manifest.content_scripts[0].matches.push(local);
  manifest.web_accessible_resources[0].matches.push(local);
  fs.writeFileSync(path.join(extCopy, "manifest.json"), JSON.stringify(manifest, null, 2));

  // Chromium (no proprietary codecs) plays WebM, not H.264: transcode the fixture for the page.
  const media = path.join(tmp, "media");
  fs.mkdirSync(media);
  const ff = spawn("ffmpeg", ["-v", "error", "-y", "-i", path.join(fixtures, "two_alternating.mp4"), "-c:v", "libvpx-vp9", "-b:v", "1M",
    "-deadline", "realtime", "-cpu-used", "8", "-c:a", "libopus", path.join(media, "two_alternating.webm")], { stdio: "inherit" });
  await new Promise((resolve, reject) => ff.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg exited ${code}`)))));

  const api = startProcess("python3", ["-m", "lipreader_api", "--port", String(API_PORT)], { cwd: apiDir, env: { ...process.env, LIPREADER_RESULT_TTL_MINUTES: "5" } });
  const page = startProcess("python3", ["test/fake_youtube.py", "--port", String(PAGE_PORT), "--media", media], { cwd: extensionDir });
  let context;
  try {
    await waitFor(`http://127.0.0.1:${API_PORT}/health`);
    await waitFor(`http://127.0.0.1:${PAGE_PORT}/watch?v=two_alternating`);
    log("services up");

    context = await chromium.launchPersistentContext(path.join(tmp, "profile"), {
      channel: "chromium",
      headless: true,
      args: [`--disable-extensions-except=${extCopy}`, `--load-extension=${extCopy}`, "--autoplay-policy=no-user-gesture-required"],
    });
    let [worker] = context.serviceWorkers();
    if (!worker) worker = await context.waitForEvent("serviceworker");
    const extensionId = new URL(worker.url()).host;
    log("extension", extensionId);

    const tab = await context.newPage();
    await tab.goto(`http://127.0.0.1:${PAGE_PORT}/watch?v=two_alternating`);
    await tab.waitForSelector("#movie_player video");
    await tab.waitForFunction(() => document.querySelector("video")?.readyState >= 1);
    const button = await tab.waitForSelector(".lipread-button", { timeout: 15000 });
    assert(button, "the Lip Read button is in the player controls");
    const overlay = await tab.waitForSelector(".lipread-overlay");
    assert(overlay, "the overlay is attached to the player");

    // Start the analysis from the player button.
    await button.click();
    await tab.waitForFunction(() => /lip reading/.test(document.querySelector(".lipread-status")?.textContent ?? ""), null, { timeout: 15000 });
    assert(true, "the status shows lip reading in progress");
    await tab.waitForFunction(() => document.querySelectorAll(".lipread-box").length === 2, null, { timeout: 180000 });
    assert(true, "two people are boxed when the analysis is done");

    // Subtitles follow the video clock: the left person speaks first, the right one after 3 s.
    const setTime = (t) => tab.evaluate((t) => { const v = document.querySelector("video"); v.pause(); v.currentTime = t; }, t);
    await setTime(1.6);
    await tab.waitForFunction(() => document.querySelector(".lipread-text")?.textContent?.includes("set blue"), null, { timeout: 5000 });
    const line1 = await tab.$eval(".lipread-line", (el) => el.textContent);
    assert(/Person \d/.test(line1) && /set blue/.test(line1), `subtitle at 1.6 s names a person and reads "set blue…": ${line1}`);
    await setTime(4.6);
    await tab.waitForFunction(() => document.querySelector(".lipread-text")?.textContent?.includes("bin red"), null, { timeout: 5000 });
    assert(true, "subtitle at 4.6 s reads the second person");
    const speakingBox = await tab.$eval(".lipread-box.lipread-speaking", (el) => el.dataset.track);
    assert(speakingBox, `the speaking person's box is highlighted (track ${speakingBox})`);
    const shots = process.env.E2E_SCREENSHOTS;
    if (shots) {
      fs.mkdirSync(shots, { recursive: true });
      await tab.locator("#movie_player").screenshot({ path: path.join(shots, "extension-overlay.png") });
    }

    // Click a face to follow only that person.
    const boxes = await tab.$$(".lipread-box");
    const leftBox = (await Promise.all(boxes.map(async (b) => ({ b, x: parseFloat(await b.evaluate((e) => e.style.left)) })))).sort((a, b) => a.x - b.x)[0].b;
    const leftId = await leftBox.evaluate((e) => e.dataset.track);
    await leftBox.click();
    await tab.waitForFunction((id) => document.querySelector(".lipread-status")?.textContent?.includes(`following Person ${id}`), leftId, { timeout: 5000 });
    await setTime(4.6);
    await new Promise((r) => setTimeout(r, 300));
    const linesWhileFollowing = await tab.$$eval(".lipread-line", (els) => els.length);
    assert(linesWhileFollowing === 0, "following the left person hides the right person's subtitle");
    await setTime(1.6);
    await tab.waitForFunction(() => document.querySelectorAll(".lipread-line").length === 1, null, { timeout: 5000 });
    assert(true, "and shows the followed person's subtitle");
    await leftBox.click(); // release
    await tab.waitForFunction(() => !(document.querySelector(".lipread-status")?.textContent ?? "").includes("following"), null, { timeout: 5000 });

    // The side panel, opened as a page for the same tab.
    const tabId = await worker.evaluate(async (port) => {
      const tabs = await chrome.tabs.query({ url: `http://127.0.0.1:${port}/*` });
      return tabs[0].id;
    }, PAGE_PORT);
    const panel = await context.newPage();
    await panel.goto(`chrome-extension://${extensionId}/panel.html?tabId=${tabId}`);
    await panel.waitForSelector(".seg", { timeout: 10000 });
    const segs = await panel.$$eval(".seg", (els) => els.map((e) => ({ who: e.querySelector(".who").textContent, time: e.querySelector(".time").textContent, lang: e.querySelector(".lang").textContent, conf: e.querySelector(".conf").textContent, text: e.querySelector(".text").textContent.trim() })));
    assert(segs.length === 2, `the panel lists two transcript lines: ${JSON.stringify(segs)}`);
    assert(segs.every((s) => /Person \d/.test(s.who) && /\d:\d\d\.\d/.test(s.time) && s.lang === "en" && /%$/.test(s.conf)), "each line shows speaker, timestamps, language and confidence");
    const status = await panel.$eval("#status", (e) => e.textContent);
    assert(/2 people/.test(status) && /processed on this machine/.test(status), `the panel says where processing happened: ${status}`);
    const notice = await panel.$eval("#notice", (e) => e.textContent);
    assert(/probabilistic/.test(notice), "the probabilistic notice is shown");
    if (shots) {
      await panel.setViewportSize({ width: 380, height: 640 });
      await panel.screenshot({ path: path.join(shots, "extension-panel.png") });
    }

    // Clicking a line seeks the video.
    await setTime(0);
    await panel.$$eval(".seg", (els) => els[els.length - 1].click());
    await tab.waitForFunction(() => document.querySelector("video").currentTime > 2.5, null, { timeout: 5000 });
    assert(true, "clicking a transcript line seeks the video to its start");

    // Exports.
    for (const format of ["srt", "vtt", "json", "txt"]) {
      const [download] = await Promise.all([panel.waitForEvent("download", { timeout: 15000 }), panel.click(`#export-${format}`)]);
      const file = await download.path();
      const text = fs.readFileSync(file, "utf8");
      const okBody = format === "srt" ? /-->/.test(text) && /Person/.test(text)
        : format === "vtt" ? text.startsWith("WEBVTT")
        : format === "json" ? JSON.parse(text).segments.length === 2
        : /probabilistic/.test(text);
      if (!okBody) log("export body was:", JSON.stringify(text.slice(0, 300)), "name", download.suggestedFilename());
      assert(okBody && download.suggestedFilename().endsWith(`.${format}`), `export ${format} downloads a well-formed file`);
    }

    // Pause / resume and stop from the panel.
    await panel.click("#stop");
    await tab.waitForFunction(() => document.querySelectorAll(".lipread-box").length === 0, null, { timeout: 5000 });
    assert(true, "stop clears the overlay");
    await panel.click("#start");
    await panel.waitForFunction(() => /Reading lips|Starting/.test(document.querySelector("#status").textContent), null, { timeout: 10000 });
    await panel.click("#pause");
    await panel.waitForFunction(() => document.querySelector("#status").textContent === "Paused.", null, { timeout: 5000 });
    assert(true, "pause is reflected in the panel");
    await panel.click("#pause"); // resume
    await panel.waitForFunction(() => /2 people/.test(document.querySelector("#status").textContent), null, { timeout: 180000 });
    assert(true, "resume finishes the analysis");

    log("ALL PASSED");
  } finally {
    if (context) await context.close().catch(() => undefined);
    api.kill();
    page.kill();
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
