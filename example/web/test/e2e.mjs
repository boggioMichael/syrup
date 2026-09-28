// The web page against the real service: upload a fixture, wait for the
// transcript, check the lanes, the segments and an export.
//   NODE_PATH=$(npm root -g) node test/e2e.mjs
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const here = path.dirname(fileURLToPath(import.meta.url));
const webDir = path.resolve(here, "..");
const fixtures = path.resolve(here, "../../lipreader/tests/fixtures");
const apiDir = path.resolve(here, "../../inference-api");

const wait = async (url) => { for (let i = 0; i < 100; i++) { try { if ((await fetch(url)).ok) return; } catch {} await new Promise((r) => setTimeout(r, 300)); } throw new Error(`not up: ${url}`); };
const assert = (c, m) => { if (!c) throw new Error(`assertion failed: ${m}`); console.log("ok  ", m); };

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "lipread-web-"));
const webm = path.join(tmp, "two_simultaneous.webm");
await new Promise((res, rej) => spawn("ffmpeg", ["-v", "error", "-y", "-i", path.join(fixtures, "two_simultaneous.mp4"), "-c:v", "libvpx-vp9", "-b:v", "1M", "-deadline", "realtime", "-cpu-used", "8", "-c:a", "libopus", webm], { stdio: "inherit" }).on("exit", (c) => (c === 0 ? res() : rej(new Error("ffmpeg")))));
const api = spawn("python3", ["-m", "lipreader_api", "--port", "8765"], { cwd: apiDir, stdio: "ignore" });
const web = spawn("python3", ["-m", "http.server", "8792", "--bind", "127.0.0.1"], { cwd: webDir, stdio: "ignore" });
let browser;
try {
  await wait("http://127.0.0.1:8765/health");
  await wait("http://127.0.0.1:8792/index.html");
  browser = await chromium.launch({ channel: "chromium", headless: true });
  const page = await browser.newPage();
  await page.goto("http://127.0.0.1:8792/index.html");
  await page.waitForFunction(() => /Visual models here: en/.test(document.querySelector("#languages").textContent), null, { timeout: 10000 });
  assert(true, "the page shows the service's honest language table");
  await page.setInputFiles("#file", webm);
  await page.click("button.primary");
  await page.waitForFunction(() => /people/.test(document.querySelector("#status").textContent), null, { timeout: 180000 });
  const status = await page.$eval("#status", (e) => e.textContent);
  assert(/2 people, 2 segments/.test(status) && /on this machine/.test(status), `status: ${status}`);
  const lanes = await page.$$eval(".lane", (els) => els.length);
  assert(lanes === 2, "a timeline lane per person");
  const segs = await page.$$eval("#transcript .seg", (els) => els.map((e) => e.textContent));
  assert(segs.length === 2 && segs.some((s) => /set blue with e five now/.test(s)) && segs.some((s) => /bin red by k seven now/.test(s)), `segments: ${JSON.stringify(segs)}`);
  await page.click("#transcript .seg");
  await page.waitForFunction(() => document.querySelector("video").currentTime > 0, null, { timeout: 5000 });
  assert(true, "clicking a segment seeks the video");
  const [download] = await Promise.all([page.waitForEvent("download"), page.click("#exports a:nth-child(3)")]);
  const text = fs.readFileSync(await download.path(), "utf8");
  assert(/-->/.test(text) && download.suggestedFilename() === "lipread.srt", "SRT export downloads");
  await page.click("#delete");
  await page.waitForFunction(() => /Deleted/.test(document.querySelector("#status").textContent), null, { timeout: 5000 });
  assert(true, "delete removes the result from the service");
  console.log("ALL PASSED");
} finally {
  if (browser) await browser.close();
  api.kill(); web.kill();
  fs.rmSync(tmp, { recursive: true, force: true });
}
