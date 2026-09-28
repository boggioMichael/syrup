// The live camera path: Chromium's fake camera plays a silent GRID clip on a
// loop, the page opens it on load and must read the sentence from the lips.
//   NODE_PATH=$(npm root -g) node test/camera.test.mjs
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const here = path.dirname(fileURLToPath(import.meta.url));
const root = process.env.SERVE_DIR || path.resolve(here, "..");
const pagePath = process.env.PAGE_PATH || "/thelip.syrup.html";
const server = spawn("python3", ["-m", "http.server", "8794", "--bind", "127.0.0.1"], { cwd: root, stdio: "ignore" });
const wait = async (url) => { for (let i = 0; i < 100; i++) { try { if ((await fetch(url)).ok) return; } catch {} await new Promise((r) => setTimeout(r, 200)); } throw new Error(`not up: ${url}`); };
let browser;
try {
  await wait(`http://127.0.0.1:8794${pagePath}`);
  browser = await chromium.launch({ channel: "chromium", headless: true, args: [
    "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-video-capture=${path.join(here, "sbwe5n.y4m")}`,
  ] });
  const context = await browser.newContext({ viewport: { width: 390, height: 800 }, permissions: ["camera"] });
  // No oval given: the page finds the mouth in the camera's picture by itself and puts
  // the oval there (the fake camera is GRID's sbwe5n: the mouth by hand at (127,176) 113x57).
  const auto = await context.newPage();
  auto.on("pageerror", (e) => console.error("page error:", e.message));
  await auto.goto(`http://127.0.0.1:8794${pagePath}`);
  await auto.waitForFunction(() => window.lipLive && window.lipLive.ready && window.lipLive.faceReady, null, { timeout: 60000 });
  await auto.waitForFunction(() => { const g = window.lipLive.guide; return document.querySelector("#guide").classList.contains("auto") && g && Math.hypot(g.x + g.w / 2 - 183.5, g.y + g.h / 2 - 204.5) < 10; }, null, { timeout: 20000 }).catch(() => {});
  const g = await auto.evaluate(() => window.lipLive.guide), following = await auto.$eval("#guide", (e) => e.classList.contains("auto"));
  const off = g ? Math.hypot(g.x + g.w / 2 - 183.5, g.y + g.h / 2 - 204.5) : Infinity;
  if (!following || off >= 10) { console.error(`FAIL: the oval did not find the mouth in the camera's picture: ${JSON.stringify(g)}`); process.exit(1); }
  console.log(`ok   the oval found the mouth in the camera's picture by itself: ${JSON.stringify(g)} (${off.toFixed(1)} px from the one placed by hand)`);
  // Dragging it places it by hand: it stays there, and the choice is kept.
  const box = await auto.$eval("#guide", (e) => { const r = e.getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; });
  await auto.mouse.move(box.x, box.y); await auto.mouse.down(); await auto.mouse.move(box.x + 40, box.y + 30, { steps: 5 }); await auto.mouse.up();
  await new Promise((r) => setTimeout(r, 800));
  const moved = await auto.evaluate(() => ({ auto: window.lipLive.auto, kept: localStorage.getItem("thelip.auto"), guide: window.lipLive.guide, following: document.querySelector("#guide").classList.contains("auto") }));
  if (moved.auto || moved.kept !== "0" || moved.following || Math.abs(moved.guide.x - g.x) < 10) { console.error(`FAIL: dragging did not place the oval by hand: ${JSON.stringify(moved)}`); process.exit(1); }
  console.log(`ok   dragging the oval places it by hand, and it stays: ${JSON.stringify(moved.guide)}`);
  await auto.evaluate(() => localStorage.removeItem("thelip.auto"));
  await auto.close();
  const page = await context.newPage();
  page.on("pageerror", (e) => console.error("page error:", e.message));
  await page.goto(`http://127.0.0.1:8794${pagePath}#g127-176-113-57`);
  await page.waitForFunction(() => window.lipLive && window.lipLive.ready, null, { timeout: 60000 });
  await page.waitForFunction(() => !document.querySelector("#cam").hidden && document.querySelector("#cam").videoWidth > 0, null, { timeout: 20000 });
  console.log("ok   the camera opened on load, no button pressed");
  const seen = new Set();
  const deadline = Date.now() + 90000;
  let hit = null;
  while (Date.now() < deadline) {
    const text = await page.$eval("#sub", (e) => e.textContent.trim());
    if (text) seen.add(text);
    const right = "set blue with e five now".split(" ").filter((w, i) => text.split(" ")[i] === w).length;
    if (right >= 5 && (await page.$eval("#dot", (e) => e.dataset.state)) === "idle") { hit = text; break; }
    await new Promise((r) => setTimeout(r, 250));
  }
  console.log("     subtitles seen:", [...seen].slice(-6));
  console.log("     debug:", JSON.stringify(await page.evaluate(() => window.lipLive.debug)), "video", await page.$eval("#cam", (v) => `${v.videoWidth}x${v.videoHeight} t=${v.currentTime.toFixed(1)}`));
  if (!hit) { console.error("FAIL: the sentence was not read from the live camera"); process.exit(1); }
  console.log(`ok   read live from the camera: "${hit}"`);
  await page.screenshot({ path: path.join(here, "camera.png") });

  console.log("ALL PASSED");
} finally {
  if (browser) await browser.close();
  server.kill();
}
