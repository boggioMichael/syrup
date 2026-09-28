// The built page in Chromium: a silent GRID clip through the file path,
// the guide placed on the mouth, the sentence read.
//   NODE_PATH=$(npm root -g) node test/page.test.mjs
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { spawn, execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const here = path.dirname(fileURLToPath(import.meta.url));
// SERVE_DIR / PAGE_PATH test another copy, e.g. the standalone build in docs/.
const root = process.env.SERVE_DIR || path.resolve(here, "..");
const pagePath = process.env.PAGE_PATH || "/thelip.syrup.html";
const server = spawn("python3", ["-m", "http.server", "8793", "--bind", "127.0.0.1"], { cwd: root, stdio: "ignore" });
const wait = async (url) => { for (let i = 0; i < 100; i++) { try { if ((await fetch(url)).ok) return; } catch {} await new Promise((r) => setTimeout(r, 200)); } throw new Error(`not up: ${url}`); };
let browser;
try {
  await wait(`http://127.0.0.1:8793${pagePath}`);
  browser = await chromium.launch({ channel: "chromium", headless: true });
  const page = await browser.newPage({ viewport: { width: 420, height: 860 } });
  page.on("pageerror", (e) => console.error("page error:", e.message));
  page.on("console", (m) => { if (m.type() === "error") console.error("console:", m.text()); });
  // The GRID mouth of sbwe5n sits at (127,176) 113x57 in the 360x288 frame.
  await page.goto(`http://127.0.0.1:8793${pagePath}#g127-176-113-57`);
  await page.waitForFunction(() => window.lipLive && window.lipLive.ready, null, { timeout: 60000 });
  // No camera in headless Chromium: the page must say so and offer the clip.
  await page.waitForFunction(() => document.querySelector("#msg") && !document.querySelector("#msg").hidden && /camera/i.test(document.querySelector("#msg").textContent), null, { timeout: 20000 });
  const speed = await page.$eval("#speed", (e) => e.textContent);
  console.log("ok   thelip ready in the worker:", speed);
  await page.setInputFiles("#file2", path.join(here, "sbwe5n.webm"));
  await page.waitForFunction(() => document.querySelector("#dot").dataset.state === "idle" && document.querySelector("#sub").textContent.trim().length > 0, null, { timeout: 180000 });
  await new Promise((r) => setTimeout(r, 500));
  const text = await page.$eval("#sub", (e) => e.textContent.trim());
  console.log(`ok   reading "${text}"`);
  const right = "set blue with e five now".split(" ").filter((w, i) => text.split(" ")[i] === w).length;
  if (right < 5) { console.error(`FAIL: expected "set blue with e five now", got "${text}"`); process.exit(1); }
  await page.screenshot({ path: path.join(here, "page.png"), fullPage: true });

  // With no oval given, the page finds the mouth by itself (BlazeFace, in its own worker).
  const auto = await browser.newPage({ viewport: { width: 420, height: 860 } });
  auto.on("pageerror", (e) => console.error("page error:", e.message));
  await auto.goto(`http://127.0.0.1:8793${pagePath}`);
  await auto.waitForFunction(() => window.lipLive && window.lipLive.ready && window.lipLive.faceReady, null, { timeout: 60000 });
  console.log(`ok   the face finder is ready: ${await auto.evaluate(() => window.lipLive.debug.faceMs)} ms a look`);
  if (!(await auto.evaluate(() => window.lipLive.auto))) { console.error("FAIL: the oval does not find the mouth by default"); process.exit(1); }
  const grid = path.join(here, "..", "..", "..", "docs", "test", "grid");
  const SENTENCES = { bbaf2n: "bin blue at f two now", brbk7n: "bin red by k seven now", lbax4n: "lay blue at x four now", lbbc2a: "lay blue by c two again",
    lrwp9a: "lay red with p nine again", lwbsza: "lay white by s zero again", pwij3p: "place white in j three please", sbia1a: "set blue in a one again",
    sbwe5n: "set blue with e five now", swiz3n: "set white in z three now" };
  const readClip = async (file) => {
    await auto.evaluate(() => { document.querySelector("#sub").replaceChildren(); });
    await auto.setInputFiles("#file2", file);
    await auto.waitForFunction(() => document.querySelector("#dot").dataset.state === "idle" && document.querySelector("#sub").textContent.trim().length > 0 && !/opening/.test(document.querySelector("#sub").textContent), null, { timeout: 180000 });
    await new Promise((r) => setTimeout(r, 300));
    return { text: (await auto.$eval("#sub", (e) => e.textContent.trim())).replace(/[[\]?]/g, ""), guide: await auto.evaluate(() => window.lipLive.guide), mouths: await auto.evaluate(() => window.lipLive.mouths) };
  };
  const first = await readClip(path.join(grid, "sbwe5n.webm"));
  // The oval placed by hand for this clip (the test above): (127,176) 113x57.
  const g = first.guide, off = Math.hypot(g.x + g.w / 2 - 183.5, g.y + g.h / 2 - 204.5);
  if (off > 8 || Math.abs(g.w - 113) > 20) { console.error(`FAIL: the oval found ${JSON.stringify(g)}, not the mouth at (127,176) 113x57`); process.exit(1); }
  console.log(`ok   the oval found the mouth by itself: ${JSON.stringify(g)} (by hand: 127,176 113x57; ${off.toFixed(1)} px apart)`);
  let wordsRight = 0, total = 0;
  for (const [clip, said] of Object.entries(SENTENCES)) {
    const r = clip === "sbwe5n" ? first : await readClip(path.join(grid, clip + ".webm"));
    const got = r.text.split(/\s+/), want = said.split(" ");
    const hits = want.filter((w, i) => got[i] === w).length;
    wordsRight += hits; total += want.length;
    console.log(`     ${clip}: "${r.text}" (${hits}/${want.length}), oval ${JSON.stringify(r.guide)}`);
  }
  console.log(`${wordsRight >= 57 ? "ok  " : "FAIL"} GRID's sample clips with the oval found by itself: ${wordsRight} of ${total} words`);
  if (wordsRight < 57) process.exit(1);

  // Two people: the larger face's mouth is the oval, the other's gets an oval of its own.
  const two = path.join(os.tmpdir(), "thelip-two-grid.webm");
  execFileSync("ffmpeg", ["-v", "error", "-y", "-i", path.join(grid, "bbaf2n.webm"), "-i", path.join(grid, "sbwe5n.webm"), "-filter_complex",
    "[0:v]crop=180:288:63:0[a];[1:v]crop=180:288:93:0[b];[a][b]hstack=inputs=2", "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30", two]);
  const both = await readClip(two);
  const mc = (b) => [b.x + b.w / 2, b.y + b.h / 2];
  const sides = [mc(both.guide)[0] < 180, both.mouths.length === 1 && mc(both.mouths[0])[0] < 180];
  if (both.mouths.length !== 1 || sides[0] === sides[1]) { console.error(`FAIL: two faces: oval ${JSON.stringify(both.guide)}, other mouths ${JSON.stringify(both.mouths)}`); process.exit(1); }
  console.log(`ok   two faces: the oval on one mouth ${JSON.stringify(both.guide)}, the other mouth marked ${JSON.stringify(both.mouths[0])}`);
  await auto.screenshot({ path: path.join(here, "auto.png") });
  console.log("ALL PASSED");
} finally {
  if (browser) await browser.close();
  server.kill();
}
