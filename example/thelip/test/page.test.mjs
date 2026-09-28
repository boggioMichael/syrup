// The built page in Chromium: a silent GRID clip through the file path,
// the guide placed on the mouth, the sentence read.
//   NODE_PATH=$(npm root -g) node test/page.test.mjs
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
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
  console.log("ALL PASSED");
} finally {
  if (browser) await browser.close();
  server.kill();
}
