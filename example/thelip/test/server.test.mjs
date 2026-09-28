// Server mode: the page opened with ?server= keeps the address, reports the
// server's health in its sheet, and on the fake camera's utterance sends the
// frames to the server (thelip-server --fake here) and shows what it answers.
//   NODE_PATH=$(npm root -g) node test/server.test.mjs
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const here = path.dirname(fileURLToPath(import.meta.url));
const root = process.env.SERVE_DIR || path.resolve(here, "..");
const pagePath = process.env.PAGE_PATH || "/thelip.syrup.html";
const web = spawn("python3", ["-m", "http.server", "8795", "--bind", "127.0.0.1"], { cwd: root, stdio: "ignore" });
const api = spawn("python3", [path.resolve(here, "..", "..", "thelip-server", "server.py"), "--fake", "--port", "8798", "--token", "t0k"], { stdio: "ignore" });
const wait = async (url) => { for (let i = 0; i < 100; i++) { try { if ((await fetch(url)).ok) return; } catch {} await new Promise((r) => setTimeout(r, 200)); } throw new Error(`not up: ${url}`); };
let browser, failures = 0;
const check = (name, ok, detail = "") => { console.log((ok ? "ok   " : "FAIL ") + name + (ok || !detail ? "" : `: ${detail}`)); if (!ok) failures++; };
try {
  await wait(`http://127.0.0.1:8795${pagePath}`);
  await wait("http://127.0.0.1:8798/health");
  browser = await chromium.launch({ channel: "chromium", headless: true, args: [
    "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-video-capture=${path.join(here, "sbwe5n.y4m")}`,
  ] });
  const context = await browser.newContext({ viewport: { width: 390, height: 800 }, permissions: ["camera"] });
  const page = await context.newPage();
  page.on("pageerror", (e) => console.error("page error:", e.message));
  await page.goto(`http://127.0.0.1:8795${pagePath}?server=http://127.0.0.1:8798&token=t0k#g127-176-113-57`);
  await page.waitForFunction(() => window.lipLive && window.lipLive.ready, null, { timeout: 60000 });
  await page.waitForFunction(() => window.lipLive.server, null, { timeout: 20000 });
  check("the address bar keeps the guide but drops the server", await page.evaluate(() => location.search === "" && location.hash === "#g127-176-113-57"));
  check("the server is kept for next time", await page.evaluate(() => localStorage.getItem("thelip.server") === "http://127.0.0.1:8798" && localStorage.getItem("thelip.token") === "t0k"));
  await page.click("#helpBtn");
  const status = await page.$eval("#serverStatus", (e) => e.textContent);
  check("the sheet shows the server's health", /connected: .*Auto-AVSR.* on none/.test(status), status);
  check("the prompt says any English", (await page.$eval("#sentence", (e) => e.textContent)) === "anything, in English");
  await page.click("#closeSheet");
  await page.waitForFunction(() => !document.querySelector("#cam").hidden && document.querySelector("#cam").videoWidth > 0, null, { timeout: 20000 });
  const deadline = Date.now() + 90000;
  let text = "";
  while (Date.now() < deadline) {
    text = await page.$eval("#sub", (e) => e.textContent.trim());
    if (/^fake reading of \d+ frames$/.test(text)) break;
    await new Promise((r) => setTimeout(r, 250));
  }
  const debug = await page.evaluate(() => window.lipLive.debug);
  console.log("     debug:", JSON.stringify(debug));
  check("the frames went to the server and its answer is the subtitle", /^fake reading of \d+ frames$/.test(text), text);
  const n = +(text.match(/(\d+) frames/) || [])[1];
  check("a sentence's worth of frames was sent (20..250)", n >= 20 && n <= 250, String(n));
  check("no local words were shown for that utterance", debug.finished === 0 && debug.reads === 0, JSON.stringify(debug));
  // Stop using the server: the local reader is back.
  await page.click("#helpBtn"); await page.click("#serverOff");
  check("stopping forgets the server", await page.evaluate(() => !window.lipLive.server && localStorage.getItem("thelip.server") === null));
  check("the prompt is a GRID sentence again", /^(bin|lay|place|set) /.test(await page.$eval("#sentence", (e) => e.textContent)));
  await page.click("#closeSheet");
  console.log(failures ? `${failures} FAILED` : "ALL PASSED");
  if (failures) process.exit(1);
} finally {
  if (browser) await browser.close();
  web.kill(); api.kill();
}
