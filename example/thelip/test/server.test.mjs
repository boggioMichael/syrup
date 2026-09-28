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
import fs from "node:fs";
import os from "node:os";
// The site as served: the page plus a server.json that points at the fake server (discovery, no ?server=).
const site = fs.mkdtempSync(path.join(os.tmpdir(), "thelip-site-"));
fs.copyFileSync(path.join(root, pagePath), path.join(site, "index.html"));
fs.writeFileSync(path.join(site, "server.json"), JSON.stringify({ url: "http://127.0.0.1:8798" }));
const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "thelip-data-"));
const web = spawn("python3", ["-m", "http.server", "8795", "--bind", "127.0.0.1"], { cwd: root, stdio: "ignore" });
const web2 = spawn("python3", ["-m", "http.server", "8796", "--bind", "127.0.0.1"], { cwd: site, stdio: "ignore" });
const api = spawn("python3", [path.resolve(here, "..", "..", "thelip-server", "server.py"), "--fake", "--port", "8798", "--token", "t0k"], { stdio: "ignore", env: { ...process.env, THELIP_DATA: dataDir } });
const wait = async (url) => { for (let i = 0; i < 100; i++) { try { if ((await fetch(url)).ok) return; } catch {} await new Promise((r) => setTimeout(r, 200)); } throw new Error(`not up: ${url}`); };
let browser, failures = 0;
const check = (name, ok, detail = "") => { console.log((ok ? "ok   " : "FAIL ") + name + (ok || !detail ? "" : `: ${detail}`)); if (!ok) failures++; };
try {
  await wait(`http://127.0.0.1:8795${pagePath}`);
  await wait("http://127.0.0.1:8798/health");
  browser = await chromium.launch({ channel: "chromium", headless: true, args: [
    "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-video-capture=${path.join(here, "sbwe5n.y4m")}`,
  ] });
  const context = await browser.newContext({ viewport: { width: 390, height: 800 }, permissions: ["camera", "microphone"] });
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
    if (/^fake en reading of \d+ frames$/.test(text)) break;
    await new Promise((r) => setTimeout(r, 250));
  }
  const debug = await page.evaluate(() => window.lipLive.debug);
  console.log("     debug:", JSON.stringify(debug));
  check("the frames went to the server and its answer is the subtitle", /^fake en reading of \d+ frames$/.test(text), text);
  const n = +(text.match(/(\d+) frames/) || [])[1];
  check("a sentence's worth of frames was sent (20..250)", n >= 20 && n <= 250, String(n));
  check("no local words were shown for that utterance", debug.finished === 0 && debug.reads === 0, JSON.stringify(debug));
  // Stop using the server: the local reader is back.
  await page.click("#helpBtn"); await page.click("#serverOff");
  check("stopping forgets the server", await page.evaluate(() => !window.lipLive.server && localStorage.getItem("thelip.server") === null));
  check("the prompt is a GRID sentence again", /^(bin|lay|place|set) /.test(await page.$eval("#sentence", (e) => e.textContent)));
  await page.click("#closeSheet");

  // Discovery: the site's server.json names the server; no ?server= needed. The
  // token is not in it, so the fake server (which wants one) must be run open
  // for this part: use a second fake server without a token.
  const api2 = spawn("python3", [path.resolve(here, "..", "..", "thelip-server", "server.py"), "--fake", "--port", "8797"], { stdio: "ignore", env: { ...process.env, THELIP_DATA: dataDir } });
  try {
    await wait("http://127.0.0.1:8797/health");
    fs.writeFileSync(path.join(site, "server.json"), JSON.stringify({ url: "http://127.0.0.1:8797" }));
    const page2 = await context.newPage();
    page2.on("pageerror", (e) => console.error("page error:", e.message));
    await page2.goto("http://127.0.0.1:8796/index.html#g127-176-113-57");
    await page2.waitForFunction(() => window.lipLive && window.lipLive.ready, null, { timeout: 60000 });
    await page2.waitForFunction(() => window.lipLive.server === "http://127.0.0.1:8797", null, { timeout: 20000 });
    check("the page found the server named in /server.json on its own", true);
    await page2.click("#helpBtn");
    check("the sheet says it is thelip's server", /connected to thelip's server/.test(await page2.$eval("#serverStatus", (e) => e.textContent)));
    check("keeping sentences is off by default", !(await page2.$eval("#improve", (e) => e.checked)));
    const options = await page2.$$eval("#lang option", (os) => os.map((o) => [o.value, o.textContent]));
    check("the language list starts English, Hebrew, Spanish, Arabic, Chinese, French, German", options.map((o) => o[0]).join(",").startsWith("en,he,es,ar,zh,fr,de"), JSON.stringify(options));
    check("languages without a model say so", options.find((o) => o[0] === "he")[1].includes("no model yet") && !options.find((o) => o[0] === "es")[1].includes("no model"), JSON.stringify(options));
    await page2.selectOption("#lang", "es");
    check("the language is remembered and the prompt follows it", await page2.evaluate(() => localStorage.getItem("thelip.language") === "es" && window.lipLive.language === "es") && (await page2.$eval("#sentence", (e) => e.textContent)) === "anything, in Español");
    check("the sheet shows that language's quality", /44\.5%/.test(await page2.$eval("#langStatus", (e) => e.textContent)), await page2.$eval("#langStatus", (e) => e.textContent));
    await page2.click("#improve");
    check("the switch is remembered", await page2.evaluate(() => localStorage.getItem("thelip.improve") === "1"));
    await page2.waitForFunction(() => window.lipLive.mic === "running", null, { timeout: 15000 }).catch(() => {});
    check("the switch turns the microphone on", (await page2.evaluate(() => window.lipLive.mic)) === "running", await page2.$eval("#micStatus", (e) => e.textContent));
    await page2.click("#closeSheet");
    await page2.waitForFunction(() => window.lipLive.lastRead && window.lipLive.lastRead.id && window.lipLive.lastHeard, null, { timeout: 90000 });
    const read = await page2.evaluate(() => window.lipLive.lastRead);
    check("with the switch on, the server kept a sample and the page holds its id", /^[0-9a-f]{32}$/.test(read.id), JSON.stringify(read));
    check("the read went in the chosen language", /^FAKE ES READING/.test(read.raw), read.raw);
    const heard = await page2.evaluate(() => window.lipLive.lastHeard);
    check("the sound of the utterance was heard by the server and joined the sample", heard && /^FAKE HEARD \d+\.\ds$/.test(heard.heard) && (heard.id === null || heard.id === read.id) && heard.seconds > 0.3, JSON.stringify(heard));
    await page2.waitForFunction(() => window.lipLive.heard, null, { timeout: 5000 }).catch(() => {});
    check("what was heard shows under the subtitle", /🎤 FAKE HEARD/.test(await page2.evaluate(() => window.lipLive.heard || "")), await page2.evaluate(() => window.lipLive.heard));
    // Tap the subtitle, fix the text, send it. (The fake camera keeps talking, so
    // the subtitle is read the moment the fix lands, before the next sentence.)
    await page2.evaluate(() => window.lipLive.openFix());
    await page2.fill(".fix input", "hello there");
    const shown = await page2.evaluate(() => { const i = document.querySelector(".fix input"); i.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true })); return document.querySelector("#sub").textContent.trim(); });
    check("the subtitle shows the correction", shown === "hello there", shown);
    await page2.waitForFunction(() => window.lipLive.debug.server.lastFeedback && window.lipLive.debug.server.lastFeedback.corrected === "hello there", null, { timeout: 10000 });
    await new Promise((r) => setTimeout(r, 800));
    const meta = JSON.parse(fs.readFileSync(path.join(dataDir, read.id, "meta.json"), "utf8"));
    check("the correction reached the kept sample, with the language", meta.corrected === "hello there" && meta.raw === read.raw && meta.language === "es", JSON.stringify(meta));
    check("the sample carries what was heard, with its confidence", /^FAKE HEARD/.test(meta.heard || "") && meta.heard_confidence === 0.9, JSON.stringify(meta));
    check("the sample holds the crops", fs.existsSync(path.join(dataDir, read.id, "crops.npy")));

    // A server a link set (a throwaway tunnel, usually) that has died: thelip's server takes its place.
    await page2.evaluate(() => { localStorage.setItem("thelip.server", "http://127.0.0.1:8799"); localStorage.setItem("thelip.token", "old"); });
    await page2.reload();
    await page2.waitForFunction(() => window.lipLive && window.lipLive.ready, null, { timeout: 60000 });
    await page2.waitForFunction(() => window.lipLive.server === "http://127.0.0.1:8797", null, { timeout: 20000 });
    check("a dead server from a link gives way to thelip's server, and is forgotten",
          await page2.evaluate(() => localStorage.getItem("thelip.server") === null && localStorage.getItem("thelip.token") === null && window.lipLive.language === "es"));
    check("the sheet says so", /connected to thelip's server/.test(await page2.$eval("#serverStatus", (e) => e.textContent)));
  } finally { api2.kill(); }
  console.log(failures ? `${failures} FAILED` : "ALL PASSED");
  if (failures) process.exit(1);
} finally {
  if (browser) await browser.close();
  web.kill(); web2.kill(); api.kill();
  fs.rmSync(site, { recursive: true, force: true }); fs.rmSync(dataDir, { recursive: true, force: true });
}
