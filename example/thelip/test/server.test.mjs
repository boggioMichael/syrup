// Server mode: the page opened with ?server= keeps the address, reports the
// server's health in its sheet, and on the fake camera's utterance sends the
// frames to the server (thelip-server --fake here) and shows what it answers.
//   NODE_PATH=$(npm root -g) node test/server.test.mjs
import path from "node:path";
import { spawn, execFileSync } from "node:child_process";
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
    check("languages without a model say they learn your phrases", options.find((o) => o[0] === "ar")[1].includes("learns your phrases") && !options.find((o) => o[0] === "es")[1].includes("learns"), JSON.stringify(options));
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
    check("what was heard shows under the subtitle, word by word, with the lips' score", /🎤 heard\s*fake heard .*— \d+ of \d+/.test(await page2.evaluate(() => window.lipLive.heard || "")), await page2.evaluate(() => window.lipLive.heard));
    const align = await page2.evaluate(() => window.lipLive.debug.server.lastAlign);
    check("the lips' line is shown again against it, marked right and wrong", align && align.hits >= 1 && (await page2.evaluate(() => document.querySelectorAll("#sub .hit").length)) >= 1 && (await page2.evaluate(() => document.querySelectorAll("#sub .wrong").length)) >= 1 && /👄/.test(await page2.evaluate(() => window.lipLive.text)), JSON.stringify(align));
    await page2.screenshot({ path: path.join(here, "server.png") });
    check("the sheet keeps a running score", /the lips got \d+ of \d+ words/.test(await page2.$eval("#tally", (e) => e.textContent)), await page2.$eval("#tally", (e) => e.textContent));
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

    // A language with no model: the page sends its id, the server keeps the sentence's
    // features, the microphone labels it, and it becomes one of the reader's phrases.
    await page2.evaluate(() => window.lipLive.debug.server.lastHow = null);
    await page2.click("#helpBtn");
    await page2.selectOption("#lang", "ar");
    check("the prompt asks for a phrase", (await page2.$eval("#sentence", (e) => e.textContent)) === "a phrase in العربية");
    check("the sheet says how it learns, and that none are learned yet", /reads your own phrases.*None learned yet/.test(await page2.$eval("#langStatus", (e) => e.textContent)), await page2.$eval("#langStatus", (e) => e.textContent));
    await page2.click("#closeSheet");
    await page2.waitForFunction(() => ["none", "unmatched", "learned"].includes(window.lipLive.debug.server.lastHow) && window.lipLive.lastHeard && window.lipLive.lastRead && window.lipLive.lastRead.id, null, { timeout: 90000 });
    const profile = await page2.evaluate(() => window.lipLive.profile);
    check("the page made itself a random id", /^[0-9a-f]{32}$/.test(profile || ""), profile);
    const arRead = await page2.evaluate(() => window.lipLive.lastRead);
    const arMeta = JSON.parse(fs.readFileSync(path.join(dataDir, arRead.id, "meta.json"), "utf8"));
    check("the sentence was kept with the id and the encoder's features", arMeta.profile === profile && arMeta.language === "ar" && fs.existsSync(path.join(dataDir, arRead.id, "features.npy")), JSON.stringify(arMeta));
    await page2.waitForFunction(() => window.lipLive.phrases.count >= 1, null, { timeout: 20000 }).catch(() => {});
    const learnedList = await (await fetch(`http://127.0.0.1:8797/phrases?profile=${profile}&language=ar`)).json();
    check("what the microphone heard became one of the reader's phrases", learnedList.phrases.length >= 1 && /^FAKE HEARD/.test(learnedList.phrases[0].text), JSON.stringify(learnedList));
    check("the page counts it", (await page2.evaluate(() => window.lipLive.phrases.count)) >= 1, JSON.stringify(await page2.evaluate(() => window.lipLive.phrases)));
    await page2.evaluate(() => { const s = document.querySelector("#lang"); s.value = "es"; s.dispatchEvent(new Event("change", { bubbles: true })); });

    // Opening a clip with a server: the whole clip goes to it, whatever the oval sees.
    const clip = fs.readFileSync(path.join(here, "sbwe5n.webm")).toString("base64");
    await page2.evaluate(async (b64) => {
      window.lipLive.debug.server.lastText = null;
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      window.lipLive.startFile(new File([bytes], "clip.webm", { type: "video/webm" }));
    }, clip);
    await page2.waitForFunction(() => window.lipLive.debug.server.fileFrames && /^FAKE ES READING OF/.test(window.lipLive.debug.server.lastText || ""), null, { timeout: 90000 })
      .catch(async (e) => { console.log("file read state:", JSON.stringify(await page2.evaluate(() => ({ server: window.lipLive.debug.server, text: window.lipLive.text })))); });
    const fileRead = await page2.evaluate(() => ({ text: window.lipLive.debug.server.lastText, frames: window.lipLive.debug.server.fileFrames, shown: window.lipLive.text }));
    const sent = Number((fileRead.text.match(/OF (\d+) FRAMES/) || [])[1]);
    check("an opened clip is read on the server, all of it (~3 s = ~74 frames)", sent >= 70 && sent <= 76 && fileRead.frames >= 70 && /fake es reading/.test(fileRead.shown), JSON.stringify(fileRead));

    // Two people in a clip (the fake server sees two faces in a frame twice as wide as tall;
    // the left one's brightness moves like a speaking mouth, the right one's does not).
    const two = path.join(os.tmpdir(), "thelip-two-faces.webm");
    execFileSync("ffmpeg", ["-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=192x64:d=3:r=25,format=yuv420p,geq=lum='if(lt(X,W/2),128+70*sin(2*PI*T*3),120)':cb=128:cr=128",
      "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30", two]);
    const twoB64 = fs.readFileSync(two).toString("base64");
    await page2.evaluate(async (b64) => {
      window.lipLive.debug.server.fileFrames = null; window.lipLive.debug.server.lastFaces = null;
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      window.lipLive.startFile(new File([bytes], "two.webm", { type: "video/webm" }));
    }, twoB64);
    await page2.waitForFunction(() => window.lipLive.debug.server.fileFrames && window.lipLive.debug.server.lastFaces, null, { timeout: 90000 }).catch(() => {});
    const faceEls = await page2.$$eval("#faces .face", (els) => els.map((e) => ({ speaking: e.classList.contains("speaking"), said: e.querySelector(".said").textContent })));
    check("two faces in a clip: a frame around each, the speaking one's words under it, the other marked not speaking",
          faceEls.length === 2 && faceEls[0].speaking && /fake es face 1 reading/.test(faceEls[0].said) && !faceEls[1].speaking && /not speaking/.test(faceEls[1].said), JSON.stringify(faceEls));
    await page2.screenshot({ path: path.join(here, "faces.png") });

    // A clip recorded in a browser (MediaRecorder) does not state its duration at first.
    const recorded = await page2.evaluate(async (b64) => {
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      const v = document.createElement("video"); v.muted = true; v.src = URL.createObjectURL(new Blob([bytes], { type: "video/webm" })); document.body.appendChild(v);
      await new Promise((r) => (v.onloadedmetadata = r));
      const c = document.createElement("canvas"); c.width = v.videoWidth; c.height = v.videoHeight; const ctx = c.getContext("2d");
      const rec = new MediaRecorder(c.captureStream(25), { mimeType: "video/webm;codecs=vp8" }); const chunks = [];
      rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); }; rec.start(200);
      for (let i = 0; i < 50; i++) { await new Promise((r) => { v.addEventListener("seeked", r, { once: true }); v.currentTime = i / 25; }); ctx.drawImage(v, 0, 0); await new Promise((r) => setTimeout(r, 40)); }
      rec.stop(); await new Promise((r) => (rec.onstop = r)); v.remove();
      const blob = new Blob(chunks, { type: "video/webm" });
      const probe = document.createElement("video"); probe.muted = true; probe.src = URL.createObjectURL(blob);
      await new Promise((r) => (probe.onloadedmetadata = r));
      window.lipLive.debug.server.fileFrames = null;
      window.lipLive.startFile(new File([blob], "recorded.webm", { type: "video/webm" }));
      return { stated: String(probe.duration) };
    }, clip);
    await page2.waitForFunction(() => window.lipLive.debug.server.fileFrames, null, { timeout: 90000 }).catch(() => {});
    const recFrames = await page2.evaluate(() => window.lipLive.debug.server.fileFrames);
    check("a browser recording without a stated duration is read for its real length", recorded.stated === "Infinity" && recFrames >= 40 && recFrames <= 150, JSON.stringify({ recorded, recFrames }));

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
