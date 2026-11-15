// Records the README demo from the live website: 5 short scenes, each captured once fully loaded.
import puppeteer from "puppeteer-core";

const SITE = "https://sakibhossain9999.github.io/Brain-Tumor-Segmentation-Mlops/";
const CASE = "case=BRATS_155&seg=segresnet";
const OUT = "/tmp/gif";
const W = 1280, H = 720;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: "/usr/bin/google-chrome", headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", `--window-size=${W},${H}`],
  defaultViewport: { width: W, height: H },
});

async function open(url) {
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: "networkidle0", timeout: 90000 });
  await sleep(4000); // coi-serviceworker reloads once; meshes load
  await page.waitForNetworkIdle({ idleTime: 1000, timeout: 60000 }).catch(() => {});
  return page;
}

/** Auto-rotate on; both side panels stay visible. */
async function prepareViewer(page) {
  const ok = await page.evaluate(() => {
    for (const c of document.querySelectorAll(".lil-controller")) {
      if (c.querySelector(".lil-name")?.textContent.trim() === "Auto-rotate") {
        const box = c.querySelector("input");
        if (box && !box.checked) box.click();
        return box?.checked;
      }
    }
    return false;
  });
  if (!ok) throw new Error("auto-rotate control not found");
  await sleep(800);
}

async function setSlider(page, name, value) {
  await page.evaluate((name, value) => {
    for (const c of document.querySelectorAll(".lil-controller")) {
      if (c.querySelector(".lil-name")?.textContent.trim() !== name) continue;
      const inp = c.querySelector("input");
      inp.value = String(value);
      inp.dispatchEvent(new Event("input", { bubbles: true }));
      inp.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }, name, value);
}

async function record(page, file, seconds, during) {
  const rec = await page.screencast({ path: `${OUT}/${file}.webm` });
  const t0 = Date.now();
  if (during) await during(t0);
  const left = seconds * 1000 - (Date.now() - t0);
  if (left > 0) await sleep(left);
  await rec.stop();
  await page.close();
  console.log("recorded", file);
}

const scenes = [
  ["1_toon", `${SITE}?${CASE}&style=toon`, 3.5],
  ["2_hatching", `${SITE}?${CASE}&style=hatching`, 3],
  ["3_compare", `${SITE}?${CASE}&style=toon&cmp=WT`, 3.5],
];
for (const [file, url, secs] of scenes) {
  const page = await open(url);
  await prepareViewer(page);
  await record(page, file, secs);
}

// MRI slice sweeping up through the brain, realistic tumour
{
  const page = await open(`${SITE}?${CASE}&style=realistic`);
  await prepareViewer(page);
  await setSlider(page, "Position", 0.3);
  await sleep(600);
  await record(page, "4_slice", 3, async (t0) => {
    while (Date.now() - t0 < 2900) {
      const f = (Date.now() - t0) / 2900;
      await setSlider(page, "Position", (0.3 + 0.45 * f).toFixed(3));
      await sleep(60);
    }
  });
}

// the Models page, scrolling past the charts
{
  const page = await open(`${SITE}models.html`);
  await record(page, "5_models", 2.5, async (t0) => {
    const max = await page.evaluate(() => document.body.scrollHeight - innerHeight);
    while (Date.now() - t0 < 2400) {
      const f = (Date.now() - t0) / 2400;
      await page.evaluate((y) => scrollTo(0, y), Math.round(Math.min(max, 2200) * f));
      await sleep(40);
    }
  });
}
await browser.close();
