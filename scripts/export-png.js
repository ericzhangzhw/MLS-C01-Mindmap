// scripts/export-png.js
const puppeteer = require("puppeteer");

(async () => {
  const browser = await puppeteer.launch({ args: ['--no-sandbox'] });
  const page = await browser.newPage();
  await page.goto(`file://${process.cwd()}/02-data-analysis/img/mindmap.html`);
  await page.setViewport({ width: 1200, height: 800 });
  await page.screenshot({ path: "02-data-analysis/img/mindmap.png" });
  await browser.close();
})();
