const fs = require('fs');
const path = require('path');
const puppeteer = require('F:/zillow_actualizado/.work/tools/slides/node_modules/puppeteer-core');

const DIR = 'F:/zillow_actualizado/.work/previews/slides';
const OUT = path.join(DIR, 'png');
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const files = fs.readdirSync(DIR).filter((f) => f.endsWith('.html')).sort();
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--hide-scrollbars', '--font-render-hinting=none']
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1334, height: 750, deviceScaleFactor: 2 });
  for (const f of files) {
    const url = 'file:///' + path.join(DIR, f).replace(/\\/g, '/');
    await page.goto(url, { waitUntil: 'load' });
    await new Promise((r) => setTimeout(r, 250));
    const out = path.join(OUT, f.replace('.html', '.png'));
    await page.screenshot({ path: out, clip: { x: 0, y: 0, width: 1333.33, height: 750 } });
    console.log('captured', path.basename(out));
  }
  await browser.close();
})();
