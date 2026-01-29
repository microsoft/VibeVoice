const { chromium } = require('playwright');
const fs = require('fs');

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();

  const mockVoices = {
    voices: [
      { id: 'carter', name: 'Carter', language: 'English', gender: 'male', path: '/voices/carter.pt' },
      { id: 'emma', name: 'Emma', language: 'English', gender: 'female', path: '/voices/emma.pt' },
    ],
  };

  await page.route('**/api/voices', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(mockVoices) }));
  await page.route('**/api/tts/convert', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ success: true }) }));
  await page.route('**/api/export/download', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ success: true }) }));
  await page.route('**/static/audio/output.wav', (route) => route.fulfill({ status: 200, body: Buffer.from([0,1,2]) , contentType: 'audio/wav' }));
  await page.route('**/api/tts/preview/purge', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ success: true }) }));

  await page.goto('http://127.0.0.1:3000');
  await page.waitForFunction(() => document.documentElement.hasAttribute('data-theme'));

  // upload file
  const input = await page.$('input[type="file"]');
  await input.setInputFiles({ name: 'demo.md', mimeType: 'text/markdown', buffer: Buffer.from('# title\nHello') });
  await page.waitForTimeout(2000);
  await page.screenshot({ path: 'debug-page.png', fullPage: true });

  const buttons = await page.$$eval('button, [role="button"]', els => els.map(e => ({ text: e.innerText, aria: e.getAttribute('aria-label') }))); 
  console.log('Found buttons:', buttons);
  fs.writeFileSync('debug-buttons.json', JSON.stringify(buttons, null, 2));

  // keep browser open for manual inspection
  console.log('Screenshot saved to debug-page.png and buttons to debug-buttons.json. Close manually to end.');
})();