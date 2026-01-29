import { test, expect, type Page } from '@playwright/test';

const mockVoices = {
  voices: [
    { id: 'carter', name: 'Carter', language: 'English', gender: 'male', path: '/voices/carter.pt' },
    { id: 'emma', name: 'Emma', language: 'English', gender: 'female', path: '/voices/emma.pt' },
  ],
};
const UPLOAD_TIMEOUT = 15000;
const minimalWav = Buffer.from([
  0x52, 0x49, 0x46, 0x46, // RIFF
  0x24, 0x00, 0x00, 0x00, // chunk size
  0x57, 0x41, 0x56, 0x45, // WAVE
  0x66, 0x6d, 0x74, 0x20, // fmt 
  0x10, 0x00, 0x00, 0x00, // subchunk1 size
  0x01, 0x00, // audio format (PCM)
  0x01, 0x00, // channels
  0x40, 0x1f, 0x00, 0x00, // sample rate (8000)
  0x80, 0x3e, 0x00, 0x00, // byte rate (16000)
  0x02, 0x00, // block align
  0x10, 0x00, // bits per sample
  0x64, 0x61, 0x74, 0x61, // data
  0x00, 0x00, 0x00, 0x00, // data size
]);

async function uploadAndConvert(page: Page, voiceName: string, content: string) {
  await page.goto('/');
  await page.waitForFunction(() => document.documentElement.hasAttribute('data-theme'));

  const input = page.locator('input[type="file"]');
  await expect(input).toHaveCount(1);
  await input.setInputFiles({
    name: 'demo.md',
    mimeType: 'text/markdown',
    buffer: Buffer.from(content),
  });
  await expect(page.getByText('demo.md', { exact: true }).first()).toBeVisible({ timeout: UPLOAD_TIMEOUT });
  await expect(page.getByText('Voice Selection')).toBeVisible({ timeout: UPLOAD_TIMEOUT });

  // Wait for voice buttons to render (flaky on slow machines) before selecting
  await page.getByRole('button', { name: /Select voice/i }).first().waitFor({ timeout: UPLOAD_TIMEOUT });

  await page.getByRole('button', { name: new RegExp(`Select voice ${voiceName}`, 'i') }).click();

  const convertButton = page.getByRole('button', { name: 'Convert to Speech' });
  await expect(convertButton).toBeEnabled();
  await Promise.all([
    page.waitForResponse((resp) => resp.url().endsWith('/api/tts/convert') && resp.status() === 200),
    convertButton.click(),
  ]);
}

test.beforeEach(async ({ page }) => {
  await page.route('**/api/voices', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockVoices),
    });
  });

  // Also handle direct backend voice endpoint (buildApiUrl('/voices')) used by the client
  await page.route('**/voices', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockVoices),
    });
  });

  await page.route('**/api/tts/convert', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        success: true,
        message: 'Mock conversion ok',
        audio_url: '/static/audio/output.wav',
        duration: 12.5,
      }),
    });
  });

  await page.route('**/api/export/download', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        success: true,
        message: 'Mock download ok',
        download_url: '/static/audio/output.wav',
      }),
    });
  });

  // Serve the static audio file used by the mock download
  await page.route('**/static/audio/output.wav', async (route) => {
    await route.fulfill({
      status: 200,
      body: minimalWav,
      contentType: 'audio/wav',
    });
  });

  // Mock preview purge endpoint to avoid network error during session reset
  await page.route('**/api/tts/preview/purge', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ success: true, message: 'Purge scheduled' }),
    });
  });
});

test('loads dashboard and tabs', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => document.documentElement.hasAttribute('data-theme'));
  await expect(page.getByText('VibeVoice-Narrator')).toBeVisible();
  await expect(page.getByRole('tab', { name: 'Document' })).toBeVisible();
  await expect(page.getByRole('tab', { name: 'Voice' })).toBeVisible();
  await expect(page.getByRole('tab', { name: 'Settings' })).toBeVisible();
  await expect(page.getByRole('tab', { name: 'Player' })).toBeVisible();
});

test('upload flow selects voice and converts', async ({ page }) => {
  await uploadAndConvert(page, 'Carter', '# Title\n\nHello world');

  await page.getByRole('tab', { name: 'Player' }).click();
  await expect(page.locator('audio')).toHaveCount(1);
});

test('export flow uses mock API', async ({ page }) => {
  await uploadAndConvert(page, 'Emma', '# Title\n\nExport test');

  const exportButton = page.getByRole('button', { name: 'Export Audio' });
  await expect(exportButton).toBeEnabled();

  // Click and wait for download to be triggered
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    exportButton.click(),
  ]);

  const suggested = download.suggestedFilename();
  expect(suggested).toBeTruthy();
  // The filename should match pattern like 'demo.md.wav' or similar
  expect(suggested).toMatch(/^demo\.[^.]+\.wav$/);
});
