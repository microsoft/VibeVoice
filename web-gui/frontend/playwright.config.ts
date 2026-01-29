import { defineConfig, devices } from '@playwright/test';

const isCI = !!process.env.CI;

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  timeout: 30_000,
  expect: {
    timeout: 10_000,
  },
  // Configure retries (higher on CI)
  retries: isCI ? 2 : 0,
  forbidOnly: isCI,
  // Reporter configuration: always produce HTML, and add GitHub reporter on CI
  reporter: isCI ? [['html'], ['github']] : [['html']],
  use: {
    baseURL: 'http://127.0.0.1:3000',
    trace: 'on-first-retry',
  },

  // Define cross-browser projects. For faster local runs, run only Chromium by default.
  // Enable Firefox and WebKit in CI or when PLAYWRIGHT_BROWSERS is truthy.
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    ...((isCI || !!process.env.PLAYWRIGHT_BROWSERS)
      ? [
          {
            name: 'firefox',
            use: { ...devices['Desktop Firefox'] },
          },
          {
            name: 'webkit',
            use: { ...devices['Desktop Safari'] },
          },
        ]
      : []),
  ],

  webServer: [
    {
      command: 'npm run dev -- --port 3000',
      url: 'http://127.0.0.1:3000',
      timeout: 120_000,
      reuseExistingServer: !process.env.CI,
      env: {
        NEXT_PUBLIC_API_URL: 'http://127.0.0.1:8000',
      },
    },
    {
      command: 'python -m uvicorn main:app --host 127.0.0.1 --port 8000',
      cwd: '../backend',
      env: {
        PYTHONPATH: '..',
      },
      url: 'http://127.0.0.1:8000/health',
      timeout: 120_000,
      reuseExistingServer: !process.env.CI,
    },
  ],
});
