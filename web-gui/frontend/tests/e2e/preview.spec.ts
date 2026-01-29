import { test, expect } from '@playwright/test';

test.skip('preview: manual play works when autoplay is blocked', async ({ page }) => {
  // NOTE: this test is currently flaky in CI; it exercises autoplay-block behavior and
  // requires the client bundle to fetch voices. It's skipped for now but kept here as
  // a reference and to be re-enabled once the test environment is stabilized.
});
