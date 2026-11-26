// Playwright smoke test (README 14.1). Uses the system Chrome (no browser
// download) with a fake camera fed by a synthetic face clip, against an
// already running dev server (default http://localhost:5173, override with
// FV_BASE_URL). It does not start or stop servers.
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: 'e2e',
  timeout: 90_000,
  reporter: 'list',
  use: {
    baseURL: process.env.FV_BASE_URL ?? 'http://localhost:5173',
    channel: 'chrome',
    headless: true,
    permissions: ['camera'],
  },
});
