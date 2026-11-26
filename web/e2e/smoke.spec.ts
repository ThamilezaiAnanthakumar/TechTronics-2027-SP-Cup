// Smoke test: the app loads, runs the camera path on a synthetic-face fake
// camera, shows a status, makes no network request to any other host, and
// throws no page errors. Synthetic video only.
import { chromium, expect, test } from '@playwright/test';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { writeSyntheticY4m } from './clip';

test('loads, runs the fake camera, stays on-device', async ({ baseURL }) => {
  const reachable = await fetch(baseURL!).then((r) => r.ok).catch(() => false);
  test.skip(!reachable, `no server at ${baseURL} (start npm run dev, or set FV_BASE_URL)`);

  const clip = join(tmpdir(), 'facevitals_synthetic_face.y4m');
  writeSyntheticY4m(clip);
  const browser = await chromium.launch({
    channel: 'chrome',
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${clip}`],
  });
  const page = await (await browser.newContext({ permissions: ['camera'] })).newPage();
  const host = new URL(baseURL!).host;
  const foreign: string[] = [], errors: string[] = [];
  page.on('request', (r) => { const u = new URL(r.url()); if (!['data:', 'blob:'].includes(u.protocol) && u.host !== host) foreign.push(r.url()); });
  page.on('pageerror', (e) => errors.push(e.message));

  await page.goto(baseURL!);
  await expect(page.getByRole('note')).toContainText('Video never leaves this device');
  await page.getByRole('button', { name: 'Start camera' }).click();
  const badge = page.locator('.badge');
  await expect(badge).toBeVisible({ timeout: 45_000 });
  await expect(page.locator('dl.facts')).toContainText('fps', { timeout: 20_000 });
  await page.waitForTimeout(5_000);
  console.log(`status after start: ${(await badge.innerText()).replace(/\s+/g, ' ')}`);
  await page.getByRole('button', { name: 'Stop' }).click();
  await browser.close();

  expect(foreign, `requests to other hosts: ${foreign.join(', ')}`).toEqual([]);
  expect(errors).toEqual([]);
});
