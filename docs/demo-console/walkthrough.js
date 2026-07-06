// CE Console demo walkthrough — records a video of the real console served by
// the real CE gateway, backed by a local Ollama model. No cloud keys, no mocks.
const { chromium } = require("playwright");

const BASE = process.env.DEMO_BASE || "http://localhost:8080";
const OLLAMA = process.env.DEMO_OLLAMA || "http://localhost:11434";
const MODEL = process.env.DEMO_MODEL || "qwen2.5:0.5b";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// A visible cursor: Playwright videos don't render the OS pointer, so inject a
// dot that follows real mousemove events.
const CURSOR_JS = `
  (() => {
    const d = document.createElement('div');
    d.id = '__pw_cursor';
    Object.assign(d.style, {
      position: 'fixed', top: '0', left: '0', width: '18px', height: '18px',
      borderRadius: '50%', background: 'rgba(56,132,255,.45)',
      border: '2px solid rgba(56,132,255,.95)', zIndex: '2147483647',
      pointerEvents: 'none', transform: 'translate(-50%,-50%)',
      transition: 'width .12s, height .12s',
    });
    const attach = () => document.body && document.body.appendChild(d);
    document.body ? attach() : addEventListener('DOMContentLoaded', attach);
    addEventListener('mousemove', (e) => {
      d.style.left = e.clientX + 'px';
      d.style.top = e.clientY + 'px';
    }, true);
    addEventListener('mousedown', () => { d.style.width = '13px'; d.style.height = '13px'; }, true);
    addEventListener('mouseup', () => { d.style.width = '18px'; d.style.height = '18px'; }, true);
  })();
`;

async function glideTo(page, locator) {
  const box = await locator.boundingBox();
  if (!box) throw new Error("no bounding box for locator");
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 28 });
  await sleep(280);
}

async function click(page, locator) {
  await glideTo(page, locator);
  await locator.click();
}

async function type(page, locator, text) {
  await click(page, locator);
  await locator.pressSequentially(text, { delay: 55 });
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    recordVideo: { dir: "video", size: { width: 1280, height: 800 } },
  });
  await context.addInitScript(CURSOR_JS);
  const page = await context.newPage();
  page.setDefaultTimeout(20000);

  // ── Beat 1: the login screen ──────────────────────────────────────────────
  await page.goto(BASE + "/");
  await page.getByRole("button", { name: "Sign in", exact: true }).waitFor();
  await page.mouse.move(640, 300, { steps: 5 });
  await sleep(2200);

  // ── Beat 2: create the operator account ──────────────────────────────────
  await click(page, page.getByRole("button", { name: "Create one" }));
  await sleep(600);
  await type(page, page.getByPlaceholder("you@company.com"), "operator@example.com");
  await type(page, page.getByPlaceholder("At least 10 characters"), "a-strong-passphrase");
  await type(page, page.getByPlaceholder("••••••••"), "a-strong-passphrase");
  await sleep(500);
  await click(page, page.getByRole("button", { name: "Create account" }));

  // The app shell renders on success (Overview).
  await page.getByRole("link", { name: "Provider Integrations" }).waitFor();
  await sleep(2800);

  // ── Beat 3: add local Ollama as a custom provider — no restart ───────────
  await click(page, page.getByRole("link", { name: "Provider Integrations" }));
  await page.getByText("No custom providers yet").waitFor();
  await sleep(1600);
  await click(page, page.getByRole("button", { name: "Add provider" }).first());
  await page.locator("#p-name").waitFor();
  await sleep(500);
  await type(page, page.locator("#p-name"), "ollama-local");
  await type(page, page.locator("#p-url"), OLLAMA);
  await type(page, page.locator("#p-key"), "ollama-no-auth");
  await type(page, page.locator("#p-models"), MODEL);
  await sleep(700);
  await click(page, page.locator("form").getByRole("button", { name: "Add provider" }));
  // Toast + table row confirm it's live immediately.
  await page.getByText("usable now").waitFor();
  await sleep(2600);

  // ── Beat 4: playground — stream a completion through the gateway ─────────
  await click(page, page.getByRole("link", { name: "Playground" }));
  await page.getByRole("button", { name: "Run" }).waitFor();
  await sleep(1200);
  // Pick the model the custom provider just added.
  await click(page, page.getByRole("combobox").first());
  await sleep(700);
  await click(page, page.getByRole("option", { name: MODEL }));
  await sleep(700);
  const userBox = page.locator("textarea").nth(1);
  await click(page, userBox);
  await userBox.fill("");
  await userBox.pressSequentially("Write a haiku about self-hosting your AI.", { delay: 45 });
  await sleep(600);
  await click(page, page.getByRole("button", { name: "Run" }));
  // Streaming: Run flips to Stop, then back when the stream ends.
  await page.getByRole("button", { name: "Stop" }).waitFor();
  await page.getByRole("button", { name: "Run" }).waitFor({ timeout: 120000 });
  // Show the full response card (the run scrolled the page down).
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));
  await sleep(2800);

  // ── Beat 5: reveal the gateway key ────────────────────────────────────────
  // The SPA preserves scroll across routes — reset so each beat starts at the top.
  await click(page, page.getByRole("link", { name: "API Keys" }));
  await page.getByRole("button", { name: "Reveal key" }).waitFor();
  await page.evaluate(() => window.scrollTo({ top: 0 }));
  await sleep(1400);
  await click(page, page.getByRole("button", { name: "Reveal key" }));
  await sleep(2400);

  // ── Beat 6: the traffic shows up in Usage & Analytics ────────────────────
  await click(page, page.getByRole("link", { name: "Usage & Analytics" }));
  await page.getByText("Requests over time").waitFor();
  await page.evaluate(() => window.scrollTo({ top: 0 }));
  await sleep(3200);

  await context.close(); // flushes the video
  await browser.close();
  console.log("done");
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
