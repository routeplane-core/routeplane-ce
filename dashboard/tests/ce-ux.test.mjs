import assert from "node:assert/strict";
import { test } from "node:test";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { build } from "esbuild";

// Render the actual header JSX in isolation: theme/session/query side effects
// are not needed to verify the fixed 64px collapsed-rail geometry contract.
const shellSource = readFileSync(new URL('../src/components/layout/AppShell.tsx', import.meta.url), 'utf8');
const headerStart = shellSource.indexOf('<div', shellSource.indexOf('<aside'));
const headerEnd = shellSource.indexOf('</div>', headerStart) + '</div>'.length;
assert(headerStart >= 0 && headerEnd > headerStart);
const headerJsx = shellSource.slice(headerStart, headerEnd);

// Bundle the actual application source and its SSR renderer together. No browser,
// gateway, provider or Rust process is started by this focused unit gate.
const bundle = await build({
  stdin: { contents: `
    export { streamChat, api } from './src/lib/api/client';
    export { validChatTarget } from './src/lib/api/chat-target';
    import React from 'react';
    import { renderToStaticMarkup } from 'react-dom/server';
    import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
    import { MemoryRouter } from 'react-router-dom';
    import { Playground } from './src/routes/Playground';
    import { Overview } from './src/routes/Overview';
    import { Health } from './src/routes/Health';
    import { Cache } from './src/routes/Cache';
    import { Providers } from './src/routes/Providers';
    import { ToastProvider } from './src/components/ui/toast';
    import { ResponseRequestId } from './src/components/ui/request-id';
    import { Logs } from './src/routes/Logs';
    import { DataTable } from './src/components/ui/table';
    import { BrandLockup, LogoMark } from './src/components/ui/logo';
    import { PanelLeft, PanelLeftClose } from 'lucide-react';
    import { cn } from './src/lib/utils';
    export function renderSidebarHeader(collapsed) {
      const setCollapsed = () => {};
      return renderToStaticMarkup(${headerJsx});
    }
    export function renderRequestId(requestId) { return renderToStaticMarkup(React.createElement(ResponseRequestId, {requestId})); }
    export function renderPage(page, data = {}) {
      const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
      client.setQueryData(['models'], data.models ?? {data: []});
      // Match the current public wire contract; undefined explicitly exercises
      // the pending/no-status case, while legacy snapshots remain opt-in.
      client.setQueryData(['status'], Object.hasOwn(data, 'status') ? data.status : {status: 'ok'});
      client.setQueryData(['metrics'], data.metrics ?? []);
      client.setQueryData(['analytics'], []);
      client.setQueryData(['providers'], []);
      client.setQueryData(['logs'], data.logs ?? []);
      const pages = {playground: Playground, logs: Logs, overview: Overview, health: Health, cache: Cache, providers: Providers};
      return renderToStaticMarkup(React.createElement(QueryClientProvider, {client},
        React.createElement(MemoryRouter, null, React.createElement(ToastProvider, null, React.createElement(pages[page])))));
    }
    export function renderTable() { return renderToStaticMarkup(React.createElement(DataTable, {
      columns: [{key:'id', header:'ID', cell:r=>r.id, sortValue:r=>r.id}],
      rows: [{id:'log_kept'}], getRowId:r=>r.id, onRowClick:()=>{}
    })); }
  `, resolveDir: new URL("../", import.meta.url).pathname, loader: "tsx" },
  bundle: true, platform: "node", format: "cjs", write: false,
  define: { "import.meta.env.VITE_API_BASE": '""', "process.env.NODE_ENV": '"production"' },
  tsconfig: new URL("../tsconfig.app.json", import.meta.url).pathname,
});
const mod = { exports: {} };
new Function("require", "module", "exports", bundle.outputFiles[0].text)(createRequire(import.meta.url), mod, mod.exports);
const { streamChat, api, validChatTarget, renderPage, renderTable, renderSidebarHeader, renderRequestId } = mod.exports;
const originalFetch = globalThis.fetch;
const session = new Map([['rp_ce_session', 'synthetic-unit-session']]);
globalThis.localStorage = { getItem:k=>session.get(k) ?? null, removeItem:k=>session.delete(k) };

test("collapsed 64px header contains only its centered 32px labelled click target", () => {
  const html = renderSidebarHeader(true);
  assert.equal((html.match(/<button\b/g) ?? []).length, 1);
  assert.equal((html.match(/<svg\b/g) ?? []).length, 1, 'no competing brand mark inside collapsed header');
  assert.match(html, /justify-center px-2/);
  assert.match(html, /h-8 w-8 shrink-0/);
  assert.match(html, /aria-label="Expand sidebar"/);
  // Tailwind's default 4px spacing: 8px side padding leaves 48px; centered 32px
  // button occupies x16..48. This is a sizing contract, not browser UAT proof.
  const rail = 64, padding = 2 * 4, target = 8 * 4;
  const x = padding + (rail - 2 * padding - target) / 2;
  assert(x >= 0 && x + target <= rail);
  const expanded = renderSidebarHeader(false);
  assert.match(expanded, /CE Console/);
  assert.match(expanded, /aria-label="Collapse sidebar"/);
  assert.match(expanded, /gap-2 px-4/);
});

test("explicit target permits self-hosted model but not URLs, chains or blank input", () => {
  assert.equal(validChatTarget("qwen2.5:0.5b", "self_hosted"), true);
  for (const [model, provider] of [["", "self_hosted"], ["  ", "self_hosted"], ["m", ""], ["m", "http://host"], ["m", "openai,anthropic"], ["m", "foo\r\nx-header:bar"]]) {
    assert.equal(validChatTarget(model, provider), false);
  }
});

test("empty catalogue still renders labelled model and provider inputs, initially disabled Run", () => {
  const html = renderPage("playground");
  assert.match(html, /No catalogue entries/);
  assert.match(html, /<input[^>]*id="chat-model"/);
  assert.match(html, /<input[^>]*id="chat-provider"/);
  assert.match(html, /<button[^>]*disabled=""[^>]*>.*?Run<\/button>/);
  assert.match(html, /Catalogue entries are suggestions/);
  assert.doesNotMatch(html, /No models available/);
});

test("public liveness JSON is preserved without inventing diagnostics", async () => {
  globalThis.fetch = async () => new Response('{"status":"ok"}', {headers:{"content-type":"application/json"}});
  try {
    const status = await api.getStatus();
    assert.deepEqual(status, {status:"ok"});
    assert.match(renderPage("playground", {status}), /id="chat-provider"/);
  } finally { globalThis.fetch = originalFetch; }
});

test("manual targets and actual model suggestions do not require status diagnostics", () => {
  for (const status of [{status:"ok"}, undefined, {providers:[]}]) {
    const html = renderPage("playground", {status, models:{data:[{id:"actual-model",owned_by:"catalogue-owner"}]}});
    assert.match(html, /<option value="actual-model"/);
    assert.match(html, /<input[^>]*id="chat-provider"[^>]*value=""/);
    assert.match(html, /<button[^>]*disabled=""[^>]*>.*?Run<\/button>/);
    assert.doesNotMatch(html, /<option value="catalogue-owner"/);
  }
  assert.equal(validChatTarget("actual-model", "self_hosted"), true);
});

test("liveness-only overview and cache explicitly withhold unavailable diagnostics", () => {
  const overview = renderPage("overview");
  assert.match(overview, /Provider inventory unavailable/);
  assert.match(overview, /Provider diagnostics unavailable/);
  assert.match(overview, /Cache diagnostics unavailable/);
  assert.doesNotMatch(overview, /0 providers|Healthy circuits|Open circuits/);
  const cache = renderPage("cache");
  assert.match(cache, /Cache diagnostics unavailable/);
  assert.match(cache, /Missing statistics do not mean the cache is empty or disabled/);
  assert.match(cache, /<button[^>]*disabled=""[^>]*>.*?Purge cache<\/button>/);
  assert.doesNotMatch(cache, /cached responses|0 hits/);
});

test("traffic and catalogue rows never invent healthy circuits without diagnostics", () => {
  const health = renderPage("health", {metrics:[{provider:"observed-provider",outcome:"success",value:3}]});
  assert.match(health, /observed-provider/);
  assert.match(health, /Unavailable/);
  assert.doesNotMatch(health, />closed</);
  const providers = renderPage("providers", {models:{data:[{id:"m",owned_by:"catalogue-owner"}]}});
  assert.match(providers, /catalogue-owner/);
  assert.match(providers, /Unavailable/);
  assert.doesNotMatch(providers, />closed<|No providers configured/);
  assert.match(renderPage("health"), /Provider diagnostics unavailable/);
});

test("legacy snapshots still render actual suggestions, circuit states and cache counts", () => {
  const status = {
    providers:[{provider:"legacy-provider",circuit:"open",latency_ewma_ms:12}],
    cache:{approx_bytes:1024,entries:7,hit_rate:0.5,hits:9,misses:9,oversize_drops:0,write_drops:0},
  };
  assert.match(renderPage("playground", {status}), /<option value="legacy-provider"/);
  assert.match(renderPage("health", {status}), />open</);
  assert.match(renderPage("providers", {status}), />open</);
  assert.match(renderPage("overview", {status}), /7 entries/);
  assert.match(renderPage("cache", {status}), /9 hits/);
  const empty = renderPage("overview", {status:{providers:[]}});
  assert.match(empty, /0 providers in status snapshot/);
  assert.doesNotMatch(empty, /Provider inventory unavailable/);
});

test("table rows and sort headers are keyboard focusable without replacing row identity", () => {
  const html = renderTable();
  assert.match(html, /<tr[^>]*tabindex="0"[^>]*aria-label="Inspect log_kept"/);
  assert.match(html, /<th[^>]*tabindex="0"/);
  assert.match(html, /role="region" aria-label="Scrollable table"/);
});

test("old log rows without request_id render honestly and retain bounded-history description", () => {
  const html = renderPage("logs", { logs:[{id:"log_kept",timestamp:new Date().toISOString(),provider:"self_hosted",model:"m",outcome:"success",virtual_key_name:"test",prompt_tokens:1,completion_tokens:1,total_tokens:2}] });
  assert.match(html, /Inspect log_kept/);
  assert.match(html, /Restarted or evicted history is not searchable/);
  assert.match(html, /aria-label="Search retained logs"/);
});

function response(parts, contentType = "text/event-stream") {
  return new Response(new ReadableStream({start(controller) {
    for (const part of parts) controller.enqueue(new TextEncoder().encode(part));
    controller.close();
  }}), {headers:{"content-type":contentType}});
}

test("stream sends exact explicit provider/model with session, and parses split SSE bytes", async () => {
  let request;
  globalThis.fetch = async (url, init) => {
    request = {url,init};
    return response(['data: {"choices":[{"delta":{"cont', 'ent":"hello"}}]}\r\n\r\ndata: [DONE]\r\n\r\n']);
  };
  try {
    let output = "";
    await streamChat({model:"qwen2.5:0.5b",messages:[{role:"user",content:"hello"}]}, d=>output+=d, undefined, "self_hosted");
    assert.equal(output, "hello");
    assert.equal(request.url, "/v1/chat/completions");
    assert.equal(request.init.headers["x-routeplane-provider"], "self_hosted");
    assert.equal(request.init.headers.Authorization, "Bearer synthetic-unit-session");
    assert.equal(JSON.parse(request.init.body).stream, true);
    assert.equal(JSON.parse(request.init.body).model, "qwen2.5:0.5b");
  } finally { globalThis.fetch = originalFetch; }
});

test("invalid/missing explicit target rejects before dispatch", async () => {
  let calls = 0;
  globalThis.fetch = async () => { calls++; throw new Error("unexpected dispatch"); };
  try {
    for (const provider of ["", "http://host", "openai,anthropic"]) {
      await assert.rejects(streamChat({model:"m"}, ()=>{}, undefined, provider), /Enter a model/);
    }
    assert.equal(calls, 0);
  } finally { globalThis.fetch = originalFetch; }
});

test("response identity uses actual gateway aliases, never completion or W3C IDs", async () => {
  try {
    for (const [headers, expected] of [
      [{"x-routeplane-request-id":"req_primary","x-routeplane-trace-id":"req_alias"}, "req_primary"],
      [{"x-routeplane-trace-id":"req_legacy"}, "req_legacy"],
      [{"traceparent":"00-not-a-gateway-request-id"}, null],
    ]) {
      globalThis.fetch = async () => new Response('data: {"id":"provider_completion_id","choices":[]}\n\ndata: [DONE]\n\n', {headers:{"content-type":"text/event-stream",...headers}});
      const observed=[];
      assert.equal(await streamChat({model:"m"},()=>{},undefined,"self_hosted",id=>observed.push(id)),expected);
      assert.deepEqual(observed,[expected]);
    }
  } finally { globalThis.fetch = originalFetch; }
});

test("response identity arrives before deltas and survives HTTP or stream failure", async () => {
  try {
    for (const [body,status,expectedError] of [
      ['upstream unavailable',500,/upstream unavailable/],
      ['data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',200,/before \[DONE\]/],
    ]) {
      const events=[];
      globalThis.fetch = async () => new Response(body,{status,headers:{"content-type":"text/event-stream","x-routeplane-request-id":"req_failure"}});
      await assert.rejects(streamChat({model:"m"},text=>events.push(text),undefined,"self_hosted",id=>events.push(id)),expectedError);
      assert.equal(events[0],"req_failure");
      if(status===200)assert.equal(events[1],"partial");
    }
  } finally { globalThis.fetch = originalFetch; }
});

test("Playground exposes copyable escaped identity, hides absence and resets each run", () => {
  assert.equal(renderRequestId(null),"");
  const html=renderRequestId('req_<unsafe>');
  assert.match(html,/Gateway request ID/);
  assert.match(html,/Copy request ID/);
  assert.match(html,/req_&lt;unsafe&gt;/);
  assert.match(html,/Only retained rows/);
  assert.match(html,/break-all/);
  const source=readFileSync(new URL('../src/routes/Playground.tsx',import.meta.url),'utf8');
  const run=source.slice(source.indexOf('const run ='),source.indexOf('const stop ='));
  assert(run.indexOf('setRequestId(null)')>=0 && run.indexOf('setRequestId(null)')<run.indexOf('await streamChat'));
  assert.match(run,/provider\.trim\(\), setRequestId\)/);
});

test("upstream errors remain failures, no hidden second request", async () => {
  let calls = 0;
  globalThis.fetch = async () => { calls++; return new Response('{"error":{"message":"provider credential missing"}}', {status:500}); };
  try {
    await assert.rejects(streamChat({model:"m"}, ()=>{}, undefined, "self_hosted"), /provider credential missing/);
    assert.equal(calls, 1);
  } finally { globalThis.fetch = originalFetch; }
});

test("truncated, malformed, errored and buffered streams cannot masquerade as success", async () => {
  try {
    for (const [parts, type, expected] of [
      [['data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'], 'text/event-stream', /before \[DONE\]/],
      [['data: bad-json\n\n'], 'text/event-stream', /invalid streaming event/],
      [['data: {"error":{"message":"unavailable"}}\n\n'], 'text/event-stream', /streaming error/],
      [['{"choices":[]}'], 'application/json', /non-streaming response/],
    ]) {
      globalThis.fetch = async () => response(parts, type);
      await assert.rejects(streamChat({model:"m"}, ()=>{}, undefined, "self_hosted"), expected);
    }
  } finally { globalThis.fetch = originalFetch; }
});

test("aborted fetch remains an abort and does not retry", async () => {
  let calls = 0;
  globalThis.fetch = async (_url, init) => { calls++; assert.equal(init.signal.aborted, true); throw new DOMException("Stopped", "AbortError"); };
  try {
    const controller = new AbortController(); controller.abort();
    await assert.rejects(streamChat({model:"m"}, ()=>{}, controller.signal, "self_hosted"), {name:"AbortError"});
    assert.equal(calls, 1);
  } finally { globalThis.fetch = originalFetch; }
});
