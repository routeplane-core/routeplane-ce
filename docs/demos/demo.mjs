import { RouteplaneCoreClient } from '@routeplane/sdk/core';

const rp = new RouteplaneCoreClient({
  apiKey: 'rp_local_demo_2f8a1c',
  baseUrl: 'http://localhost:8080',
});

// Stream a completion through the gateway — token by token over SSE.
let model, usage;
for await (const chunk of rp.stream('/v1/chat/completions', {
  model: 'llama-3.1-8b-instant',
  messages: [{ role: 'user', content: 'In one sentence, what is an AI gateway?' }],
  stream: true,
  stream_options: { include_usage: true },
}, { headers: { provider: 'groq' } })) {
  model = chunk.model ?? model;
  usage = chunk.usage ?? usage;
  process.stdout.write(chunk.choices?.[0]?.delta?.content ?? '');
  await new Promise((r) => setTimeout(r, 25)); // slow the render so the stream is legible on screen
}

console.log(`\n\n>> served by groq · ${model} · ${usage.total_tokens} tokens`);
