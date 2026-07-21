from routeplane import Routeplane

rp = Routeplane(
    api_key="rp_local_demo_2f8a1c",
    base_url="http://localhost:8080/v1",
)

completion, meta = rp.create_with_meta(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "What is a sovereign AI gateway? One sentence."}],
    extra_headers={"x-routeplane-provider": "groq"},
)

print(completion.choices[0].message.content)
print(f"→ served by {meta.provider or 'groq'} · {completion.usage.total_tokens} tokens")
