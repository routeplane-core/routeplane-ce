from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="rp_your_key",
    default_headers={"x-routeplane-provider": "self_hosted"},  # my local Ollama
)
reply = client.chat.completions.create(
    model="qwen2.5:0.5b",
    messages=[{"role": "user", "content": "Say hello from a local model."}],
)
print(reply.choices[0].message.content)
