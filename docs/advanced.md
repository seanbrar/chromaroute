# Advanced Usage (Best-Effort)

This library is optimized for OpenRouter. The options below are intentional escape hatches for custom setups. They are supported on a best-effort basis and may not cover every edge case.

## Custom Base URL (OpenAI-compatible or similar)

You can point `chromaroute` at an OpenAI-compatible API by overriding the base URL. This bypasses OpenRouter routing. In this mode, `OPENROUTER_API_KEY` should be the target provider's API key.

```bash
# Example: Use OpenAI directly (best-effort)
OPENROUTER_BASE_URL=https://api.openai.com/v1
OPENROUTER_API_KEY=sk-openai-...
OPENROUTER_EMBEDDINGS_MODEL=text-embedding-3-small
```

## Manual Instantiation

If you need more control, instantiate `OpenRouterEmbeddingFunction` directly and pass `base_url`.

```python
from chromaroute import OpenRouterEmbeddingFunction

embed_fn = OpenRouterEmbeddingFunction(
    model="text-embedding-3-small",
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
)
```
