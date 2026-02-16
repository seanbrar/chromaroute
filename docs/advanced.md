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

## Raw Embeddings from the Local Provider

When using the `local` provider, `build_embedding_function()` returns vectors containing NumPy `float32` values. ChromaDB handles these natively, so normal `VectorStore` usage is unaffected. If you call the embedding function directly and need to JSON-serialize the output, cast to Python floats first:

```python
raw = embed_fn(["some text"])[0]
json_safe = [float(x) for x in raw]
```

## Accessing the Full ChromaDB API

`VectorStore` exposes common operations (`add_documents`, `query`, `query_one`, `get`, `count`, `delete_collection`). For advanced ChromaDB features — metadata filters, `update()`, `upsert()`, `where`/`where_document` queries — use the underlying collection directly:

```python
store = VectorStore("my_docs")

# The ChromaDB Collection object is always accessible
store.collection.update(ids=["doc1"], documents=["updated text"])
store.collection.get(where={"category": "science"})
store.collection.upsert(ids=["doc4"], documents=["new or updated"])
```
