# keenable-haystack

[Keenable](https://keenable.ai) web search + page fetch for
[Haystack](https://haystack.deepset.ai) 2.x, as two components:

- **`KeenableWebSearch`** — searches the web and returns `documents` +
  `links`, the same output shape as Haystack's built-in `SerperDevWebSearch` /
  `SearchApiWebSearch`, so it is drop-in for pipelines wired to those.
- **`KeenableFetcher`** — fetches a list of URLs and returns `documents` whose
  content is the page's main text as markdown (Keenable extracts it server-side,
  so you don't need a separate `LinkContentFetcher` + `HTMLToDocument` step).

**Keyless by default**: with no API key the keyless public endpoints are used.
Provide a key to use the authenticated endpoints (for higher rate limits).

## Install

```bash
pip install keenable-haystack
```

## Usage

```python
from haystack_integrations.components.websearch.keenable import KeenableWebSearch
from haystack_integrations.components.fetchers.keenable import KeenableFetcher

# No key -> keyless public endpoints. Set KEENABLE_API_KEY to lift limits.
websearch = KeenableWebSearch(top_k=5)
hits = websearch.run(query="latest developments in AI agents")
print(hits["links"])

fetcher = KeenableFetcher()
pages = fetcher.run(urls=hits["links"][:2])
print(pages["documents"][0].content)
```

In a pipeline (drop-in for any web-search component):

```python
from haystack import Pipeline
from haystack.components.builders import PromptBuilder

pipe = Pipeline()
pipe.add_component("search", KeenableWebSearch(top_k=5))
pipe.add_component("prompt", PromptBuilder(template="Answer using:\n{{ documents }}"))
pipe.connect("search.documents", "prompt.documents")
```

`KeenableWebSearch.run` accepts optional per-query filters (`site`,
`published_after/before`, `acquired_after/before`, `mode`). There is no
`max_results`: the API returns a fixed-size result set; `top_k` (constructor)
trims it client-side.

## Configuration

- **API key (optional).** `api_key=Secret.from_token(...)` / the default
  `Secret.from_env_var("KEENABLE_API_KEY", strict=False)`. Blank/unset → keyless
  public endpoints. Serializes by env-var name, never the key value.
- **Endpoint (optional).** `KEENABLE_API_URL` overrides the base URL (HTTPS
  enforced; plain `http` only for loopback). The endpoint is never a component
  argument the model can set, so it cannot be used to redirect requests.

`KeenableFetcher` rejects non-`http(s)` schemes and private/internal hosts
client-side before sending, and (like `LinkContentFetcher`) skips failed URLs by
default — set `raise_on_failure=True` to surface errors instead.

## License

MIT © Keenable
