"""Keenable web-search component for Haystack."""

from __future__ import annotations

from typing import Any, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.components.websearch.keenable._client import (
    KeenableError,
    keenable_post,
    normalize_key,
)


@component
class KeenableWebSearch:
    """Searches the web with Keenable, a search engine built for AI agents.

    Mirrors the output shape of Haystack's built-in web-search components
    (``SerperDevWebSearch`` / ``SearchApiWebSearch``): ``run()`` returns
    ``documents`` (one ``Document`` per result, snippet as content, result
    fields as ``meta``) and ``links`` (the result URLs), so it is drop-in for
    pipelines wired to those.

    Keyless by default: with no API key the keyless public endpoint
    (``/v1/search/public``) is used. Provide an API key (the ``api_key`` argument
    or the ``KEENABLE_API_KEY`` environment variable) to use the authenticated
    endpoint (``/v1/search``), required for ``mode="realtime"`` and for higher
    rate limits.

    The API endpoint is read from ``KEENABLE_API_URL`` (HTTPS enforced), never a
    ``run`` argument, so the search cannot be redirected to an arbitrary host.

    ### Usage

    ```python
    from haystack_integrations.components.websearch.keenable import KeenableWebSearch

    # No key -> keyless public endpoint. Set KEENABLE_API_KEY to lift limits.
    websearch = KeenableWebSearch(top_k=5)
    result = websearch.run(query="latest developments in AI agents")
    print(result["documents"])
    print(result["links"])
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("KEENABLE_API_KEY", strict=False),  # noqa: B008
        top_k: Optional[int] = None,
        mode: str = "pro",
        site: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        """
        :param api_key: Keenable API key. Falls back to ``KEENABLE_API_KEY``; when
            absent (or blank) the keyless public endpoint is used.
        :param top_k: Keep at most this many results (applied client-side; the API
            returns a fixed-size set with no count parameter). ``None`` keeps all.
        :param mode: Default search mode, ``"pro"`` (deeper) or ``"realtime"``
            (low latency). ``"realtime"`` requires an API key. Overridable per run.
        :param site: Default single-domain restriction, e.g. ``"github.com"``.
            Overridable per run.
        :param timeout: Per-request timeout in seconds.
        """
        if timeout <= 0:
            msg = f"timeout must be a positive number of seconds, got {timeout!r}"
            raise ValueError(msg)
        if top_k is not None and top_k < 1:
            msg = f"top_k must be None or a positive integer, got {top_k!r}"
            raise ValueError(msg)
        self.api_key = api_key
        self.top_k = top_k
        self.mode = mode
        self.site = site
        self.timeout = timeout

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            top_k=self.top_k,
            mode=self.mode,
            site=self.site,
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KeenableWebSearch":
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document], links=list[str])
    def run(
        self,
        query: str,
        site: Optional[str] = None,
        published_after: Optional[str] = None,
        published_before: Optional[str] = None,
        acquired_after: Optional[str] = None,
        acquired_before: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run a Keenable web search.

        :param query: The search query.
        :param site: Restrict results to a single domain (overrides the default).
        :param published_after: Only pages published on/after this date (YYYY-MM-DD).
        :param published_before: Only pages published on/before this date (YYYY-MM-DD).
        :param acquired_after: Only pages indexed on/after this date (YYYY-MM-DD).
        :param acquired_before: Only pages indexed on/before this date (YYYY-MM-DD).
        :param mode: Override the default search mode for this query.
        :returns: A dict with ``documents`` (``list[Document]``) and ``links``
            (``list[str]``).
        """
        payload: dict[str, Any] = {"query": query, "mode": mode or self.mode}
        for field, value in (
            ("site", site or self.site),
            ("published_after", published_after),
            ("published_before", published_before),
            ("acquired_after", acquired_after),
            ("acquired_before", acquired_before),
        ):
            if value:
                payload[field] = value

        data = keenable_post(
            "/v1/search/public",
            "/v1/search",
            payload,
            normalize_key(self.api_key.resolve_value()),
            self.timeout,
        )
        results = data.get("results")
        if not isinstance(results, list):
            msg = f"Unexpected response from the Keenable search API: {data!r}"
            raise KeenableError(msg)

        if self.top_k is not None:
            results = results[: self.top_k]

        documents: list[Document] = []
        links: list[str] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            content = result.get("description") or result.get("title") or ""
            documents.append(Document(content=content, meta=dict(result)))
            link = result.get("url")
            if link:
                links.append(link)

        return {"documents": documents, "links": links}
