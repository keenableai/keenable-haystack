"""Keenable page-fetch component for Haystack."""

from __future__ import annotations

import logging
from typing import Any, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

# The transport lives in the websearch leaf package so it is defined once; the
# fetcher reuses it (keyed/keyless selection, attribution headers, SSRF guard).
from haystack_integrations.components.websearch.keenable._client import (
    KeenableError,
    keenable_get,
    normalize_key,
    reject_private_fetch_target,
)

logger = logging.getLogger(__name__)


@component
class KeenableFetcher:
    """Fetches web pages via Keenable and returns their content as Documents.

    Given a list of URLs, returns one ``Document`` per successfully fetched page
    (``content`` is the page's main content as markdown; ``meta`` carries
    ``url``, ``title`` and any other fields the page exposes). Pairs with
    :class:`KeenableWebSearch` — discover URLs with search, then read full pages
    here. Unlike Haystack's ``LinkContentFetcher`` + ``HTMLToDocument`` two-step,
    Keenable returns clean extracted markdown, so this is a single component that
    returns Documents directly.

    Keyless by default (``/v1/fetch/public``); an ``api_key`` (or
    ``KEENABLE_API_KEY``) switches to ``/v1/fetch`` and lifts limits. Non-http(s)
    and private/internal URLs are rejected client-side before sending.

    ### Usage

    ```python
    from haystack_integrations.components.fetchers.keenable import KeenableFetcher

    fetcher = KeenableFetcher()
    result = fetcher.run(urls=["https://example.com/article"])
    print(result["documents"][0].content)
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("KEENABLE_API_KEY", strict=False),  # noqa: B008
        raise_on_failure: bool = False,
        timeout: float = 30.0,
    ) -> None:
        """
        :param api_key: Keenable API key. Falls back to ``KEENABLE_API_KEY``; when
            absent (or blank) the keyless public endpoint is used.
        :param raise_on_failure: If ``True``, a failed fetch raises; if ``False``
            (default, matching ``LinkContentFetcher``), the URL is logged and
            skipped so one bad URL does not fail the whole batch.
        :param timeout: Per-request timeout in seconds.
        """
        if timeout <= 0:
            msg = f"timeout must be a positive number of seconds, got {timeout!r}"
            raise ValueError(msg)
        self.api_key = api_key
        self.raise_on_failure = raise_on_failure
        self.timeout = timeout

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            raise_on_failure=self.raise_on_failure,
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KeenableFetcher":
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _fetch_one(self, url: str, api_key: Optional[str]) -> Document:
        if not url.lower().startswith(("http://", "https://")):
            msg = f"Refusing to fetch a non-http(s) URL: {url!r}"
            raise KeenableError(msg)
        reject_private_fetch_target(url)
        data = keenable_get("/v1/fetch/public", "/v1/fetch", {"url": url}, api_key, self.timeout)
        content = data.get("content") or ""
        return Document(content=content, meta=dict(data))

    @component.output_types(documents=list[Document])
    def run(self, urls: list[str]) -> dict[str, Any]:
        """Fetch each URL and return the extracted pages as Documents.

        :param urls: The URLs to fetch.
        :returns: A dict with ``documents`` (``list[Document]``), one per page
            fetched successfully.
        """
        api_key = normalize_key(self.api_key.resolve_value())
        documents: list[Document] = []
        for url in urls:
            try:
                documents.append(self._fetch_one(url, api_key))
            except Exception as e:  # noqa: BLE001 - contract: one bad URL must not fail the batch
                if self.raise_on_failure:
                    raise
                logger.warning("Keenable could not fetch %r; skipping (%s).", url, e)
        return {"documents": documents}
