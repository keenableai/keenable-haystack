"""Shared transport for the Keenable Haystack components.

One place for the parts of the Keenable contract both components need: keyed vs.
keyless endpoint selection, the attribution headers, HTTPS-only base-URL
resolution, the client-side SSRF guard, and turning a non-2xx response into a
readable error. The endpoint is read from the environment and is never a
component argument the model/pipeline can set (an arbitrary base URL is an SSRF
foothold).

The fetcher component imports from this module too, so the transport lives in
exactly one place across both `haystack_integrations.components.websearch.keenable`
and `...fetchers.keenable`.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from importlib import metadata
from typing import Any
from urllib.parse import urlsplit

import requests

try:
    _VERSION = metadata.version("keenable-haystack")
except metadata.PackageNotFoundError:  # pragma: no cover - editable/source checkout
    _VERSION = "unknown"

# Tagged User-Agent so Keenable can attribute traffic from this integration.
_USER_AGENT = f"keenable-haystack/{_VERSION}"

# The load-bearing attribution signal: the Keenable backend segments traffic by
# this header (adoption dashboards). The User-Agent above is a secondary tag.
_ATTRIBUTION_TITLE = "Haystack"

_DEFAULT_BASE_URL = "https://api.keenable.ai"
_BASE_URL_ENV = "KEENABLE_API_URL"


class KeenableError(RuntimeError):
    """A Keenable transport/API error carrying a message safe to show a user."""


def normalize_key(raw: str | None) -> str | None:
    """Return the non-blank key, else ``None`` to use the keyless free tier.

    Haystack's :class:`~haystack.utils.Secret` already resolves the
    ``KEENABLE_API_KEY`` env var (with ``strict=False`` it yields ``None`` when
    unset); this just collapses a blank/whitespace value to ``None`` so an empty
    string never selects the authenticated endpoint.
    """
    key = raw.strip() if isinstance(raw, str) else ""
    return key or None


def resolve_base_url() -> str:
    """Resolve the API base URL from ``KEENABLE_API_URL`` and enforce HTTPS."""
    base = (os.environ.get(_BASE_URL_ENV) or _DEFAULT_BASE_URL).rstrip("/")
    parsed = urlsplit(base)
    # A usable absolute URL needs a host; bail out clearly on e.g. "https://"
    # rather than letting a malformed base produce a broken request URL later.
    if parsed.hostname:
        if parsed.scheme == "https":
            return base
        # Permit plain http only for local development against a loopback host.
        if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
            return base
    msg = f"{_BASE_URL_ENV} must be an https:// URL with a host, got {base!r}"
    raise KeenableError(msg)


def _candidate_ips(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Every IP address ``host`` could denote, without doing DNS.

    Covers dotted/colon literals *and* the numeric IPv4 encodings that resolvers
    accept but :func:`ipaddress.ip_address` rejects as strings — decimal
    (``2130706433``), hex (``0x7f000001``), octal (``0177.0.0.1``) and short
    ``a.b``/``a.b.c`` forms — all of which ``socket.inet_aton`` canonicalizes to a
    real IPv4 so the private-range check below sees the true address.
    """
    candidates: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    try:
        candidates.append(ipaddress.ip_address(host))
    except ValueError:
        pass
    try:
        packed = socket.inet_aton(host)
    except OSError:
        pass
    else:
        candidates.append(ipaddress.ip_address(socket.inet_ntoa(packed)))
    return candidates


def reject_private_fetch_target(url: str) -> None:
    """Refuse obviously private/internal fetch targets before sending (SSRF).

    The backend enforces this server-side too, but a client-side guard avoids
    leaking an internal hostname in a request and is required by our integration
    contract. Hostnames that are not IP literals (and not a numeric IPv4 form)
    pass through; the backend's SSRF guard is the backstop for those.
    """
    host = (urlsplit(url).hostname or "").strip().lower()
    # A trailing dot is the FQDN form of the same name (``localhost.`` ==
    # ``localhost``); strip it so it can't slip past the checks below.
    host = host.rstrip(".")
    if not host:
        msg = f"Refusing to fetch a URL with no host: {url!r}"
        raise KeenableError(msg)
    if host in {"localhost", "metadata.google.internal"}:
        msg = f"Refusing to fetch a private/internal host: {host!r}"
        raise KeenableError(msg)
    for ip in _candidate_ips(host):
        # ``is_reserved`` is intentionally omitted: it flags non-routable but
        # harmless ranges (e.g. the 2001:db8::/32 documentation prefix). The
        # checks below are the ones that matter for SSRF.
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_unspecified
        ):
            msg = f"Refusing to fetch a private/internal address: {host!r}"
            raise KeenableError(msg)


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"User-Agent": _USER_AGENT, "X-Keenable-Title": _ATTRIBUTION_TITLE}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _raise_for_status(response: requests.Response) -> None:
    """Map a non-2xx Keenable response to a readable :class:`KeenableError`."""
    if response.ok:
        return
    detail = ""
    try:
        body = response.json()
        if isinstance(body, dict):
            detail = str(body.get("message") or body.get("error") or body.get("detail") or "")
    except ValueError:
        detail = (response.text or "").strip()
    label = {
        401: "Keenable authentication failed (401)",
        402: "Keenable: insufficient credits (402)",
        429: "Keenable rate limit exceeded (429)",
    }.get(response.status_code, f"Keenable API error ({response.status_code})")
    raise KeenableError(f"{label}: {detail}" if detail else label)


def _decode(response: requests.Response) -> dict[str, Any]:
    _raise_for_status(response)
    try:
        data = response.json()
    except ValueError as e:
        snippet = (response.text or "")[:200]
        msg = f"Keenable API returned a non-JSON response: {snippet!r}"
        raise KeenableError(msg) from e
    if not isinstance(data, dict):
        msg = f"Unexpected response from the Keenable API: {data!r}"
        raise KeenableError(msg)
    return data


def keenable_post(
    public_path: str, keyed_path: str, payload: dict[str, Any], api_key: str | None, timeout: float
) -> dict[str, Any]:
    """POST ``payload`` to the keyed or keyless endpoint and return the body."""
    path = keyed_path if api_key else public_path
    url = f"{resolve_base_url()}{path}"
    headers = {**_headers(api_key), "Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        msg = f"Could not reach the Keenable API: {e!r}"
        raise KeenableError(msg) from e
    return _decode(response)


def keenable_get(
    public_path: str, keyed_path: str, params: dict[str, Any], api_key: str | None, timeout: float
) -> dict[str, Any]:
    """GET the keyed or keyless endpoint with query ``params``; return the body."""
    path = keyed_path if api_key else public_path
    url = f"{resolve_base_url()}{path}"
    try:
        response = requests.get(url, params=params, headers=_headers(api_key), timeout=timeout)
    except requests.RequestException as e:
        msg = f"Could not reach the Keenable API: {e!r}"
        raise KeenableError(msg) from e
    return _decode(response)
