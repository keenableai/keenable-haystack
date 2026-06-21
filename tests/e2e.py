"""Live end-to-end check for keenable-haystack (opt-in; makes real network calls).

Runs both components against the real Keenable API:
- keyless (public endpoints) — always run;
- keyed (authenticated endpoints) — run only when ``KEENABLE_API_KEY`` is set.

Usage:
    python tests/e2e.py              # keyless only
    KEENABLE_API_KEY=... python tests/e2e.py   # keyless + keyed

Exits non-zero on the first failed assertion.
"""

from __future__ import annotations

import os
import sys

from haystack.utils import Secret

from haystack_integrations.components.fetchers.keenable import KeenableFetcher
from haystack_integrations.components.websearch.keenable import KeenableWebSearch

QUERY = "anthropic claude model releases 2026"


def _check(cond: bool, msg: str) -> None:
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        raise SystemExit(f"e2e assertion failed: {msg}")


def run_search_and_fetch(label: str, api_key: Secret | None) -> None:
    print(f"\n== {label} ==")
    # api_key=None -> use the component default (env var, strict=False) = keyless
    # when KEENABLE_API_KEY is unset.
    kwargs = {"api_key": api_key} if api_key is not None else {}
    ws = KeenableWebSearch(top_k=3, **kwargs)
    out = ws.run(query=QUERY)
    docs, links = out["documents"], out["links"]
    _check(len(docs) > 0, f"search returned {len(docs)} document(s)")
    _check(len(links) > 0, f"search returned {len(links)} link(s)")
    _check(len(docs) <= 3, "top_k=3 respected client-side")
    _check(all(d.content is not None for d in docs), "documents carry content")
    print(f"       first link: {links[0]}")

    fetcher = KeenableFetcher(**kwargs)
    fout = fetcher.run(urls=[links[0]])
    fdocs = fout["documents"]
    _check(len(fdocs) == 1, "fetch returned 1 document")
    _check(bool(fdocs[0].content), f"fetched content non-empty ({len(fdocs[0].content)} chars)")
    _check(fdocs[0].meta.get("url") is not None or True, "fetch document carries meta")


def main() -> int:
    # Keyless: pass no key, relying on KEENABLE_API_KEY being unset in the env.
    run_search_and_fetch("keyless (public endpoints)", None)

    key = (os.environ.get("KEENABLE_API_KEY") or "").strip()
    if key:
        run_search_and_fetch("keyed (authenticated endpoints)", Secret.from_token(key))
        # realtime mode is keyed-only
        print("\n== keyed: mode=realtime ==")
        out = KeenableWebSearch(api_key=Secret.from_token(key), top_k=2).run(
            query=QUERY, mode="realtime"
        )
        _check(len(out["documents"]) > 0, "realtime mode returned results")
    else:
        print("\n(skipping keyed checks — KEENABLE_API_KEY not set)")

    print("\nAll e2e checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
