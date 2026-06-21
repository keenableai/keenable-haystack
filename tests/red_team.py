"""Automatic Claude-Code red-teaming harness for ``keenable-haystack``.

This drives Claude Code in headless mode (``claude -p``) as an *adversary* against
the :class:`KeenableWebSearch` / :class:`KeenableFetcher` Haystack components
(which call the Keenable web-search / page-fetch HTTP API). Claude hunts for
security / robustness defects (API-key leakage in serialization/repr/logs,
injection / SSRF, malformed-response and transport-error handling,
endpoint-selection mistakes), reads the source, and writes+runs adversarial probe
scripts in an isolated scratch directory. It returns machine-readable findings
validated against a JSON schema.

Standalone:  ENABLE_CLAUDE_RED_TEAM=1 python tests/red_team.py
(writes a report and exits non-zero if any high/critical finding is reported).

No third-party dependencies — stdlib only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Layout: this file lives in ``<root>/tests/``; the package source is the
# ``haystack_integrations`` namespace tree at ``<root>/``.
TEST_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TEST_DIR.parent
PACKAGE_SOURCE = PACKAGE_DIR / "haystack_integrations" / "components"
WEBSEARCH_SRC = PACKAGE_SOURCE / "websearch" / "keenable" / "web_search.py"
CLIENT_SRC = PACKAGE_SOURCE / "websearch" / "keenable" / "_client.py"
FETCHER_SRC = PACKAGE_SOURCE / "fetchers" / "keenable" / "fetcher.py"
SCRATCH_DIR = TEST_DIR / ".red_team_scratch"

BLOCKING_SEVERITIES = ("high", "critical")

# Bound the cost/work of a single run. ``sonnet`` is cheap, fast, and very capable
# for this audit; the budget is a hard cap that ABORTS the run.
MAX_BUDGET_USD = "2.00"
RUN_TIMEOUT_SECONDS = 600
MODEL = "sonnet"

FINDINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "category": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                    },
                    "title": {"type": "string"},
                    "evidence": {"type": "string"},
                    "location": {"type": "string"},
                    "reproduced": {"type": "boolean"},
                    "recommendation": {"type": "string"},
                },
                "required": ["category", "severity", "title", "evidence"],
            },
        },
        "passed_checks": {"type": "array", "items": {"type": "string"}},
        "probes_run": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["findings", "summary"],
}


def claude_available() -> bool:
    try:
        result = subprocess.run(["claude", "--version"], capture_output=True, timeout=5)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def build_prompt(python_exe: str, scratch_dir: Path) -> str:
    return f"""\
You are a security red-teamer. Your targets are two Haystack 2.x components in
the package `keenable-haystack`:
- `KeenableWebSearch` (haystack_integrations.components.websearch.keenable) — source: {WEBSEARCH_SRC}
- `KeenableFetcher`  (haystack_integrations.components.fetchers.keenable)  — source: {FETCHER_SRC}
Their shared HTTP transport (endpoint selection, headers, SSRF guard, error
mapping) lives at: {CLIENT_SRC}

Both call the Keenable web-search / page-fetch HTTP API. The API key is a Haystack
`haystack.utils.Secret` (default `Secret.from_env_var("KEENABLE_API_KEY", strict=False)`).

Your job: find real security or robustness defects, and ACTUALLY TRY TO BREAK IT —
do not just read the code. Write small adversarial probe scripts and run them.

Rules of engagement:
- Write any probe scripts ONLY inside this directory: {scratch_dir}
- Run probes with EXACTLY this Python interpreter (it already has the package
  importable, offline): {python_exe}
  e.g.  {python_exe} {scratch_dir / "probe_key_leak.py"}
- Do NOT make real network calls. Monkeypatch
  `haystack_integrations.components.websearch.keenable._client.requests`
  (or `requests.post` / `requests.get`) to simulate hostile/malformed server
  responses and transport errors. You are testing the CLIENT, not the live API.
- Only report findings that you either REPRODUCED with a probe or that are directly
  evident in the source. Set "reproduced": true only when a probe demonstrates it.
- Be calibrated with severity. Reserve high/critical for genuine exploitable issues
  (e.g. the API key actually leaking to an attacker-controlled destination, or the
  key appearing in to_dict/from_dict serialization, repr, logs, or exceptions). A
  component that raises a clear exception from `run()` is NOT a vulnerability — that
  is normal Haystack behavior.

Attack surface to cover (write at least one probe per relevant area):
1. API-key confidentiality. The key is a Haystack `Secret`. Verify it NEVER appears
   in: `component.to_dict()` (must serialize the Secret by ENV VAR NAME, never the
   token value), repr(component), str(component), logging output, or any raised
   exception text. Build a component with a `Secret.from_token("super-secret")` and
   confirm to_dict raises or omits the value (token secrets are not serializable).
   Confirm from_dict round-trips by env var name only.
2. Endpoint selection & key handling. With a key -> POST <base>/v1/search (search)
   / GET <base>/v1/fetch (fetch) with an X-API-Key header. Without a key (or a
   blank/whitespace key) -> the /v1/search/public, /v1/fetch/public path and NO
   X-API-Key. Probe that the key is never sent to the public endpoint. The base URL
   comes only from the KEENABLE_API_URL env var and must be https (or http loopback);
   probe that a non-https / attacker-controlled KEENABLE_API_URL is rejected and
   cannot smuggle the X-API-Key header off-host.
3. Malformed / hostile responses. Make the mocked server return: `results` not a
   list, missing `results`, non-dict items, a body that is not valid JSON, an HTTP
   401/402/429/500 with a hostile body, and a huge payload. The component should
   raise a clear `KeenableError` (websearch) — never leak internals, never surface a
   raw JSONDecodeError, never put the API key in the message. For `KeenableFetcher`
   with `raise_on_failure=False` (default), a bad URL must be skipped (logged), not
   crash the batch; with `raise_on_failure=True` it should raise.
4. Transport errors. requests.RequestException must be turned into a handled
   KeenableError, with the API key NOT present in the text.
5. Injection / SSRF. `query`, `site`, dates are JSON-encoded into the body — confirm
   no injection. `KeenableFetcher` must refuse non-http(s) URLs (file://, etc.) and
   private/internal hosts (localhost, 127.0.0.1, 169.254.169.254, 10.x, ::1,
   metadata.google.internal) BEFORE sending. Try to bypass the guard (uppercase
   scheme, embedded credentials user@host, IPv6 brackets, decimal/hex IP encodings,
   trailing dots, unicode). Review User-Agent / header construction.
6. Parameter edge cases: negative/zero timeout, very large or unicode `query`,
   malformed date filters, mode=realtime on the keyless endpoint, top_k <= 0 or
   huge, an empty urls list to the fetcher.

When finished, return the structured JSON object required by the output schema:
- "findings": each with category, severity, title, evidence (what you observed,
  ideally the probe output), location (file:line if known), reproduced, and a
  recommendation.
- "passed_checks": short strings naming the defenses you verified hold.
- "probes_run": the probe filenames you created and executed.
- "summary": one paragraph of overall assessment.
If you find nothing exploitable, return an empty "findings" list and say so.
"""


def _parse_claude_output(stdout: str, stderr: str) -> dict[str, Any]:
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        msg = (
            "Could not parse claude output as JSON.\n"
            f"stdout (first 2000 chars):\n{stdout[:2000]}\n"
            f"stderr (first 1000 chars):\n{stderr[:1000]}"
        )
        raise RuntimeError(msg) from exc

    if isinstance(envelope, dict) and envelope.get("is_error"):
        subtype = envelope.get("subtype", "unknown")
        raise RuntimeError(
            f"claude reported an error (subtype={subtype}): {envelope.get('result') or envelope}"
        )

    candidate: Any = None
    if isinstance(envelope, dict):
        if isinstance(envelope.get("structured_output"), dict):
            candidate = envelope["structured_output"]
        elif isinstance(envelope.get("result"), dict):
            candidate = envelope["result"]
        elif isinstance(envelope.get("result"), str):
            try:
                candidate = json.loads(envelope["result"])
            except json.JSONDecodeError:
                candidate = None
        if candidate is None and "findings" in envelope:
            candidate = envelope

    if not isinstance(candidate, dict) or "findings" not in candidate:
        raise RuntimeError(
            "claude output did not contain a structured findings object. "
            f"Envelope keys: {list(envelope) if isinstance(envelope, dict) else type(envelope)}\n"
            f"stdout (first 2000 chars):\n{stdout[:2000]}"
        )
    return candidate


def run_red_team(*, model: str = MODEL, timeout: int = RUN_TIMEOUT_SECONDS) -> dict[str, Any]:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(sys.executable, SCRATCH_DIR)

    cmd = [
        "claude",
        "-p",
        prompt,
        "--add-dir",
        str(PACKAGE_DIR),
        "--add-dir",
        str(SCRATCH_DIR),
        "--allowedTools",
        "Read,Write,Edit,Bash",
        "--permission-mode",
        "dontAsk",
        "--model",
        model,
        "--max-budget-usd",
        MAX_BUDGET_USD,
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(FINDINGS_SCHEMA),
    ]

    proc = subprocess.run(
        cmd, cwd=str(TEST_DIR), capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        if "not logged in" in proc.stdout.lower() or "/login" in proc.stdout.lower():
            raise RuntimeError(
                "claude is not authenticated — run `claude` and `/login` (or set "
                "ANTHROPIC_API_KEY) before running the red-team."
            )
        raise RuntimeError(
            f"claude exited with code {proc.returncode}.\n"
            f"stderr (first 2000 chars):\n{proc.stderr[:2000]}\n"
            f"stdout (first 2000 chars):\n{proc.stdout[:2000]}"
        )
    return _parse_claude_output(proc.stdout, proc.stderr)


def blocking_findings(findings: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        f
        for f in findings.get("findings", [])
        if str(f.get("severity", "")).lower() in BLOCKING_SEVERITIES
    ]


def _render_markdown(findings: dict[str, Any]) -> str:
    items = findings.get("findings", [])
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    items = sorted(items, key=lambda f: order.get(str(f.get("severity")).lower(), 9))

    lines = [
        "# Keenable × Haystack — Claude-Code red-team report",
        "",
        f"_Model: `{MODEL}` · targets: `KeenableWebSearch`, `KeenableFetcher`_",
        "",
        "## Summary",
        "",
        findings.get("summary", "(no summary provided)"),
        "",
        f"**{len(items)} finding(s)** · {len(blocking_findings(findings))} blocking (high/critical).",
        "",
    ]

    if items:
        lines += ["## Findings", ""]
        for i, f in enumerate(items, 1):
            lines += [
                f"### {i}. [{str(f.get('severity', '?')).upper()}] {f.get('title', '(untitled)')}",
                "",
                f"- **Category:** {f.get('category', '-')}",
                f"- **Location:** {f.get('location', '-')}",
                f"- **Reproduced:** {f.get('reproduced', False)}",
                f"- **Evidence:** {f.get('evidence', '-')}",
                f"- **Recommendation:** {f.get('recommendation', '-')}",
                "",
            ]
    else:
        lines += ["## Findings", "", "_None reported._", ""]

    passed = findings.get("passed_checks") or []
    if passed:
        lines += ["## Verified defenses", ""]
        lines += [f"- {c}" for c in passed]
        lines += [""]

    probes = findings.get("probes_run") or []
    if probes:
        lines += ["## Probes run", ""]
        lines += [f"- `{p}`" for p in probes]
        lines += [""]

    return "\n".join(lines)


def write_report(findings: dict[str, Any], out_dir: Path = TEST_DIR) -> tuple[Path, Path]:
    json_path = out_dir / "red_team_report.json"
    md_path = out_dir / "red_team_report.md"
    json_path.write_text(json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_render_markdown(findings), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    if not claude_available():
        print("claude CLI not available (install + authenticate Claude Code first).")
        return 2
    findings = run_red_team()
    json_path, md_path = write_report(findings)
    blocking = blocking_findings(findings)

    print(f"\nReport written to:\n  {json_path}\n  {md_path}")
    print(f"\n{len(findings.get('findings', []))} finding(s); {len(blocking)} blocking.")
    for f in blocking:
        print(f"  [{f.get('severity', '?').upper()}] {f.get('title')}")
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
