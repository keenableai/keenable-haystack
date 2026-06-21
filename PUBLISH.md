# Publishing `keenable-haystack` to PyPI

deepset's `haystack-core-integrations` monorepo is deepset-maintained; new
third-party integrations are published independently to PyPI and then listed in
the catalog (a markdown PR — see the integration `STATUS.md`). So this is our own
package, published via **Trusted Publishing (OIDC)** the same way as
`langchain-keenable` / `llama-index-tools-keenable`: GitHub Actions mints a
short-lived identity token and PyPI trusts it, so **no API tokens are stored**.
Workflow: [`.github/workflows/publish.yml`](.github/workflows/publish.yml), runs
on a published GitHub Release.

## One-time setup (PyPI account owner, once)

PyPI account needs a **verified email** + **2FA**. Then at
**https://pypi.org/manage/account/publishing/** add a *pending publisher* with
exactly these values (a mismatch, usually the environment name, is the #1 cause
of a silent "not a trusted publisher" failure):

| Field | Value |
|---|---|
| PyPI Project Name | `keenable-haystack` |
| Owner | `keenableai` |
| Repository name | `keenable-haystack` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

Then in the repo: **Settings → Environments → New environment → `pypi`**, and
add required reviewers so a release needs human approval before it publishes.

## Cut a release

```bash
# in keenableai/keenable-haystack, on main:
# 1) bump `version` in pyproject.toml
git tag v0.1.0 && git push origin v0.1.0      # tag must match pyproject version
gh release create v0.1.0 --title v0.1.0 --notes "Initial release"
```

Publishing the Release triggers the workflow (build → `twine check` → OIDC
publish). Approve the `pypi` environment when prompted; on success the package
is live at https://pypi.org/project/keenable-haystack/.

## Pre-release checks (local)

```bash
rm -rf dist && uv build && uvx twine check dist/*
uv venv && . .venv/bin/activate
uv pip install -e . pytest
pytest                                          # offline
ruff check haystack_integrations
mypy haystack_integrations
```

## If it goes silent

- No verification email → Spam; resend; link expires; try a personal email.
- 2FA not enabled → can't publish or make tokens; enable a TOTP app.
- "not a trusted publisher" → pending-publisher fields don't match (env `pypi` /
  workflow `publish.yml`).
- Same version re-upload → bump the version.

Manual token fallback (only if Actions is unavailable):
`UV_PUBLISH_TOKEN=pypi-XXXX uv publish` (user is `__token__`). Prefer the OIDC
workflow.
