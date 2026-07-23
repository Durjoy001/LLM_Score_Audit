# -*- coding: utf-8 -*-
"""Quick provider key validator for OpenAI-compatible and Gemini providers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

from src.backend.utils.llm_provider import chat_json, default_model, provider_api_key


@dataclass
class CheckResult:
    provider: str
    ok: bool
    message: str = ""


def _mask_key(key: str) -> str:
    key = (key or "").strip()
    if not key:
        return "<empty>"
    if len(key) <= 10:
        return key[:4] + "..."
    return f"{key[:4]}...{key[-4:]}"


def _check(provider: str, model: str) -> CheckResult:
    key = provider_api_key(provider)
    if not key:
        return CheckResult(provider=provider, ok=False, message="missing key")
    try:
        resp = chat_json(
            provider=provider,
            model=model,
            temperature=0.0,
            max_tokens=8,
            messages=[
                {"role": "system", "content": "Return only OK."},
                {"role": "user", "content": "Health check."},
            ],
        )
        return CheckResult(provider=provider, ok=True, message=str(resp)[:120])
    except Exception as exc:
        return CheckResult(provider=provider, ok=False, message=str(exc))


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate configured provider keys with a tiny live call.")
    ap.add_argument("providers", nargs="*", help="Providers to test, e.g. gemini deepseek openai.")
    ap.add_argument("--model", default="", help="Override model for all providers.")
    args = ap.parse_args()

    providers: List[str] = [p.strip().lower() for p in args.providers if p.strip()]
    if not providers:
        providers = ["gemini", "deepseek"]

    any_ok = False
    for provider in providers:
        model = args.model.strip() or default_model(provider)
        result = _check(provider, model)
        any_ok = any_ok or result.ok
        status = "OK" if result.ok else "FAIL"
        key = provider_api_key(provider)
        print(f"[{status}] {provider} key={_mask_key(key)} model={model} {result.message}")

    raise SystemExit(0 if any_ok else 1)


if __name__ == "__main__":
    main()
