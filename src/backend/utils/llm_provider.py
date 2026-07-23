# -*- coding: utf-8 -*-
"""Small provider adapter for OpenAI-compatible chat and Gemini REST calls."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ChatResult:
    content: str
    usage: Dict[str, Any] | None = None
    provider: str = ""
    model: str = ""


def active_provider() -> str:
    return (os.getenv("PROVIDER", "openai") or "openai").strip().lower()


def default_model(provider: str | None = None) -> str:
    provider = (provider or active_provider()).strip().lower()
    if provider in ("openai", "chatgpt"):
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    if provider == "deepseek":
        return os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()


def provider_api_key(provider: str | None = None) -> str:
    provider = (provider or active_provider()).strip().lower()
    if provider in ("openai", "chatgpt"):
        return os.getenv("OPENAI_API_KEY", "").strip()
    if provider == "deepseek":
        return os.getenv("DEEPSEEK_API_KEY", "").strip()
    if provider == "gemini":
        return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    return ""


def looks_like_placeholder_key(key: str) -> bool:
    key_l = (key or "").strip().lower()
    if not key_l:
        return True
    placeholder_fragments = (
        "your_",
        "replace",
        "placeholder",
        "example",
        "changeme",
        "here",
        "todo",
        "xxx",
    )
    return any(fragment in key_l for fragment in placeholder_fragments)


def configured_providers(include_placeholders: bool = False) -> List[str]:
    """Return text-chat providers that have API keys configured in .env."""
    providers: List[str] = []
    for provider in ("openai", "gemini", "deepseek"):
        key = provider_api_key(provider)
        if not key:
            continue
        if not include_placeholders and looks_like_placeholder_key(key):
            continue
        providers.append(provider)
    if not providers:
        active = active_provider()
        if active in {"openai", "gemini", "deepseek"}:
            providers.append(active)
    return providers


def probe_provider(provider: str, model: str | None = None) -> tuple[bool, str]:
    """Run a tiny health check to confirm the provider can answer now."""
    provider = (provider or "").strip().lower()
    if provider not in {"openai", "chatgpt", "deepseek", "gemini"}:
        return False, f"unknown provider: {provider}"
    try:
        result = chat_completion(
            messages=[
                {"role": "system", "content": "Reply with OK."},
                {"role": "user", "content": "Health check."},
            ],
            provider=provider,
            model=model or default_model(provider),
            temperature=0.0,
            max_tokens=2,
        )
        content = (result.content or "").strip()
        if not content:
            return False, "empty health-check response"
        return True, content
    except Exception as exc:
        return False, str(exc)


def openai_base_url(provider: str) -> str:
    if provider == "deepseek":
        return os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1").strip()
    return os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").strip()


def _timeout() -> tuple[int, int]:
    connect = int(os.getenv("HTTP_TIMEOUT_CONNECT", "12"))
    read = int(os.getenv("HTTP_TIMEOUT_READ", os.getenv("OPENAI_TIMEOUT_SECONDS", "120")))
    return connect, read


def _messages_to_gemini_parts(messages: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    system_parts: List[str] = []
    contents: List[Dict[str, Any]] = []
    for message in messages:
        role = (message.get("role") or "user").strip().lower()
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text") or ""))
            text = "\n".join(part for part in text_parts if part)
        else:
            text = str(content or "")
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
        else:
            contents.append({
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": text}],
            })
    return "\n\n".join(system_parts), contents


def _call_openai_compatible(
    provider: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    json_object: bool,
    seed: Optional[int],
) -> ChatResult:
    key = provider_api_key(provider)
    if not key:
        raise RuntimeError(f"{provider.upper()} API key is missing.")
    url = openai_base_url(provider).rstrip("/") + "/chat/completions"
    body: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if json_object:
        body["response_format"] = {"type": "json_object"}
    if seed is not None:
        body["seed"] = seed
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=body,
        timeout=_timeout(),
    )
    if resp.status_code != 200:
        raise RuntimeError(f"{provider} error {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if not content:
        raise RuntimeError(f"{provider} returned empty content.")
    return ChatResult(content=content, usage=data.get("usage"), provider=provider, model=model)


def _call_gemini(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    json_object: bool,
) -> ChatResult:
    key = provider_api_key("gemini")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing.")
    system_text, contents = _messages_to_gemini_parts(messages)
    if not contents:
        raise RuntimeError("Gemini request has no user/model content.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}
    if json_object:
        body["generationConfig"]["responseMimeType"] = "application/json"
    resp = requests.post(
        url,
        params={"key": key},
        headers={"Content-Type": "application/json"},
        json=body,
        timeout=_timeout(),
    )
    if resp.status_code != 200:
        raise RuntimeError(f"gemini error {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
    content = "".join(str(part.get("text") or "") for part in parts if isinstance(part, dict))
    if not content:
        raise RuntimeError("Gemini returned empty content.")
    usage = data.get("usageMetadata")
    return ChatResult(content=content, usage=usage, provider="gemini", model=model)


def _call_openai_compatible_image(
    provider: str,
    model: str,
    prompt: str,
    image_b64: str,
    mime_type: str,
    max_tokens: int,
) -> ChatResult:
    key = provider_api_key(provider)
    if not key:
        raise RuntimeError(f"{provider.upper()} API key is missing.")
    url = openai_base_url(provider).rstrip("/") + "/chat/completions"
    body: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime_type};base64,{image_b64}",
                    "detail": "high",
                }},
            ],
        }],
    }
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=body,
        timeout=_timeout(),
    )
    if resp.status_code != 200:
        raise RuntimeError(f"{provider} vision error {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if not content:
        raise RuntimeError(f"{provider} vision returned empty content.")
    return ChatResult(content=content, usage=data.get("usage"), provider=provider, model=model)


def _call_gemini_image(model: str, prompt: str, image_b64: str, mime_type: str, max_tokens: int) -> ChatResult:
    key = provider_api_key("gemini")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": mime_type, "data": image_b64}},
            ],
        }],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": max_tokens},
    }
    resp = requests.post(
        url,
        params={"key": key},
        headers={"Content-Type": "application/json"},
        json=body,
        timeout=_timeout(),
    )
    if resp.status_code != 200:
        raise RuntimeError(f"gemini vision error {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
    content = "".join(str(part.get("text") or "") for part in parts if isinstance(part, dict))
    if not content:
        raise RuntimeError("Gemini vision returned empty content.")
    return ChatResult(content=content, usage=data.get("usageMetadata"), provider="gemini", model=model)


def chat_completion(
    messages: List[Dict[str, Any]],
    model: str | None = None,
    provider: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    json_object: bool = False,
    seed: Optional[int] = None,
) -> ChatResult:
    provider = (provider or active_provider()).strip().lower()
    model = model or default_model(provider)
    if provider in ("openai", "chatgpt"):
        return _call_openai_compatible("openai", model, messages, temperature, max_tokens, json_object, seed)
    if provider == "deepseek":
        return _call_openai_compatible("deepseek", model, messages, temperature, max_tokens, json_object, seed)
    if provider == "gemini":
        return _call_gemini(model, messages, temperature, max_tokens, json_object)
    raise ValueError(f"Unknown PROVIDER: {provider}. Expected openai / deepseek / gemini.")


def vision_completion(
    prompt: str,
    image_b64: str,
    mime_type: str = "image/png",
    model: str | None = None,
    provider: str | None = None,
    max_tokens: int = 512,
) -> ChatResult:
    provider = (provider or active_provider()).strip().lower()
    model = model or default_model(provider)
    if provider in ("openai", "chatgpt"):
        return _call_openai_compatible_image("openai", model, prompt, image_b64, mime_type, max_tokens)
    if provider == "deepseek":
        raise RuntimeError("DeepSeek vision is not supported by this adapter.")
    if provider == "gemini":
        return _call_gemini_image(model, prompt, image_b64, mime_type, max_tokens)
    raise ValueError(f"Unknown PROVIDER: {provider}. Expected openai / deepseek / gemini.")


def chat_json(
    messages: List[Dict[str, Any]],
    model: str | None = None,
    provider: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    result = chat_completion(
        messages=messages,
        model=model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        json_object=True,
        seed=seed,
    )
    return json.loads(result.content)


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Usage:
    def __init__(self, usage: Dict[str, Any] | None):
        usage = usage or {}
        self.prompt_tokens = usage.get("prompt_tokens") or usage.get("promptTokenCount")
        self.completion_tokens = usage.get("completion_tokens") or usage.get("candidatesTokenCount")
        self.total_tokens = usage.get("total_tokens") or usage.get("totalTokenCount")


class _Response:
    def __init__(self, result: ChatResult):
        self.choices = [_Choice(result.content)]
        self.usage = _Usage(result.usage)


class _ChatCompletions:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    def create(self, **kwargs: Any) -> _Response:
        response_format = kwargs.get("response_format") or {}
        json_object = response_format.get("type") == "json_object"
        result = chat_completion(
            provider=self.provider,
            model=kwargs.get("model") or self.model,
            messages=kwargs.get("messages") or [],
            temperature=float(kwargs.get("temperature", 0.0)),
            max_tokens=int(kwargs.get("max_tokens", 1000)),
            json_object=json_object,
            seed=kwargs.get("seed"),
        )
        return _Response(result)


class _Chat:
    def __init__(self, provider: str, model: str):
        self.completions = _ChatCompletions(provider, model)


class UnifiedChatClient:
    """OpenAI-like client for code that calls client.chat.completions.create."""

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or active_provider()).strip().lower()
        self.model = model or default_model(self.provider)
        self.chat = _Chat(self.provider, self.model)
