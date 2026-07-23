# -*- coding: utf-8 -*-
"""
General model selector (v5.1; compatible with old/new Gemini SDKs and explicit key priority)
- PROVIDER: openai / deepseek / gemini
- Returns: {"client": <client_instance>, "model_name": str, "provider": str}
"""

import os
from dotenv import load_dotenv

load_dotenv()

def get_llm_client():
    provider = (os.getenv("PROVIDER", "openai") or "openai").lower().strip()

    # ========== OpenAI ==========
    if provider in ("openai", "chatgpt"):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        client = OpenAI(api_key=api_key)
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"Loaded OpenAI model: {model_name}")
        return {"client": client, "model_name": model_name, "provider": "openai"}

    # ========== DeepSeek ==========
    elif provider == "deepseek":
        from openai import OpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured.")
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        print(f"Loaded DeepSeek model: {model_name}")
        return {"client": client, "model_name": model_name, "provider": "deepseek"}

    # ========== Gemini ==========
    elif provider == "gemini":
        # Prefer GEMINI_API_KEY; fall back to GOOGLE_API_KEY if needed.
        api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured, and GOOGLE_API_KEY fallback was not found.")

        # Prefer the new SDK (from google import genai), then fall back to google.generativeai.
        client = None
        try:
            from google import genai  # New SDK.
            client = genai.Client(api_key=api_key)
            using_new = True
        except Exception:
            import google.generativeai as genai  # Old SDK.
            genai.configure(api_key=api_key)
            # The old SDK has no Client instance semantics; use the module as a client placeholder.
            client = genai
            using_new = False

        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        suffix = "(new SDK)" if using_new else "(old SDK)"
        print(f"Loaded Gemini model: {model_name} {suffix}")
        return {"client": client, "model_name": model_name, "provider": "gemini"}

    else:
        raise ValueError(f"Unknown PROVIDER: {provider}. Expected openai / deepseek / gemini.")
