from __future__ import annotations

import logging
from typing import Optional

import httpx

from ..dto import TranslationRequest, TranslationResponse
from ..errors import AIAdapterError, ConfigError

logger = logging.getLogger(__name__)


class OpenAITranslationAdapter:
    def __init__(self, api_key: Optional[str], model: str = "gpt-4o-mini", base_url: Optional[str] = None, timeout: float = 60.0) -> None:
        if not api_key:
            raise ConfigError("OPENAI_API_KEY is required for OpenAI MT")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/") if base_url else "https://api.openai.com/v1"
        self.timeout = timeout

    def translate(self, req: TranslationRequest) -> TranslationResponse:
        url = f"{self.base_url}/chat/completions"
        src = req.source_lang or "auto"
        tgt = req.target_lang
        glossary_txt = ""
        if req.glossary:
            pairs = ", ".join(f"{k}->{v}" for k, v in req.glossary.items())
            glossary_txt = f"\nUse glossary where applicable: {pairs}"
        system = (
            "You are a professional advertising translator. "
            "Translate the user text preserving meaning, tone and marketing impact. "
            f"Source language: {src}. Target language: {tgt}." + glossary_txt + "\nOutput only the translated text."
        )
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": req.text},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                out = data["choices"][0]["message"]["content"].strip()
                return TranslationResponse(text=out)
        except Exception as e:  # noqa: BLE001
            raise AIAdapterError(f"OpenAI MT failed: {e}") from e

