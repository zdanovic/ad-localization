from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import httpx

from ..dto import TTSRequest, TTSResponse
from ..errors import AIAdapterError, ConfigError

logger = logging.getLogger(__name__)


class OpenAITTSAdapter:
    """OpenAI TTS adapter using /v1/audio/speech.

    - Model: gpt-4o-mini-tts (default)
    - Voices: alloy, verse, shimmer, etc.
    - Language: auto; select output language via input text language.
    - Voice cloning: not supported; provides high-quality natural voices.
    """

    def __init__(self, api_key: Optional[str], model: str = "gpt-4o-mini-tts", base_url: Optional[str] = None, timeout: float = 120.0) -> None:
        if not api_key:
            raise ConfigError("OPENAI_API_KEY is required for OpenAI TTS")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/") if base_url else "https://api.openai.com/v1"
        self.timeout = timeout

    def synthesize_to_file(self, req: TTSRequest, out_path: str, audio_format: str = "wav") -> TTSResponse:
        url = f"{self.base_url}/audio/speech"
        voice = req.voice or "alloy"
        payload = {
            "model": self.model,
            "input": req.text,
            "voice": voice,
            "format": audio_format,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            attempts = 3
            last_exc: Exception | None = None
            with httpx.Client(timeout=self.timeout) as client:
                for i in range(attempts):
                    try:
                        r = client.post(url, headers=headers, json=payload)
                        r.raise_for_status()
                        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                        Path(out_path).write_bytes(r.content)
                        return TTSResponse(audio_path=out_path, format=audio_format)
                    except Exception as e:  # noqa: BLE001
                        last_exc = e
                        if i < attempts - 1:
                            # brief backoff
                            import time

                            time.sleep(1.0 * (i + 1))
                        else:
                            raise
        except Exception as e:  # noqa: BLE001
            raise AIAdapterError(f"OpenAI TTS failed: {e}") from e
