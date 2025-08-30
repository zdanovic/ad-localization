from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import httpx

from ..dto import Transcript, TranscriptSegment
from ..errors import AIAdapterError, ConfigError

logger = logging.getLogger(__name__)


class OpenAIWhisperSTTAdapter:
    def __init__(self, api_key: Optional[str], model: str = "whisper-1", base_url: Optional[str] = None, timeout: float = 120.0) -> None:
        if not api_key:
            raise ConfigError("OPENAI_API_KEY is required for OpenAI STT")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/") if base_url else "https://api.openai.com/v1"
        self.timeout = timeout

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Transcript:
        url = f"{self.base_url}/audio/transcriptions"
        files = {
            "file": (Path(audio_path).name, Path(audio_path).open("rb"), "audio/wav"),
        }
        data = {
            "model": self.model,
            "response_format": "verbose_json",
        }
        if language:
            data["language"] = language
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, headers=headers, data=data, files=files)
                resp.raise_for_status()
                payload = resp.json()
        except Exception as e:  # noqa: BLE001
            raise AIAdapterError(f"OpenAI STT failed: {e}") from e
        # payload may contain segments
        segments: List[TranscriptSegment] = []
        for seg in payload.get("segments", []) or []:
            segments.append(
                TranscriptSegment(start=float(seg.get("start", 0.0)), end=float(seg.get("end", 0.0)), text=seg.get("text", ""))
            )
        if not segments:
            text = payload.get("text", "")
            if text:
                segments.append(TranscriptSegment(start=0.0, end=max(0.1, len(text) / 15.0), text=text))
        detected_lang = payload.get("language") or language
        # Normalize to ISO-2 if possible (OpenAI often returns 'en', 'es', etc.)
        if isinstance(detected_lang, str) and len(detected_lang) > 5:
            # Probably a language name like 'english'; leave as-is
            pass
        return Transcript(language=detected_lang, segments=segments)
