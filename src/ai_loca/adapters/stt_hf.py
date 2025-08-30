from __future__ import annotations

"""
Hugging Face Whisper STT Adapter
--------------------------------

Transcribe audio using Whisper (or compatible ASR models) served by the
Hugging Face Inference API.

Examples
~~~~~~~~

Basic usage:

    from ai_loca.adapters.stt_hf import HFWhisperSTTAdapter
    from ai_loca.config import Settings

    settings = Settings.from_env()
    stt = HFWhisperSTTAdapter(api_token=settings.hf_api_token, model=settings.stt_model, timeout=settings.http_timeout, retries=settings.http_retries)
    transcript = stt.transcribe("/path/to/audio.wav", language="en")
    print(transcript.text)

With explicit model:

    stt = HFWhisperSTTAdapter(api_token="hf_xxx", model="openai/whisper-small")
    transcript = stt.transcribe("audio.wav")

Notes
~~~~~
- Requires `HF_API_TOKEN` unless the model is public and free.
- Response structure varies by model; adapter normalizes to Transcript DTO.

"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..dto import Transcript, TranscriptSegment
from ..errors import AIAdapterError, ConfigError
from ..http_client import HttpClient

logger = logging.getLogger(__name__)


class HFWhisperSTTAdapter:
    def __init__(
        self,
        api_token: Optional[str],
        model: str,
        endpoint_url: Optional[str] = None,
        timeout: float = 60.0,
        retries: int = 3,
    ) -> None:
        if not model:
            raise ConfigError("STT model must be provided")
        self.model = model
        self.endpoint_url = endpoint_url
        self.timeout = timeout
        self.retries = retries
        self._client = HttpClient(timeout=timeout, retries=retries)
        self._headers = {"Accept": "application/json"}
        if api_token:
            self._headers["Authorization"] = f"Bearer {api_token}"

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Transcript:
        url = self.endpoint_url or f"https://api-inference.huggingface.co/models/{self.model}"
        audio_bytes = Path(audio_path).read_bytes()
        # For HF ASR, send raw audio bytes. Some models accept parameters, but
        # binary body + JSON is not consistently supported; keep bytes only.
        headers = {**self._headers, "Content-Type": "audio/wav", "Accept": "application/json"}
        try:
            resp = self._client.request(
                "POST",
                url,
                headers=headers,
                data=None,
                content=audio_bytes,
                params=None,
                json=None,
            )
            data = resp.json()
            # Normalize possible outputs: {"text": "...", "chunks"|"segments": [...]}
            text = data.get("text", "")
            segments_raw = data.get("segments") or data.get("chunks") or []
            segments: List[TranscriptSegment] = []
            if segments_raw and isinstance(segments_raw, list):
                for seg in segments_raw:
                    start = seg.get("timestamp", [seg.get("start", 0), seg.get("end", 0)])[0]
                    end = seg.get("timestamp", [seg.get("start", 0), seg.get("end", 0)])[1]
                    segments.append(TranscriptSegment(start=float(start or 0), end=float(end or 0), text=seg.get("text", "")))
            elif text:
                segments.append(TranscriptSegment(start=0.0, end=max(0.1, len(text) / 15.0), text=text))
            return Transcript(language=language, segments=segments)
        except Exception as e:  # noqa: BLE001
            raise AIAdapterError(f"STT failed: {e}") from e
