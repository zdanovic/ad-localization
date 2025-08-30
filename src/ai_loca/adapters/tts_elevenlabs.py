from __future__ import annotations

import base64
import hashlib
import logging
from pathlib import Path
from typing import Optional

import httpx

from ..dto import TTSRequest, TTSResponse
from ..errors import AIAdapterError, ConfigError

logger = logging.getLogger(__name__)


class ElevenLabsTTSAdapter:
    """ElevenLabs TTS adapter with optional instant voice cloning.

    - If `req.voice` is provided, treated as `voice_id`.
    - Else if `req.speaker_wav_path` is provided, creates a temporary voice via /v1/voices/add,
      then uses it for synthesis.
    - Uses `eleven_multilingual_v2` model for multi-language support by default.
    """

    def __init__(self, api_key: Optional[str], model_id: str = "eleven_multilingual_v2", timeout: float = 120.0) -> None:
        if not api_key:
            raise ConfigError("ELEVENLABS_API_KEY is required for ElevenLabs TTS")
        self.api_key = api_key
        self.model_id = model_id
        self.timeout = timeout
        self._voice_cache: dict[str, str] = {}

    def _ensure_voice(self, client: httpx.Client, req: TTSRequest) -> str:
        if req.voice:
            return req.voice
        if not req.speaker_wav_path:
            raise ConfigError("Either voice (voice_id) or speaker_wav_path must be provided for ElevenLabs")
        wav_bytes = Path(req.speaker_wav_path).read_bytes()
        h = hashlib.sha256(wav_bytes).hexdigest()[:8]
        if h in self._voice_cache:
            return self._voice_cache[h]
        url = "https://api.elevenlabs.io/v1/voices/add"
        name = f"ai-loca-{h}"
        files = {"files": (Path(req.speaker_wav_path).name, wav_bytes, "audio/wav")}
        data = {"name": name}
        headers = {"xi-api-key": self.api_key}
        r = client.post(url, headers=headers, data=data, files=files)
        r.raise_for_status()
        vid = r.json().get("voice_id")
        if not vid:
            raise AIAdapterError("Failed to create ElevenLabs voice")
        self._voice_cache[h] = vid
        return vid

    def synthesize_to_file(self, req: TTSRequest, out_path: str, audio_format: str = "wav") -> TTSResponse:
        headers = {"xi-api-key": self.api_key, "Accept": f"audio/{audio_format}", "Content-Type": "application/json"}
        with httpx.Client(timeout=self.timeout) as client:
            try:
                voice_id = self._ensure_voice(client, req)
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                payload = {
                    "text": req.text,
                    "model_id": self.model_id,
                    "voice_settings": {"stability": 0.45, "similarity_boost": 0.85},
                }
                # ElevenLabs auto-detects language in multilingual models; explicit code not required
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                Path(out_path).write_bytes(resp.content)
                return TTSResponse(audio_path=out_path, format=audio_format)
            except Exception as e:  # noqa: BLE001
                raise AIAdapterError(f"ElevenLabs TTS failed: {e}") from e
