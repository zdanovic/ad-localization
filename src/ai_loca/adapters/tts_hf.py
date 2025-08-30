from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import base64

from ..dto import TTSRequest, TTSResponse
from ..errors import AIAdapterError, ConfigError
from ..http_client import HttpClient

logger = logging.getLogger(__name__)


class HFTTSAdapter:
    """Hugging Face TTS adapter.

    Many TTS models return raw audio bytes from the Inference API. This adapter
    writes them to a file and returns a `TTSResponse`.

    Example:

        from ai_loca.adapters.tts_hf import HFTTSAdapter
        tts = HFTTSAdapter(api_token="hf_xxx", model="espnet/kan-bayashi-ljspeech")
        res = tts.synthesize_to_file(TTSRequest(text="Hello world"), out_path="./out.wav")
        print(res.audio_path)

    Voice selection is model-dependent; many models do not accept a `voice` parameter.
    Provide multiple adapters or distinct models for different voices as needed.
    """

    def __init__(self, api_token: Optional[str], model: str, endpoint_url: Optional[str] = None, timeout: float = 60.0, retries: int = 3) -> None:
        if not model:
            raise ConfigError("TTS model must be provided")
        self.model = model
        self.endpoint_url = endpoint_url
        self._client = HttpClient(timeout=timeout, retries=retries)
        self._headers = {}
        if api_token:
            self._headers["Authorization"] = f"Bearer {api_token}"

    def synthesize_to_file(self, req: TTSRequest, out_path: str, audio_format: str = "wav") -> TTSResponse:
        url = self.endpoint_url or f"https://api-inference.huggingface.co/models/{self.model}"
        payload: Dict[str, Any] = {"inputs": req.text}
        params: Dict[str, Any] = {}
        if req.voice:
            params["voice"] = req.voice
        if req.language:
            params["language"] = req.language
        if req.speaker_wav_path:
            b = Path(req.speaker_wav_path).read_bytes()
            b64 = base64.b64encode(b).decode("ascii")
            # Send common keys used by popular TTS endpoints (XTTS, CosyVoice, OpenVoice)
            params["speaker_wav"] = b64
            params["prompt_wav"] = b64
        # Optional prosody/style params (adapter-dependent, best-effort)
        if req.style:
            params["style"] = req.style
        if req.emotion:
            params["emotion"] = req.emotion
        if req.speed is not None:
            params["speed"] = req.speed
        if req.pitch is not None:
            params["pitch"] = req.pitch
        if params:
            payload["parameters"] = params
        headers = {**self._headers, "Accept": f"audio/{audio_format}", "Content-Type": "application/json"}
        try:
            resp = self._client.request("POST", url, headers=headers, json=payload)
            content = resp.content
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_bytes(content)
            return TTSResponse(audio_path=out_path, format=audio_format)
        except Exception as e:  # noqa: BLE001
            raise AIAdapterError(f"TTS failed: {e}") from e
