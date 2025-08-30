from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .errors import ConfigError


@dataclass
class Settings:
    hf_api_token: Optional[str]
    stt_model: str
    mt_model: str
    tts_model: str
    stt_endpoint: Optional[str]
    mt_endpoint: Optional[str]
    tts_endpoint: Optional[str]
    stt_provider: str  # hf | openai
    tts_provider: str  # hf | elevenlabs
    mt_provider: str  # hf | openai
    openai_api_key: Optional[str]
    openai_base_url: Optional[str]
    elevenlabs_api_key: Optional[str]
    ffmpeg_bin: str
    tmp_dir: str
    http_timeout: float
    http_retries: int
    tts_model_openai: str = "gpt-4o-mini-tts"

    @staticmethod
    def from_env() -> "Settings":
        hf_api_token = os.environ.get("HF_API_TOKEN")
        stt_model = os.environ.get("STT_MODEL", "openai/whisper-small")
        mt_model = os.environ.get("MT_MODEL", "facebook/nllb-200-distilled-600M")
        tts_model = os.environ.get("TTS_MODEL", "espnet/kan-bayashi-ljspeech")
        ffmpeg_bin = os.environ.get("FFMPEG_BIN", "ffmpeg")
        stt_endpoint = os.environ.get("STT_ENDPOINT")
        mt_endpoint = os.environ.get("MT_ENDPOINT")
        tts_endpoint = os.environ.get("TTS_ENDPOINT")
        stt_provider = os.environ.get("STT_PROVIDER", "hf").lower()
        tts_provider = os.environ.get("TTS_PROVIDER", "hf").lower()
        mt_provider = os.environ.get("MT_PROVIDER", "openai").lower()
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        openai_base_url = os.environ.get("OPENAI_BASE_URL")
        elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
        # Optional OpenAI TTS model
        tts_model_openai = os.environ.get("TTS_MODEL_OPENAI", "gpt-4o-mini-tts")
        tmp_dir = os.environ.get("TMP_DIR", ".ai_loca_tmp")
        http_timeout = float(os.environ.get("HTTP_TIMEOUT", "60"))
        http_retries = int(os.environ.get("HTTP_RETRIES", "3"))

        if not ffmpeg_bin:
            raise ConfigError("FFMPEG_BIN is required")

        return Settings(
            hf_api_token=hf_api_token,
            stt_model=stt_model,
            mt_model=mt_model,
            tts_model=tts_model,
            stt_endpoint=stt_endpoint,
            mt_endpoint=mt_endpoint,
            tts_endpoint=tts_endpoint,
            stt_provider=stt_provider,
            tts_provider=tts_provider,
            mt_provider=mt_provider,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            elevenlabs_api_key=elevenlabs_api_key,
            tts_model_openai=tts_model_openai,
            ffmpeg_bin=ffmpeg_bin,
            tmp_dir=tmp_dir,
            http_timeout=http_timeout,
            http_retries=http_retries,
        )
