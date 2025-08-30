from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioPreset:
    codec: str = "aac"
    bitrate: str = "192k"
    sample_rate: int = 44100


@dataclass(frozen=True)
class VideoPreset:
    codec: str = "libx264"
    crf: int = 23
    preset: str = "medium"


DEFAULT_AUDIO = AudioPreset()
DEFAULT_VIDEO = VideoPreset()

