from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str


class Transcript(BaseModel):
    language: Optional[str] = None
    segments: List[TranscriptSegment] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(seg.text.strip() for seg in self.segments if seg.text)


class TranslationRequest(BaseModel):
    source_lang: str
    target_lang: str
    text: str
    glossary: Optional[dict[str, str]] = None


class TranslationResponse(BaseModel):
    text: str


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    language: Optional[str] = None
    speaker_wav_path: Optional[str] = None
    # Optional prosody controls (adapter-dependent)
    style: Optional[str] = None
    emotion: Optional[str] = None
    speed: Optional[float] = None  # 1.0 = default
    pitch: Optional[float] = None  # semitone delta or provider-specific


class TTSResponse(BaseModel):
    audio_path: str
    format: str


class DubSegmentPlan(BaseModel):
    index: int
    start: float
    end: float
    text: str
    style: Optional[str] = None
    emotion: Optional[str] = None
    speed: Optional[float] = None
    pitch: Optional[float] = None


class DubPlan(BaseModel):
    language: Optional[str] = None
    segments: List[DubSegmentPlan] = Field(default_factory=list)
