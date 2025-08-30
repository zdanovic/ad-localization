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


class TTSResponse(BaseModel):
    audio_path: str
    format: str
