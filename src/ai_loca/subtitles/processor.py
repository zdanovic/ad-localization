from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List

import pysubs2

from ..dto import Transcript, TranscriptSegment


def normalize_timings(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    # Ensure non-overlapping, non-negative durations; minimal gap enforcement
    MIN_GAP = 0.05
    prev_end = 0.0
    norm: List[TranscriptSegment] = []
    for seg in segments:
        start = max(prev_end, max(0.0, seg.start))
        end = max(start + MIN_GAP, seg.end)
        norm.append(TranscriptSegment(start=start, end=end, text=seg.text))
        prev_end = end
    return norm


def transcript_to_subs(transcript: Transcript) -> pysubs2.SSAFile:
    subs = pysubs2.SSAFile()
    for seg in normalize_timings(transcript.segments):
        ev = pysubs2.SSAEvent(
            start=round(seg.start * 1000),
            end=round(seg.end * 1000),
            text=seg.text,
        )
        subs.events.append(ev)
    return subs


def import_subtitles(path: str) -> pysubs2.SSAFile:
    return pysubs2.load(path)


def export_srt(subs: pysubs2.SSAFile, out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    subs.save(out_path, format_="srt")
    return out_path


def export_vtt(subs: pysubs2.SSAFile, out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    subs.save(out_path, format_="vtt")
    return out_path


def subs_to_transcript(subs: pysubs2.SSAFile, language: str | None = None) -> Transcript:
    segments: list[TranscriptSegment] = []
    for ev in subs.events:
        segments.append(
            TranscriptSegment(start=ev.start / 1000.0, end=ev.end / 1000.0, text=str(ev.text))
        )
    return Transcript(language=language, segments=segments)
