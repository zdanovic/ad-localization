from __future__ import annotations

import json
import os
import random
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import webrtcvad

from ..config import Settings
from ..utils.ffmpeg import (
    extract_audio,
    make_music_bed,
    analyze_loudness,
)


@dataclass
class BedReport:
    method: str
    input_video: str
    input_audio: str
    bed_audio: str
    orig_lufs: float
    bed_lufs: float
    vad_ratio_bed: float
    params: dict


def _resample_to_16k_mono(ffmpeg_bin: str, in_wav: str, out_wav: str) -> str:
    from ..utils.ffmpeg import run_ffmpeg

    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    args = [ffmpeg_bin, "-y", "-i", in_wav, "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", out_wav]
    run_ffmpeg(args)
    return out_wav


def _vad_ratio(wav_path: str, frame_ms: int = 30, aggressiveness: int = 2) -> float:
    vad = webrtcvad.Vad(aggressiveness)
    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        assert sample_rate == 16000
        channels = wf.getnchannels()
        assert channels == 1
        width = wf.getsampwidth()
        assert width == 2
        frame_bytes = int(sample_rate * (frame_ms / 1000.0) * width)
        voiced = 0
        total = 0
        while True:
            buf = wf.readframes(int(frame_bytes / width))
            if len(buf) < frame_bytes:
                break
            total += 1
            if vad.is_speech(buf, sample_rate):
                voiced += 1
        return (voiced / total) if total else 0.0


def build_music_bed(
    settings: Settings,
    input_video: str,
    out_dir: Optional[str] = None,
    method: str = "ms_gate",
    mlev: float = 0.6,
    base_volume: float = 0.35,
    force: bool = False,
) -> Tuple[str, str]:
    base = Path(out_dir or settings.tmp_dir)
    base.mkdir(parents=True, exist_ok=True)
    key = Path(input_video).stem
    audio_wav = str(base / f"{key}_audio.wav")
    if not Path(audio_wav).exists() or force:
        extract_audio(settings.ffmpeg_bin, input_video, audio_wav)

    bed_wav = str(base / f"{key}_bed.wav")
    if method == "ms_gate":
        if not Path(bed_wav).exists() or force:
            make_music_bed(settings.ffmpeg_bin, audio_wav, bed_wav, mlev=mlev, base_volume=base_volume)
    else:
        raise ValueError(f"Unknown bed method: {method}")

    # Metrics
    orig_lufs = analyze_loudness(settings.ffmpeg_bin, audio_wav)
    bed_lufs = analyze_loudness(settings.ffmpeg_bin, bed_wav)
    bed_16k = str(base / f"{key}_bed_16k.wav")
    _resample_to_16k_mono(settings.ffmpeg_bin, bed_wav, bed_16k)
    vad_ratio_bed = _vad_ratio(bed_16k)

    report = BedReport(
        method=method,
        input_video=input_video,
        input_audio=audio_wav,
        bed_audio=bed_wav,
        orig_lufs=orig_lufs,
        bed_lufs=bed_lufs,
        vad_ratio_bed=vad_ratio_bed,
        params={"mlev": mlev, "base_volume": base_volume},
    )
    report_path = str(base / f"{key}_bed_report.json")
    Path(report_path).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return bed_wav, report_path

