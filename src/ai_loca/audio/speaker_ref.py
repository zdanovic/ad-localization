from __future__ import annotations

import wave
from pathlib import Path

import webrtcvad

from ..config import Settings
from ..utils.ffmpeg import run_ffmpeg, slice_wav
from .stems import build_stems_ms
from .bed import _resample_to_16k_mono


def build_speaker_ref(settings: Settings, input_video: str, out_dir: str, dur_sec: float = 2.5, force: bool = False) -> str:
    """Extract a short, clean speaker reference WAV from the original video.

    Strategy: fast mid/side stems to isolate vocals, resample to 16k mono, use VAD to find first clear speech window,
    slice ~2.5s clip, and export as PCM s16le.
    """
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    key = Path(input_video).stem
    out_wav = str(base / f"{key}_speaker_ref.wav")
    if Path(out_wav).exists() and not force:
        return out_wav
    # 1) quick vocals
    vocals_wav, _acc, _rep = build_stems_ms(settings, input_video=input_video, out_dir=out_dir, force=force)
    # 2) resample to 16k mono
    v16 = str(base / f"{key}_vocals_16k.wav")
    _resample_to_16k_mono(settings.ffmpeg_bin, vocals_wav, v16)
    # 3) scan with VAD to get first voiced region
    vad = webrtcvad.Vad(2)
    with wave.open(v16, "rb") as wf:
        sr = wf.getframerate()
        assert sr == 16000
        ch = wf.getnchannels()
        assert ch == 1
        w = wf.getsampwidth()
        assert w == 2
        frame_ms = 30
        frame_bytes = int(sr * (frame_ms / 1000.0) * w)
        voiced_positions = []
        offset = 0
        while True:
            buf = wf.readframes(int(frame_bytes / w))
            if len(buf) < frame_bytes:
                break
            t = offset * frame_ms / 1000.0
            if vad.is_speech(buf, sr):
                voiced_positions.append(t)
            offset += 1
        start = voiced_positions[0] if voiced_positions else 0.0
    # 4) slice a short ref
    slice_wav(settings.ffmpeg_bin, vocals_wav, out_wav, start_sec=max(0.0, start), duration_sec=dur_sec)
    return out_wav

