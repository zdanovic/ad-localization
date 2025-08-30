from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple
import sys

from ..config import Settings
from ..utils.ffmpeg import (
    analyze_loudness,
    run_ffmpeg,
    gate_bed_with_vocals,
    smooth_bed_with_vocals,
    refill_bed_with_original,
    multiband_refill_bed,
    stabilize_bed_dynaudnorm,
)
from .bed import _resample_to_16k_mono, _vad_ratio


@dataclass
class StemsReport:
    method: str
    input_video: str
    input_audio: str
    vocals_wav: str
    acc_wav: str
    vocals_vad: float
    acc_vad: float
    vocals_lufs: float
    acc_lufs: float
    params: dict


def _auto_tune_bed(
    settings: Settings,
    vocals_wav: str,
    acc_wav: str,
    out_wav: str,
    target_vad: float = 0.05,
    max_iters: int = 1,
) -> tuple[str, dict]:
    """Auto-tune mid-band sidechain compression to minimize residual speech VAD.

    Strategy: try a small grid of (threshold, ratio, makeup) with loudnorm to original LUFS.
    Picks the lowest VAD result; if VAD > target and max_iters>1, can iterate (kept 1 by default for speed).
    Returns output path and chosen params.
    """
    import itertools

    orig_lufs = analyze_loudness(settings.ffmpeg_bin, acc_wav)
    cand_thresholds = [0.05, 0.03, 0.02, 0.015, 0.01]
    cand_ratios = [3.0, 4.0, 6.0, 8.0]
    cand_makeups = [1.5, 2.0]
    best = None
    best_vad = 1e9
    best_path = out_wav
    tmp_base = Path(out_wav).with_suffix("")
    i = 0
    created: list[str] = []
    for thr, rt, mk in itertools.product(cand_thresholds, cand_ratios, cand_makeups):
        i += 1
        trial = f"{tmp_base}_t{i}.wav"
        smooth_bed_with_vocals(
            settings.ffmpeg_bin,
            acc_wav,
            vocals_wav,
            trial,
            target_lufs=orig_lufs,
            threshold=thr,
            ratio=rt,
            makeup=mk,
        )
        created.append(trial)
        # VAD evaluate
        bed16 = f"{tmp_base}_t{i}_16k.wav"
        _resample_to_16k_mono(settings.ffmpeg_bin, trial, bed16)
        created.append(bed16)
        vad = _vad_ratio(bed16)
        if vad < best_vad:
            best_vad = vad
            best = {"threshold": thr, "ratio": rt, "makeup": mk, "lufs": orig_lufs, "vad": vad}
            best_path = trial
        if vad <= target_vad:
            break
    # Move best to out_wav if different
    if str(best_path) != str(out_wav):
        Path(out_wav).write_bytes(Path(best_path).read_bytes())
    # Cleanup trials
    for p in created:
        if Path(p).exists() and Path(p) != Path(out_wav):
            try:
                Path(p).unlink()
            except Exception:
                pass
    return out_wav, best or {}


def _mix_acc(ffmpeg_bin: str, drums: Optional[str], bass: Optional[str], other: Optional[str], out_wav: str) -> str:
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    inputs = [p for p in [drums, bass, other] if p and Path(p).exists()]
    if not inputs:
        raise RuntimeError("No accompaniment stems to mix")
    args = [ffmpeg_bin, "-y"]
    for p in inputs:
        args += ["-i", p]
    args += [
        "-filter_complex",
        f"amix=inputs={len(inputs)}:normalize=0[mix]",
        "-map",
        "[mix]",
        "-acodec",
        "pcm_s16le",
        out_wav,
    ]
    run_ffmpeg(args)
    return out_wav


def build_stems_demucs(
    settings: Settings,
    input_video: str,
    out_dir: Optional[str] = None,
    model: str = "htdemucs",
    use_cpu: bool = True,
    auto_tune_bed: bool = True,
    keep_intermediate: bool = False,
    force: bool = False,
) -> Tuple[str, str, str]:
    base = Path(out_dir or settings.tmp_dir)
    base.mkdir(parents=True, exist_ok=True)
    key = Path(input_video).stem
    audio_wav = str(base / f"{key}_audio.wav")
    if not Path(audio_wav).exists() or force:
        from ..utils.ffmpeg import extract_audio

        extract_audio(settings.ffmpeg_bin, input_video, audio_wav)

    # Run demucs via CLI (simplest, robust)
    demucs_out = base / f"demucs_{key}"
    if force and demucs_out.exists():
        shutil.rmtree(demucs_out, ignore_errors=True)
    demucs_out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if use_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        model,
        audio_wav,
        "-o",
        str(demucs_out),
    ]
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Demucs outputs demucs_out/model/<file_stem>/{vocals,bass,drums,other}.wav
    stem_dir = demucs_out / model / Path(audio_wav).stem
    vocals = stem_dir / "vocals.wav"
    acc = stem_dir / "no_vocals.wav"
    if not acc.exists():
        # mix drums+bass+other
        acc_mixed = base / f"{key}_acc.wav"
        _mix_acc(settings.ffmpeg_bin, str(stem_dir / "drums.wav"), str(stem_dir / "bass.wav"), str(stem_dir / "other.wav"), str(acc_mixed))
        acc = acc_mixed

    # Post-process accompaniment: either auto-tune or single-pass smoothing
    if auto_tune_bed:
        acc_auto = base / f"{key}_acc_clean.wav"
        acc_path, tuned = _auto_tune_bed(settings, str(vocals), str(acc), str(acc_auto))
        acc = Path(acc_path)
    else:
        acc_lufs_raw = analyze_loudness(settings.ffmpeg_bin, str(acc))
        acc_smooth = base / f"{key}_acc_clean.wav"
        smooth_bed_with_vocals(
            settings.ffmpeg_bin,
            str(acc),
            str(vocals),
            str(acc_smooth),
            target_lufs=acc_lufs_raw,
        )
        acc = acc_smooth

    # Refill non-speech regions (frequency-dependent) and stabilize short-term loudness
    acc_refill = base / f"{key}_acc_refill.wav"
    orig_lufs = analyze_loudness(settings.ffmpeg_bin, audio_wav)
    multiband_refill_bed(
        settings.ffmpeg_bin,
        str(acc),
        audio_wav,
        str(vocals),
        str(acc_refill),
        target_lufs=orig_lufs,
    )
    acc_final = base / f"{key}_acc_final.wav"
    stabilize_bed_dynaudnorm(settings.ffmpeg_bin, str(acc_refill), str(acc_final))
    acc = acc_final

    # Metrics
    vocals_stable = base / f"{key}_vocals.wav"
    try:
        if Path(vocals).exists():
            Path(vocals_stable).write_bytes(Path(vocals).read_bytes())
    except Exception:
        pass
    vocals_16k = str(base / f"{key}_vocals_16k.wav")
    acc_16k = str(base / f"{key}_acc_16k.wav")
    _resample_to_16k_mono(settings.ffmpeg_bin, str(vocals_stable if vocals_stable.exists() else vocals), vocals_16k)
    _resample_to_16k_mono(settings.ffmpeg_bin, str(acc), acc_16k)
    vocals_vad = _vad_ratio(vocals_16k)
    acc_vad = _vad_ratio(acc_16k)
    vocals_lufs = analyze_loudness(settings.ffmpeg_bin, str(vocals_stable if vocals_stable.exists() else vocals))
    acc_lufs = analyze_loudness(settings.ffmpeg_bin, str(acc))

    report = StemsReport(
        method="demucs",
        input_video=input_video,
        input_audio=audio_wav,
        vocals_wav=str(vocals_stable if vocals_stable.exists() else vocals),
        acc_wav=str(acc),
        vocals_vad=vocals_vad,
        acc_vad=acc_vad,
        vocals_lufs=vocals_lufs,
        acc_lufs=acc_lufs,
        params={"model": model, "cpu": use_cpu, "post": "auto_smooth_mid" if auto_tune_bed else "smooth_mid"},
    )
    report_path = str(base / f"{key}_stems_report.json")
    Path(report_path).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    # Cleanup heavy demucs folders if not requested to keep
    if not keep_intermediate:
        for p in base.glob("demucs_*/"):
            try:
                shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass
    return str(vocals_stable if vocals_stable.exists() else vocals), str(acc), report_path


def build_stems_ms(
    settings: Settings,
    input_video: str,
    out_dir: Optional[str] = None,
    mlev: float = 0.5,
    blend: float = 0.2,
    force: bool = False,
) -> Tuple[str, str, str]:
    """Mid/Side-based fast stems: returns (vocals_wav, acc_wav, report_path).

    - vocals: center extract (mono), bandpass, denoise, de-esser
    - acc: use make_music_bed with given mlev/blend
    """
    from ..utils.ffmpeg import make_music_bed

    base = Path(out_dir or settings.tmp_dir)
    base.mkdir(parents=True, exist_ok=True)
    key = Path(input_video).stem
    audio_wav = str(base / f"{key}_audio.wav")
    if not Path(audio_wav).exists() or force:
        from ..utils.ffmpeg import extract_audio

        extract_audio(settings.ffmpeg_bin, input_video, audio_wav)

    # vocals
    vocals_wav = str(base / f"{key}_vocals.wav")
    if not Path(vocals_wav).exists() or force:
        # center (mid) extraction -> mono; bandpass 120..8000; denoise + de-esser
        fc = "pan=mono|c0=0.5*FL+0.5*FR,highpass=f=120,lowpass=f=8000,afftdn,deesser,alimiter"
        cmd = [
            settings.ffmpeg_bin,
            "-y",
            "-i",
            audio_wav,
            "-af",
            fc,
            "-ac",
            "1",
            "-ar",
            "44100",
            "-acodec",
            "pcm_s16le",
            vocals_wav,
        ]
        run_ffmpeg(cmd)

    # accompaniment (bed)
    acc_wav = str(base / f"{key}_acc.wav")
    if not Path(acc_wav).exists() or force:
        make_music_bed(settings.ffmpeg_bin, audio_wav, acc_wav, mlev=mlev, base_volume=blend)

    # Metrics
    vocals_16k = str(base / f"{key}_vocals_16k.wav")
    acc_16k = str(base / f"{key}_acc_16k.wav")
    _resample_to_16k_mono(settings.ffmpeg_bin, vocals_wav, vocals_16k)
    _resample_to_16k_mono(settings.ffmpeg_bin, acc_wav, acc_16k)
    vocals_vad = _vad_ratio(vocals_16k)
    acc_vad = _vad_ratio(acc_16k)
    vocals_lufs = analyze_loudness(settings.ffmpeg_bin, vocals_wav)
    acc_lufs = analyze_loudness(settings.ffmpeg_bin, acc_wav)

    report = StemsReport(
        method="ms",
        input_video=input_video,
        input_audio=audio_wav,
        vocals_wav=vocals_wav,
        acc_wav=acc_wav,
        vocals_vad=vocals_vad,
        acc_vad=acc_vad,
        vocals_lufs=vocals_lufs,
        acc_lufs=acc_lufs,
        params={"mlev": mlev, "blend": blend},
    )
    report_path = str(base / f"{key}_stems_report.json")
    Path(report_path).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return vocals_wav, acc_wav, report_path
