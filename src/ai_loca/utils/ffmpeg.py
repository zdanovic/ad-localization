from __future__ import annotations

import hashlib
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

from ..errors import FFmpegError
from ..ffmpeg_presets import DEFAULT_AUDIO, DEFAULT_VIDEO

logger = logging.getLogger(__name__)


def _escape_drawtext(text: str) -> str:
    # Escape characters significant for ffmpeg drawtext
    # Replace backslash first to avoid double-escaping
    text = text.replace("\\", "\\\\")
    text = text.replace(":", r"\:")
    text = text.replace("'", r"\'")
    text = text.replace(",", r"\,")
    return text


def run_ffmpeg(args: List[str]) -> None:
    if not args:
        raise FFmpegError("Empty ffmpeg arguments")
    # Log safe, shell-escaped command
    cmd_str = " ".join(shlex.quote(a) for a in args)
    logger.info("FFmpeg: %s", cmd_str)
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logger.error("FFmpeg failed (%s): %s", result.returncode, result.stderr.decode(errors="ignore")[:2000])
        raise FFmpegError(f"FFmpeg failed with code {result.returncode}")


def ffmpeg_version(ffmpeg_bin: str) -> str:
    try:
        proc = subprocess.run([ffmpeg_bin, "-version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise FFmpegError("ffmpeg not available")
        first_line = proc.stdout.decode(errors="ignore").splitlines()[0]
        return first_line.strip()
    except FileNotFoundError as e:  # noqa: PERF203
        raise FFmpegError("ffmpeg binary not found") from e


def extract_audio(ffmpeg_bin: str, video_in: str, audio_out: str, sample_rate: int = 44100) -> str:
    Path(audio_out).parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg_bin,
        "-y",
        "-i",
        video_in,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        audio_out,
    ]
    run_ffmpeg(args)
    return audio_out


def mux_audio_video(
    ffmpeg_bin: str,
    video_in: str,
    audio_in: str,
    output_path: str,
    reencode: bool = False,
) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if reencode:
        args = [
            ffmpeg_bin,
            "-y",
            "-i",
            video_in,
            "-i",
            audio_in,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v",
            DEFAULT_VIDEO.codec,
            "-preset",
            DEFAULT_VIDEO.preset,
            "-crf",
            str(DEFAULT_VIDEO.crf),
            "-c:a",
            DEFAULT_AUDIO.codec,
            "-b:a",
            DEFAULT_AUDIO.bitrate,
            "-shortest",
            output_path,
        ]
    else:
        args = [
            ffmpeg_bin,
            "-y",
            "-i",
            video_in,
            "-i",
            audio_in,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            DEFAULT_AUDIO.codec,
            "-b:a",
            DEFAULT_AUDIO.bitrate,
            "-shortest",
            output_path,
        ]
    run_ffmpeg(args)
    return output_path


def build_drawtext_filter(
    text: str,
    fontfile: Optional[str] = None,
    fontcolor: str = "white",
    fontsize: int = 24,
    x: str = "(w-text_w)/2",
    y: str = "(h-text_h)-20",
    box: bool = True,
    boxcolor: str = "black@0.5",
) -> str:
    esc_text = _escape_drawtext(text)
    parts = [f"text='{esc_text}'", f"fontcolor={fontcolor}", f"fontsize={fontsize}", f"x={x}", f"y={y}"]
    if fontfile:
        parts.append(f"fontfile='{fontfile}'")
    if box:
        parts.append(f"box=1")
        parts.append(f"boxcolor={boxcolor}")
    return "drawtext=" + ":".join(parts)


def render_text_overlay(
    ffmpeg_bin: str,
    video_in: str,
    output_path: str,
    drawtext_filter: str,
) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    vf = drawtext_filter
    args = [
        ffmpeg_bin,
        "-y",
        "-i",
        video_in,
        "-vf",
        vf,
        "-c:v",
        DEFAULT_VIDEO.codec,
        "-preset",
        DEFAULT_VIDEO.preset,
        "-crf",
        str(DEFAULT_VIDEO.crf),
        "-c:a",
        "copy",
        output_path,
    ]
    run_ffmpeg(args)
    return output_path


def overlay_image(
    ffmpeg_bin: str,
    video_in: str,
    png_path: str,
    output_path: str,
    x: str = "10",
    y: str = "10",
) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg_bin,
        "-y",
        "-i",
        video_in,
        "-i",
        png_path,
        "-filter_complex",
        f"[0:v][1:v]overlay={x}:{y}",
        "-c:a",
        "copy",
        output_path,
    ]
    run_ffmpeg(args)
    return output_path


def burn_subtitles(
    ffmpeg_bin: str,
    video_in: str,
    srt_path: str,
    output_path: str,
    fontsdir: Optional[str] = None,
) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    vf = f"subtitles='{srt_path}'"
    if fontsdir:
        vf += f":fontsdir='{fontsdir}'"
    args = [
        ffmpeg_bin,
        "-y",
        "-i",
        video_in,
        "-vf",
        vf,
        "-c:v",
        DEFAULT_VIDEO.codec,
        "-preset",
        DEFAULT_VIDEO.preset,
        "-crf",
        str(DEFAULT_VIDEO.crf),
        "-c:a",
        "copy",
        output_path,
    ]
    run_ffmpeg(args)
    return output_path


def deterministic_artifact(base_dir: str, key: str, suffix: str) -> str:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    path = Path(base_dir) / h
    path.mkdir(parents=True, exist_ok=True)
    return str(path / suffix)


def normalize_wav(ffmpeg_bin: str, in_path: str, out_path: str, sample_rate: int = DEFAULT_AUDIO.sample_rate) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg_bin,
        "-y",
        "-i",
        in_path,
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        out_path,
    ]
    run_ffmpeg(args)
    return out_path


def concat_wavs(ffmpeg_bin: str, inputs: List[str], out_path: str) -> str:
    if not inputs:
        raise FFmpegError("No input wavs to concat")
    # Create a temporary file list for concat demuxer
    list_path = Path(out_path).with_suffix(".txt")
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with list_path.open("w", encoding="utf-8") as f:
        for p in inputs:
            absp = str(Path(p).resolve())
            f.write(f"file '{absp}'\n")
    args = [
        ffmpeg_bin,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        out_path,
    ]
    run_ffmpeg(args)
    return out_path


def concat_wavs_crossfade(ffmpeg_bin: str, inputs: List[str], out_path: str, xfade_ms: int = 60) -> str:
    """Concatenate WAVs with short acrossfades to reduce audible cuts.

    Uses iterative acrossfade: (((a xfade b) xfade c) ...).
    Safe default triangle curve. Keep xfade_ms small (30-80ms) to avoid timing drift.
    """
    if not inputs:
        raise FFmpegError("No input wavs to concat")
    if len(inputs) == 1:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(Path(inputs[0]).read_bytes())
        return out_path
    dur = max(0.005, xfade_ms / 1000.0)
    cur = inputs[0]
    for idx in range(1, len(inputs)):
        nxt = inputs[idx]
        tmp = Path(out_path).with_suffix("")
        tmp_i = str(tmp) + f"_xf{idx}.wav"
        args = [
            ffmpeg_bin,
            "-y",
            "-i",
            cur,
            "-i",
            nxt,
            "-filter_complex",
            f"[0:a][1:a]acrossfade=d={dur}:c1=tri:c2=tri[mix]",
            "-map",
            "[mix]",
            "-acodec",
            "pcm_s16le",
            tmp_i,
        ]
        run_ffmpeg(args)
        cur = tmp_i
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(Path(cur).read_bytes())
    return out_path


def fade_wav(ffmpeg_bin: str, in_wav: str, out_wav: str, fade_ms: int = 15) -> str:
    """Apply short fade-in and fade-out to reduce clicks at segment edges."""
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    fin = max(0.001, fade_ms / 1000.0)
    args = [
        ffmpeg_bin,
        "-y",
        "-i",
        in_wav,
        "-af",
        f"afade=t=in:st=0:d={fin},areverse,afade=t=in:st=0:d={fin},areverse",
        "-acodec",
        "pcm_s16le",
        out_wav,
    ]
    run_ffmpeg(args)
    return out_wav


def trim_wav(ffmpeg_bin: str, in_path: str, out_path: str, duration_sec: float = 20.0) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg_bin,
        "-y",
        "-t",
        str(duration_sec),
        "-i",
        in_path,
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        out_path,
    ]
    run_ffmpeg(args)
    return out_path


def make_music_bed(ffmpeg_bin: str, in_audio: str, out_wav: str, mlev: float = 0.6, base_volume: float = 0.35) -> str:
    """Create a background bed preserving authenticity while reducing vocals.

    Strategy:
    - Convert to Mid/Side; attenuate Mid (mlev≈0.6) instead of hard center-cancel to keep bass/kick.
    - Light bandpass + spectral denoise to clean residuals.
    - Blend a little of the original back (≈0.35) to preserve ambience.
    """
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    filter_complex = (
        f"[0:a]asplit=2[a][b];"
        f"[a]stereotools=mlev={mlev}:slev=1.0,highpass=f=80,lowpass=f=18000,afftdn,alimiter[proc];"
        f"[b]volume={base_volume}[base];"
        f"[proc][base]amix=inputs=2:normalize=0[bed]"
    )
    args = [
        ffmpeg_bin,
        "-y",
        "-i",
        in_audio,
        "-filter_complex",
        filter_complex,
        "-map",
        "[bed]",
        "-ac",
        "2",
        "-ar",
        str(DEFAULT_AUDIO.sample_rate),
        "-acodec",
        "pcm_s16le",
        out_wav,
    ]
    try:
        run_ffmpeg(args)
        return out_wav
    except FFmpegError:
        # Fallback: denoise + mild EQ + limiter
        args_fb = [
            ffmpeg_bin,
            "-y",
            "-i",
            in_audio,
            "-af",
            "highpass=f=80,lowpass=f=18000,afftdn,alimiter",
            "-ac",
            "2",
            "-ar",
            str(DEFAULT_AUDIO.sample_rate),
            "-acodec",
            "pcm_s16le",
            out_wav,
        ]
        run_ffmpeg(args_fb)
        return out_wav


def duck_mix_tts_with_bed(
    ffmpeg_bin: str,
    tts_wav: str,
    bed_wav: str,
    out_wav: str,
    voice_lufs: float = -18.0,
    bed_lufs: float = -24.0,
) -> str:
    """Sidechain-compress music bed with TTS voice and mix into a single WAV.

    - Normalizes TTS and bed to target LUFS
    - Applies sidechain compression to duck bed under voice
    - Mixes voice + bed, producing out_wav (PCM s16le)
    """
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    # Multiband sidechain gating: only reduce vocal band in the bed during speech
    filter_complex = (
        f"[0:a]loudnorm=I={voice_lufs}:TP=-1.5:LRA=11[vox];"
        f"[1:a]loudnorm=I={bed_lufs}:TP=-2:LRA=11[bedn];"
        f"[bedn]asplit=3[bl][bm][bh];"
        f"[bl]lowpass=f=200[low];"
        f"[bm]highpass=f=200,lowpass=f=5000[mid];"
        f"[bh]highpass=f=5000[high];"
        f"[mid][vox]sidechaingate=threshold=0.015:ratio=20:attack=10:release=120[midg];"
        f"[low][midg][high]amix=inputs=3:normalize=0[bedmb];"
        f"[bedmb][vox]amix=inputs=2:normalize=0[mix]"
    )
    args = [
        ffmpeg_bin,
        "-y",
        "-i",
        tts_wav,
        "-i",
        bed_wav,
        "-filter_complex",
        filter_complex,
        "-map",
        "[mix]",
        "-acodec",
        "pcm_s16le",
        out_wav,
    ]
    run_ffmpeg(args)
    return out_wav


def analyze_loudness(ffmpeg_bin: str, in_audio: str) -> float:
    """Return integrated loudness (LUFS) measured by loudnorm.

    If analysis fails, default to -20.0 LUFS.
    """
    import json
    import subprocess

    cmd = [
        ffmpeg_bin,
        "-i",
        in_audio,
        "-af",
        "loudnorm=I=-23:TP=-1.5:LRA=11:print_format=json",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        txt = proc.stderr.decode(errors="ignore")
        start = txt.find("{\n    \"input_i\"")
        if start == -1:
            return -20.0
        end = txt.find("}\n", start)
        blob = txt[start : end + 2]
        data = json.loads(blob)
        # prefer measured_I if present
        lufs = float(data.get("measured_I") or data.get("input_i") or -20.0)
        return lufs
    except Exception:
        return -20.0


def gate_bed_with_vocals(
    ffmpeg_bin: str,
    bed_wav: str,
    vocals_wav: str,
    out_wav: str,
    threshold: float = 0.015,
    ratio: float = 20.0,
    attack: int = 10,
    release: int = 120,
) -> str:
    """Use vocals as sidechain to gate mid band of bed, reducing residual speech.

    Splits bed into low/mid/high; gates only the mid band with vocals sidechain; then re-mixes bands.
    """
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    fc = (
        f"[0:a]asplit=3[bl][bm][bh];"
        f"[bl]lowpass=f=200[low];"
        f"[bm]highpass=f=200,lowpass=f=5000[mid];"
        f"[bh]highpass=f=5000[high];"
        f"[mid][1:a]sidechaingate=threshold={threshold}:ratio={ratio}:attack={attack}:release={release}[midg];"
        f"[low][midg][high]amix=inputs=3:normalize=0[mix]"
    )
    args = [
        ffmpeg_bin,
        "-y",
        "-i",
        bed_wav,
        "-i",
        vocals_wav,
        "-filter_complex",
        fc,
        "-map",
        "[mix]",
        "-acodec",
        "pcm_s16le",
        out_wav,
    ]
    run_ffmpeg(args)
    return out_wav


def smooth_bed_with_vocals(
    ffmpeg_bin: str,
    bed_wav: str,
    vocals_wav: str,
    out_wav: str,
    target_lufs: float = -20.0,
    threshold: float = 0.03,
    ratio: float = 4.0,
    attack: int = 10,
    release: int = 200,
    makeup: float = 1.5,
) -> str:
    """Sidechain-compress only the mid band vs vocals and restore perceived loudness.

    This reduces residual speech but keeps energy more natural than hard gating.
    """
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    fc = (
        f"[0:a]asplit=3[bl][bm][bh];"
        f"[bl]lowpass=f=200[low];"
        f"[bm]highpass=f=200,lowpass=f=5000[mid];"
        f"[bh]highpass=f=5000[high];"
        f"[mid][1:a]sidechaincompress=threshold={threshold}:ratio={ratio}:attack={attack}:release={release}:makeup={makeup}[midc];"
        f"[low][midc][high]amix=inputs=3:normalize=0[mb];"
        f"[mb]loudnorm=I={target_lufs}:TP=-1.5:LRA=11[mix]"
    )
    args = [
        ffmpeg_bin,
        "-y",
        "-i",
        bed_wav,
        "-i",
        vocals_wav,
        "-filter_complex",
        fc,
        "-map",
        "[mix]",
        "-acodec",
        "pcm_s16le",
        out_wav,
    ]
    run_ffmpeg(args)
    return out_wav


def refill_bed_with_original(
    ffmpeg_bin: str,
    acc_wav: str,
    original_wav: str,
    vocals_wav: str,
    out_wav: str,
    target_lufs: float = -20.0,
    sc_threshold: float = 0.02,
    sc_ratio: float = 20.0,
    attack: int = 10,
    release: int = 200,
    acc_gain: float = 1.0,
    orig_gain: float = 1.0,
) -> str:
    """Refill accompaniment by mixing back the original where no vocals are present.

    - Sidechain-compress the original using vocals as key, so original is strongly reduced when speech is present,
      but remains in non-speech regions to preserve ambience/transients.
    - Then mix this processed original with the acc, and loudness-normalize to target.
    """
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    fc = (
        f"[1:a]volume={orig_gain}[orig];"
        f"[0:a]volume={acc_gain}[acc];"
        f"[orig][2:a]sidechaincompress=threshold={sc_threshold}:ratio={sc_ratio}:attack={attack}:release={release}:makeup=1.0[origduck];"
        f"[acc][origduck]amix=inputs=2:normalize=0[mix0];"
        f"[mix0]loudnorm=I={target_lufs}:TP=-1.5:LRA=11[mix]"
    )
    args = [
        ffmpeg_bin,
        "-y",
        "-i",
        acc_wav,
        "-i",
        original_wav,
        "-i",
        vocals_wav,
        "-filter_complex",
        fc,
        "-map",
        "[mix]",
        "-acodec",
        "pcm_s16le",
        out_wav,
    ]
    run_ffmpeg(args)
    return out_wav


## The following helper functions were used for validation tooling (A/B compares).
## They were intentionally removed from the production path to keep the library lean.


def multiband_refill_bed(
    ffmpeg_bin: str,
    acc_wav: str,
    original_wav: str,
    vocals_wav: str,
    out_wav: str,
    target_lufs: float = -20.0,
) -> str:
    """Frequency-dependent refill: low/mid/high bands sidechain original by vocals with band-specific strengths, mix with acc.

    - Low band: light compression to keep kick/bass (threshold 0.03, ratio 2)
    - Mid band: stronger compression to suppress speech band (threshold 0.02, ratio 8)
    - High band: light/moderate to preserve cymbals/air (threshold 0.03, ratio 3)
    - Final loudnorm to target LUFS.
    """
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    fc = (
        # Split original into bands
        f"[1:a]asplit=3[ol][om][oh];"
        f"[ol]lowpass=f=180[olb];"
        f"[om]highpass=f=180,lowpass=f=5000[omb];"
        f"[oh]highpass=f=5000[ohb];"
        # Split acc for level control
        f"[0:a]volume=1.0[acc];"
        # Sidechain compress each band by vocals
        f"[olb][2:a]sidechaincompress=threshold=0.03:ratio=2.0:attack=10:release=180:makeup=1.0[olc];"
        f"[omb][2:a]sidechaincompress=threshold=0.02:ratio=8.0:attack=10:release=220:makeup=1.0[omc];"
        f"[ohb][2:a]sidechaincompress=threshold=0.03:ratio=3.0:attack=5:release=150:makeup=1.0[ohc];"
        # Merge bands back
        f"[olc][omc][ohc]amix=inputs=3:normalize=0[origduck];"
        # Mix with acc and normalize loudness
        f"[acc][origduck]amix=inputs=2:normalize=0[mix0];"
        f"[mix0]loudnorm=I={target_lufs}:TP=-1.5:LRA=11[mix]"
    )
    args = [
        ffmpeg_bin,
        "-y",
        "-i",
        acc_wav,
        "-i",
        original_wav,
        "-i",
        vocals_wav,
        "-filter_complex",
        fc,
        "-map",
        "[mix]",
        "-acodec",
        "pcm_s16le",
        out_wav,
    ]
    run_ffmpeg(args)
    return out_wav


def stabilize_bed_dynaudnorm(ffmpeg_bin: str, in_wav: str, out_wav: str) -> str:
    """Apply gentle dynamic normalization to reduce short-term dips without pumping.

    Uses dynaudnorm with conservative settings.
    """
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg_bin,
        "-y",
        "-i",
        in_wav,
        "-af",
        "dynaudnorm=p=0.7:m=9:s=12",
        out_wav,
    ]
    run_ffmpeg(args)
    return out_wav


def get_audio_duration(ffprobe_bin: str, in_audio: str) -> float:
    import subprocess

    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        in_audio,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        return float(proc.stdout.decode().strip())
    except Exception:
        return 0.0


def time_stretch_wav(ffmpeg_bin: str, in_wav: str, out_wav: str, factor: float) -> str:
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    # Clamp atempo to 0.5..2.0; if outside, multiple stages.
    def atempo_chain(f: float) -> list[str]:
        chain: list[str] = []
        remaining = f
        while remaining < 0.5 or remaining > 2.0:
            if remaining < 0.5:
                chain.append("atempo=0.5")
                remaining /= 0.5
            else:
                chain.append("atempo=2.0")
                remaining /= 2.0
        chain.append(f"atempo={remaining}")
        return chain

    af = ",".join(atempo_chain(max(0.25, min(4.0, factor))))
    args = [
        ffmpeg_bin,
        "-y",
        "-i",
        in_wav,
        "-af",
        af,
        "-acodec",
        "pcm_s16le",
        out_wav,
    ]
    run_ffmpeg(args)
    return out_wav


def polish_voice(ffmpeg_bin: str, in_wav: str, out_wav: str) -> str:
    """Light voice processing: HPF, de-esser, gentle limiter."""
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    af = "highpass=f=80,deesser,alimiter"
    args = [
        ffmpeg_bin,
        "-y",
        "-i",
        in_wav,
        "-af",
        af,
        "-acodec",
        "pcm_s16le",
        out_wav,
    ]
    run_ffmpeg(args)
    return out_wav


def generate_test_video(
    ffmpeg_bin: str,
    out_path: str,
    duration: float = 2.0,
    size: str = "640x360",
    framerate: int = 30,
    with_tone: bool = True,
) -> str:
    """Generate a short test video with color background and optional tone."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if with_tone:
        args = [
            ffmpeg_bin,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={size}:r={framerate}:d={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=1000:sample_rate={DEFAULT_AUDIO.sample_rate}:duration={duration}",
            "-c:v",
            DEFAULT_VIDEO.codec,
            "-preset",
            DEFAULT_VIDEO.preset,
            "-crf",
            str(DEFAULT_VIDEO.crf),
            "-c:a",
            DEFAULT_AUDIO.codec,
            "-b:a",
            DEFAULT_AUDIO.bitrate,
            "-shortest",
            out_path,
        ]
    else:
        args = [
            ffmpeg_bin,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={size}:r={framerate}:d={duration}",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={DEFAULT_AUDIO.sample_rate}:cl=mono",
            "-c:v",
            DEFAULT_VIDEO.codec,
            "-preset",
            DEFAULT_VIDEO.preset,
            "-crf",
            str(DEFAULT_VIDEO.crf),
            "-c:a",
            DEFAULT_AUDIO.codec,
            "-b:a",
            DEFAULT_AUDIO.bitrate,
            "-shortest",
            out_path,
        ]
    run_ffmpeg(args)
    return out_path


def generate_solid_png(ffmpeg_bin: str, out_path: str, size: str = "200x80", color: str = "white") -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg_bin,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={size}",
        "-frames:v",
        "1",
        out_path,
    ]
    run_ffmpeg(args)
    return out_path


def generate_silence_wav(ffmpeg_bin: str, out_path: str, duration: float, sample_rate: int = DEFAULT_AUDIO.sample_rate) -> str:
    """Generate a silent mono WAV of given duration and sample rate."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.0, float(duration))
    args = [
        ffmpeg_bin,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={sample_rate}:cl=mono",
        "-t",
        str(dur),
        "-acodec",
        "pcm_s16le",
        out_path,
    ]
    run_ffmpeg(args)
    return out_path
