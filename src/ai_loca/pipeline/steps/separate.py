from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import json

from ...config import Settings
from ...audio.stems import build_stems_demucs


@dataclass
class SeparationResult:
    input_video: str
    out_dir: str
    vocals_wav: str
    bed_wav: str
    report_path: str
    model: str
    cpu: bool


def _sanitize(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def run(
    settings: Settings,
    input_video: str,
    out_dir: str,
    model: str = "htdemucs_ft",
    use_cpu: bool = True,
    force: bool = False,
    keep_intermediate: bool = False,
) -> SeparationResult:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    video_dir = out_root / _sanitize(Path(input_video).stem)
    video_dir.mkdir(parents=True, exist_ok=True)
    # Respect lock file to prevent accidental overwrite when vibecoding next stages
    lock_file = video_dir / ".lock"
    if lock_file.exists() and not force:
        std_vocals = video_dir / "vocals.wav"
        std_bed = video_dir / "bed.wav"
        if not std_vocals.exists() or not std_bed.exists():
            raise RuntimeError(f"Stage locked but artifacts missing in {video_dir}")
        rep = str(video_dir / "separation.json")
        return SeparationResult(
            input_video=input_video,
            out_dir=str(video_dir),
            vocals_wav=str(std_vocals),
            bed_wav=str(std_bed),
            report_path=rep,
            model=model,
            cpu=use_cpu,
        )
    vocals, bed, rep = build_stems_demucs(
        settings,
        input_video=input_video,
        out_dir=str(video_dir),
        model=model,
        use_cpu=use_cpu,
        auto_tune_bed=True,
        force=force,
        keep_intermediate=keep_intermediate,
    )
    # Standardize artifact names inside video_dir
    std_vocals = video_dir / "vocals.wav"
    std_bed = video_dir / "bed.wav"
    Path(std_vocals).write_bytes(Path(vocals).read_bytes())
    Path(std_bed).write_bytes(Path(bed).read_bytes())

    res = SeparationResult(
        input_video=input_video,
        out_dir=str(video_dir),
        vocals_wav=str(std_vocals),
        bed_wav=str(std_bed),
        report_path=rep,
        model=model,
        cpu=use_cpu,
    )
    Path(video_dir / "separation.json").write_text(json.dumps(asdict(res), indent=2), encoding="utf-8")
    return res
