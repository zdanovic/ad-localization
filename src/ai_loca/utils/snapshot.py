from __future__ import annotations

import hashlib
import json
import os
import platform
import tarfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from ..config import Settings
from .ffmpeg import ffmpeg_version


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pip_freeze() -> List[str]:
    try:
        import subprocess

        proc = subprocess.run([os.fspath(Path(os.sys.executable)), "-m", "pip", "freeze"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode == 0:
            return [line.strip() for line in proc.stdout.decode(errors="ignore").splitlines() if line.strip()]
    except Exception:
        pass
    return []


def _git_info() -> Dict[str, str]:
    try:
        import subprocess

        sha = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        status = subprocess.run(["git", "status", "--porcelain"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return {
            "commit": sha.stdout.decode().strip() if sha.returncode == 0 else "",
            "dirty": "yes" if status.returncode == 0 and status.stdout.decode().strip() else "no",
            "branch": branch.stdout.decode().strip() if branch.returncode == 0 else "",
        }
    except Exception:
        return {"commit": "", "dirty": "", "branch": ""}


def _safe_settings_snapshot(settings: Settings) -> Dict[str, str]:
    # Only non-secret, reproducibility-relevant fields
    return {
        "stt_model": settings.stt_model,
        "mt_model": settings.mt_model,
        "tts_model": settings.tts_model,
        "stt_provider": settings.stt_provider,
        "mt_provider": settings.mt_provider,
        "tts_provider": settings.tts_provider,
        "ffmpeg_bin": settings.ffmpeg_bin,
        "tmp_dir": settings.tmp_dir,
        "http_timeout": str(settings.http_timeout),
        "http_retries": str(settings.http_retries),
        "tts_model_openai": getattr(settings, "tts_model_openai", ""),
    }


def _walk_files(root: Path) -> List[Tuple[str, int, str]]:
    out: List[Tuple[str, int, str]] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            try:
                rel = p.relative_to(root).as_posix()
            except Exception:
                continue
            out.append((rel, p.stat().st_size, _sha256(p)))
    return out


def create_snapshot(settings: Settings, stage_path: str, name: str, out_dir: str) -> Tuple[str, str]:
    root = Path(stage_path)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Stage path not found: {stage_path}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{ts}_{name}"
    tar_path = Path(out_dir) / f"{base_name}.tar.gz"
    meta_path = Path(out_dir) / f"{base_name}.metadata.json"

    # Collect metadata
    meta: Dict[str, object] = {
        "name": name,
        "created": ts,
        "stage_path": os.fspath(root.resolve()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ffmpeg": "",
        "git": _git_info(),
        "settings": _safe_settings_snapshot(settings),
        "pip_freeze": _pip_freeze(),
        "files": [],
    }
    try:
        meta["ffmpeg"] = ffmpeg_version(settings.ffmpeg_bin)
    except Exception:
        meta["ffmpeg"] = ""

    files = _walk_files(root)
    meta["files"] = [{"path": f, "size": s, "sha256": h} for (f, s, h) in files]

    # Write tar.gz with the directory contents and an embedded SNAPSHOT.json
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(root, arcname="stage")
        # Embed snapshot metadata inside the archive as SNAPSHOT.json
        tmp_meta = meta_path.with_suffix("")
        tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        try:
            tar.add(tmp_meta, arcname="SNAPSHOT.json")
        finally:
            try:
                tmp_meta.unlink()
            except Exception:
                pass

    # Also write a sidecar metadata json for quick inspection
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return os.fspath(tar_path), os.fspath(meta_path)


def restore_snapshot(tar_file: str, dest_dir: str) -> str:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_file, "r:gz") as tar:
        tar.extractall(dest)
    return os.fspath(dest)

