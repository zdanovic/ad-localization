from __future__ import annotations

import argparse
import logging
from pathlib import Path
import httpx

from .config import Settings
from .logging import setup_logging
from .pipeline.orchestrator import Orchestrator
from .utils.ffmpeg import (
    build_drawtext_filter,
    render_text_overlay,
    overlay_image,
    extract_audio,
    burn_subtitles,
    ffmpeg_version,
    generate_test_video,
    generate_solid_png,
)
from .adapters import HFWhisperSTTAdapter, HFTranslationAdapter, HFTTSAdapter
from .dto import TranslationRequest, TTSRequest
from .subtitles.processor import transcript_to_subs, export_srt


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="AI Video Ad Localization CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # localize command
    p_loc = sub.add_parser("localize", help="Run full localization pipeline")
    p_loc.add_argument("--input-video", required=True)
    p_loc.add_argument("--source-lang", required=False)
    p_loc.add_argument("--target-lang", required=True)
    p_loc.add_argument("--output-video", required=True)
    p_loc.add_argument("--overlay-text")
    p_loc.add_argument("--overlay-png")
    p_loc.add_argument("--tts-voice")
    p_loc.add_argument("--tts-language")
    p_loc.add_argument("--tts-speaker-wav")
    p_loc.add_argument("--tts-segmented", action="store_true", help="Synthesize TTS per subtitle segment and concat")
    p_loc.add_argument("--burn-subtitles", action="store_true")
    p_loc.add_argument("--force", action="store_true")
    p_loc.add_argument("--input-subs", help="Path to input subtitles (SRT/VTT) to skip STT")
    p_loc.add_argument("--subs-only", action="store_true", help="Skip TTS and only translate/burn subtitles")

    # stt
    p_stt = sub.add_parser("stt", help="Transcribe audio using HF Whisper")
    p_stt.add_argument("--audio", required=True)
    p_stt.add_argument("--language")

    # mt
    p_mt = sub.add_parser("mt", help="Translate text via HF MT")
    p_mt.add_argument("--text", required=True)
    p_mt.add_argument("--source-lang", required=True)
    p_mt.add_argument("--target-lang", required=True)

    # tts
    p_tts = sub.add_parser("tts", help="Synthesize speech via HF TTS")
    p_tts.add_argument("--text", required=True)
    p_tts.add_argument("--voice")
    p_tts.add_argument("--language")
    p_tts.add_argument("--speaker-wav")
    p_tts.add_argument("--out", required=True)

    # overlay
    p_txt = sub.add_parser("overlay-text", help="Render drawtext overlay")
    p_txt.add_argument("--input-video", required=True)
    p_txt.add_argument("--text", required=True)
    p_txt.add_argument("--output-video", required=True)
    p_txt.add_argument("--fontfile")

    p_png = sub.add_parser("overlay-png", help="Overlay PNG on video")
    p_png.add_argument("--input-video", required=True)
    p_png.add_argument("--png", required=True)
    p_png.add_argument("--output-video", required=True)
    p_png.add_argument("--x", default="10")
    p_png.add_argument("--y", default="10")

    # doctor
    p_doc = sub.add_parser("doctor", help="Environment checks and readiness report")
    p_doc.add_argument("--check-hf", action="store_true", help="Try contacting HF Inference API or custom endpoints")

    # smoke
    p_smoke = sub.add_parser("smoke", help="Offline smoke test using generated media")
    p_smoke.add_argument("--out", required=True, help="Output video path for smoke test")
    p_smoke.add_argument("--with-png", action="store_true", help="Also overlay a generated PNG logo")
    p_smoke.add_argument("--text", default="Sample Overlay", help="Overlay text for drawtext test")

    # snapshots
    p_snap = sub.add_parser("snapshot", help="Create a tar.gz snapshot of a stage directory with metadata")
    p_snap.add_argument("--path", default=".ai_loca_stage_separate", help="Directory to snapshot")
    p_snap.add_argument("--name", default="separate", help="Snapshot name label")
    p_snap.add_argument("--out-dir", default="output/snapshots", help="Where to place the snapshot tar.gz")

    p_restore = sub.add_parser("restore-snapshot", help="Restore a snapshot tar.gz into a directory")
    p_restore.add_argument("--file", required=True, help="Path to snapshot tar.gz")
    p_restore.add_argument("--dest", default=".", help="Destination directory to extract to")

    # lock/unlock stage dirs
    p_lock = sub.add_parser(
        "lock-stage",
        help="Create or remove .lock files in stage subdirs to prevent overwrite",
    )
    p_lock.add_argument("--dir", default=".ai_loca_stage_separate", help="Stage root directory")
    p_lock.add_argument("--unlock", action="store_true", help="Remove locks instead of creating them")

    # pipeline: separate audio (stage)
    p_sep = sub.add_parser("separate", help="Pipeline step: separate vocals and bed (Demucs)")
    p_sep.add_argument("--input-video", nargs="+", help="Input video files")
    p_sep.add_argument("--input-dir", help="Directory with videos to sample from")
    p_sep.add_argument("--count", type=int, default=0, help="Sample N random videos from --input-dir")
    p_sep.add_argument("--out-dir", default=".ai_loca_stage_separate", help="Output directory for stage artifacts")
    p_sep.add_argument("--model", default="htdemucs_ft", help="Demucs model name")
    p_sep.add_argument("--cpu", action="store_true")
    p_sep.add_argument("--force", action="store_true")

    # (validation pack command removed for production simplification)

    args = parser.parse_args()
    settings = Settings.from_env()
    log = logging.getLogger("cli")

    try:
        if args.cmd == "localize":
            orch = Orchestrator(settings)
            out = orch.localize_video(
                input_video=args.input_video,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                output_video=args.output_video,
                overlay_text=args.overlay_text,
                overlay_png=args.overlay_png,
                burn_subs=args.burn_subtitles,
                force=args.force,
                tts_voice=args.tts_voice,
                input_subs=args.input_subs,
                subs_only=args.subs_only,
                tts_language=args.tts_language,
                tts_speaker_wav=args.tts_speaker_wav,
                tts_segmented=args.tts_segmented,
            )
            print(out)
        elif args.cmd == "stt":
            if settings.stt_provider == "openai":
                from .adapters import OpenAIWhisperSTTAdapter

                stt = OpenAIWhisperSTTAdapter(
                    api_key=settings.openai_api_key,
                    model="whisper-1",
                    base_url=settings.openai_base_url,
                    timeout=settings.http_timeout,
                )
            else:
                stt = HFWhisperSTTAdapter(
                    settings.hf_api_token,
                    settings.stt_model,
                    settings.stt_endpoint,
                    settings.http_timeout,
                    settings.http_retries,
                )
            tr = stt.transcribe(args.audio, language=args.language)
            print(tr.model_dump_json(indent=2))
        elif args.cmd == "mt":
            mt = HFTranslationAdapter(
                settings.hf_api_token,
                settings.mt_model,
                settings.mt_endpoint,
                settings.http_timeout,
                settings.http_retries,
            )
            res = mt.translate(
                TranslationRequest(source_lang=args.source_lang, target_lang=args.target_lang, text=args.text)
            )
            print(res.text)
        elif args.cmd == "tts":
            if settings.tts_provider == "elevenlabs":
                from .adapters import ElevenLabsTTSAdapter

                tts = ElevenLabsTTSAdapter(api_key=settings.elevenlabs_api_key, timeout=settings.http_timeout)
            else:
                tts = HFTTSAdapter(
                    settings.hf_api_token,
                    settings.tts_model,
                    settings.tts_endpoint,
                    settings.http_timeout,
                    settings.http_retries,
                )
            res = tts.synthesize_to_file(
                TTSRequest(text=args.text, voice=args.voice, language=args.language, speaker_wav_path=args.speaker_wav),
                args.out,
            )
            print(res.audio_path)
        elif args.cmd == "overlay-text":
            draw = build_drawtext_filter(args.text, fontfile=args.fontfile)
            out = render_text_overlay(settings.ffmpeg_bin, args.input_video, args.output_video, draw)
            print(out)
        elif args.cmd == "overlay-png":
            out = overlay_image(settings.ffmpeg_bin, args.input_video, args.png, args.output_video, x=args.x, y=args.y)
            print(out)
        elif args.cmd == "doctor":
            print("[doctor] Python: ok")
            try:
                ver = ffmpeg_version(settings.ffmpeg_bin)
                print(f"[doctor] FFmpeg: {ver}")
            except Exception as e:  # noqa: BLE001
                print(f"[doctor] FFmpeg error: {e}")
                raise
            # tmp dir write test
            Path(settings.tmp_dir).mkdir(parents=True, exist_ok=True)
            test_file = Path(settings.tmp_dir) / "doctor_write_test.txt"
            test_file.write_text("ok", encoding="utf-8")
            print(f"[doctor] TMP write: ok ({test_file})")
            # HF token presence
            if settings.hf_api_token:
                print("[doctor] HF token: present")
            else:
                print("[doctor] HF token: missing (STT/MT/TTS will fail)")
            if args.check_hf:
                from .http_client import HttpClient

                client = HttpClient(timeout=settings.http_timeout, retries=1)
                # HF endpoints
                if settings.hf_api_token:
                    checks = [
                        (settings.stt_endpoint or f"https://api-inference.huggingface.co/models/{settings.stt_model}", "HF STT"),
                        (settings.mt_endpoint or f"https://api-inference.huggingface.co/models/{settings.mt_model}", "HF MT"),
                        (settings.tts_endpoint or f"https://api-inference.huggingface.co/models/{settings.tts_model}", "HF TTS"),
                    ]
                    for url, name in checks:
                        try:
                            resp = client.request(
                                "GET",
                                url,
                                headers={"Authorization": f"Bearer {settings.hf_api_token}"},
                                expected_status=(200, 401, 403, 404, 405),
                            )
                            print(f"[doctor] Reach {name}: {url} -> {resp.status_code}")
                        except Exception as e:  # noqa: BLE001
                            print(f"[doctor] Reach {name}: {url} -> error {e}")
                # OpenAI
                if settings.stt_provider == "openai" and settings.openai_api_key:
                    try:
                        with httpx.Client(timeout=settings.http_timeout) as c:  # type: ignore[name-defined]
                            r = c.get(
                                (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/") + "/models",
                                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                            )
                            print(f"[doctor] Reach OpenAI models: {r.status_code}")
                    except Exception as e:  # noqa: BLE001
                        print(f"[doctor] Reach OpenAI: error {e}")
                # ElevenLabs
                if settings.tts_provider == "elevenlabs" and settings.elevenlabs_api_key:
                    try:
                        with httpx.Client(timeout=settings.http_timeout) as c:  # type: ignore[name-defined]
                            r = c.get("https://api.elevenlabs.io/v1/user", headers={"xi-api-key": settings.elevenlabs_api_key})
                            print(f"[doctor] Reach ElevenLabs user: {r.status_code}")
                    except Exception as e:  # noqa: BLE001
                        print(f"[doctor] Reach ElevenLabs: error {e}")
            print("[doctor] Done")
        elif args.cmd == "smoke":
            # Generate sample video
            sample = Path(settings.tmp_dir) / "smoke_sample.mp4"
            generate_test_video(settings.ffmpeg_bin, str(sample), duration=2.0)
            # Overlay text
            draw = build_drawtext_filter(args.text)
            text_vid = Path(settings.tmp_dir) / "smoke_text.mp4"
            render_text_overlay(settings.ffmpeg_bin, str(sample), str(text_vid), draw)
            current = text_vid
            # Optional PNG overlay
            if args.with_png:
                png = Path(settings.tmp_dir) / "smoke_logo.png"
                generate_solid_png(settings.ffmpeg_bin, str(png), size="160x60", color="white")
                png_vid = Path(settings.tmp_dir) / "smoke_png.mp4"
                overlay_image(settings.ffmpeg_bin, str(current), str(png), str(png_vid), x="(W-w)-20", y="20")
                current = png_vid
            # Burn simple subtitles
            from .subtitles.processor import pysubs2, export_srt, transcript_to_subs
            from .dto import Transcript, TranscriptSegment

            subs_transcript = Transcript(
                segments=[
                    TranscriptSegment(start=0.2, end=1.0, text="Hello, world!"),
                    TranscriptSegment(start=1.0, end=1.8, text="Smoke test OK"),
                ]
            )
            subs = transcript_to_subs(subs_transcript)
            srt_path = Path(settings.tmp_dir) / "smoke.srt"
            export_srt(subs, str(srt_path))
            # Burn
            burned = Path(settings.tmp_dir) / "smoke_burn.mp4"
            burn_subtitles(settings.ffmpeg_bin, str(current), str(srt_path), str(burned))
            # Output
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_bytes(burned.read_bytes())
            print(str(args.out))
        elif args.cmd == "snapshot":
            from .utils.snapshot import create_snapshot

            tar_path, meta_path = create_snapshot(
                settings,
                stage_path=args.path,
                name=args.name,
                out_dir=args.out_dir,
            )
            print(f"[snapshot] tar: {tar_path}\n[metadata] {meta_path}")
        elif args.cmd == "restore-snapshot":
            from .utils.snapshot import restore_snapshot

            out_dir = restore_snapshot(args.file, args.dest)
            print(f"[restore] extracted to: {out_dir}")
        elif args.cmd == "lock-stage":
            root = Path(args.dir)
            if not root.exists():
                print(f"No such directory: {root}")
                return
            count = 0
            for p in root.iterdir():
                if p.is_dir():
                    lf = p / ".lock"
                    if args.unlock:
                        if lf.exists():
                            try:
                                lf.unlink()
                                count += 1
                            except Exception:
                                pass
                    else:
                        try:
                            lf.write_text("locked", encoding="utf-8")
                            count += 1
                        except Exception:
                            pass
            action = "unlocked" if args.unlock else "locked"
            print(f"[lock-stage] {action} {count} subdirs under {root}")
        elif args.cmd == "build-bed":
            from .audio.bed import build_music_bed
            import glob, random

            files: list[str] = []
            if args.input_video:
                files.extend(args.input_video)
            if args.input_dir:
                vids = []
                for ext in ("*.mp4", "*.mov", "*.mkv"):
                    vids.extend(glob.glob(str(Path(args.input_dir) / ext)))
                if args.count and vids:
                    random.shuffle(vids)
                    vids = vids[: args.count]
                files.extend(vids)
            if not files:
                print("No input videos provided")
                return
            out_dir = args.out_dir or settings.tmp_dir
            for f in files:
                bed, rep = build_music_bed(
                    settings,
                    input_video=f,
                    out_dir=out_dir,
                    method=args.method,
                    mlev=args.mlev,
                    base_volume=args.blend,
                    force=args.force,
                )
                print(f"[bed] {f} -> {bed}\n[report] {rep}")
        elif args.cmd == "separate":
            from .pipeline.steps.separate import run as sep_run
            import glob, random

            files: list[str] = []
            if args.input_video:
                files.extend(args.input_video)
            if args.input_dir:
                vids = []
                for ext in ("*.mp4", "*.mov", "*.mkv"):
                    vids.extend(glob.glob(str(Path(args.input_dir) / ext)))
                if args.count and vids:
                    random.shuffle(vids)
                    vids = vids[: args.count]
                files.extend(vids)
            if not files:
                print("No input videos provided")
                return
            out_dir = args.out_dir
            for f in files:
                try:
                    res = sep_run(
                        settings,
                        input_video=f,
                        out_dir=out_dir,
                        model=args.model,
                        use_cpu=args.cpu or True,
                        force=args.force,
                    )
                    print(f"[separate] {f}\n  vocals: {res.vocals_wav}\n  bed: {res.bed_wav}\n  report: {res.report_path}")
                except Exception as e:  # noqa: BLE001
                    print(f"[separate] failed for {f}: {e}")
        # (validation pack branch removed)
    except Exception as e:  # noqa: BLE001
        log = logging.getLogger("cli")
        log.error("Error: %s", e)
        raise


if __name__ == "__main__":
    main()
