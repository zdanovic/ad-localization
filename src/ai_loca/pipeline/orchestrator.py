from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

from ..config import Settings
from ..dto import Transcript, TranscriptSegment, TranslationRequest, TTSRequest
from ..errors import OrchestratorError
from ..utils.ffmpeg import (
        build_drawtext_filter,
        deterministic_artifact,
        extract_audio,
        mux_audio_video,
        overlay_image,
        render_text_overlay,
        make_music_bed,
        duck_mix_tts_with_bed,
    )
from ..audio.stems import build_stems_demucs, build_stems_ms
from ..adapters import (
    HFWhisperSTTAdapter,
    HFTranslationAdapter,
    HFTTSAdapter,
    OpenAIWhisperSTTAdapter,
    ElevenLabsTTSAdapter,
    OpenAITranslationAdapter,
    OpenAITTSAdapter,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        Path(settings.tmp_dir).mkdir(parents=True, exist_ok=True)

        if settings.stt_provider == "openai":
            self.stt = OpenAIWhisperSTTAdapter(
                api_key=settings.openai_api_key,
                model="whisper-1",
                base_url=settings.openai_base_url,
                timeout=settings.http_timeout,
            )
        else:
            self.stt = HFWhisperSTTAdapter(
                api_token=settings.hf_api_token,
                model=settings.stt_model,
                endpoint_url=settings.stt_endpoint,
                timeout=settings.http_timeout,
                retries=settings.http_retries,
            )
        if settings.mt_provider == "openai":
            self.mt = OpenAITranslationAdapter(
                api_key=settings.openai_api_key,
                model="gpt-4o-mini",
                base_url=settings.openai_base_url,
                timeout=settings.http_timeout,
            )
        else:
            self.mt = HFTranslationAdapter(
                api_token=settings.hf_api_token,
                model=settings.mt_model,
                endpoint_url=settings.mt_endpoint,
                timeout=settings.http_timeout,
                retries=settings.http_retries,
            )
        if settings.tts_provider == "elevenlabs":
            self.tts = ElevenLabsTTSAdapter(api_key=settings.elevenlabs_api_key, timeout=settings.http_timeout)
        elif settings.tts_provider == "openai":
            self.tts = OpenAITTSAdapter(
                api_key=settings.openai_api_key,
                model=settings.tts_model_openai,
                base_url=settings.openai_base_url,
                timeout=settings.http_timeout,
            )
        else:
            self.tts = HFTTSAdapter(
                api_token=settings.hf_api_token,
                model=settings.tts_model,
                endpoint_url=settings.tts_endpoint,
                timeout=settings.http_timeout,
                retries=settings.http_retries,
            )

    def _hash_key(self, *parts: str) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode("utf-8"))
        return h.hexdigest()[:16]

    def localize_video(
        self,
        input_video: str,
        source_lang: Optional[str],
        target_lang: str,
        output_video: str,
        overlay_text: Optional[str] = None,
        overlay_png: Optional[str] = None,
        force: bool = False,
        tts_voice: Optional[str] = None,
        tts_language: Optional[str] = None,
        tts_speaker_wav: Optional[str] = None,
        tts_segmented: bool = True,
        pro_dub: bool = False,
        auto_speaker_ref: bool = False,
        bed_mode: str = "final",
        keep_tmp: bool = False,
    ) -> str:
        try:
            key = self._hash_key(input_video, target_lang)
            # 1. Extract audio
            audio_path = deterministic_artifact(self.settings.tmp_dir, key, "audio.wav")
            if not Path(audio_path).exists() or force:
                extract_audio(self.settings.ffmpeg_bin, input_video, audio_path)

            # 2. STT or load input subtitles
            trans_path = deterministic_artifact(self.settings.tmp_dir, key, "transcript.json")
            if Path(trans_path).exists() and not force:
                text = Path(trans_path).read_text(encoding="utf-8")
                transcript = Transcript.model_validate_json(text)
            else:
                transcript = self.stt.transcribe(audio_path, language=source_lang)
                Path(trans_path).write_text(transcript.model_dump_json(indent=2), encoding="utf-8")

            # 3. MT (segment-wise)
            inferred_src = source_lang or (transcript.language if transcript.language and len(transcript.language) <= 5 else "")
            translated_segments: list[TranscriptSegment] = []
            for seg in transcript.segments:
                req = TranslationRequest(source_lang=inferred_src, target_lang=target_lang, text=seg.text)
                out = self.mt.translate(req)
                translated_segments.append(TranscriptSegment(start=seg.start, end=seg.end, text=out.text))
            translated = Transcript(language=target_lang, segments=translated_segments)

            # 4-5. TTS and Mux
            muxed_video = deterministic_artifact(self.settings.tmp_dir, key, f"mux_{target_lang}.mp4")
            # Prepare speaker reference
            ref_wav = tts_speaker_wav or audio_path
            if auto_speaker_ref and not tts_speaker_wav:
                from ..audio.speaker_ref import build_speaker_ref

                ref_wav = build_speaker_ref(self.settings, input_video=input_video, out_dir=self.settings.tmp_dir, dur_sec=2.5, force=force)
            elif self.settings.tts_provider == "elevenlabs" and not tts_speaker_wav:
                from ..utils.ffmpeg import trim_wav

                trimmed_ref = deterministic_artifact(self.settings.tmp_dir, key, "voice_ref_20s.wav")
                if not Path(trimmed_ref).exists() or force:
                    trim_wav(self.settings.ffmpeg_bin, audio_path, trimmed_ref, duration_sec=20.0)
                ref_wav = trimmed_ref
            # Analyze original loudness to target bed level
            from ..utils.ffmpeg import analyze_loudness, get_audio_duration, time_stretch_wav, polish_voice
            orig_lufs = analyze_loudness(self.settings.ffmpeg_bin, audio_path)

            if tts_segmented:
                # Synthesize per segment and concat
                from ..utils.ffmpeg import normalize_wav, concat_wavs

                wavs: list[str] = []
                if pro_dub and self.settings.openai_api_key:
                    # Plan per-segment translation + prosody
                    try:
                        from ..dub.planner import ProsodyPlanner

                        planner = ProsodyPlanner(self.settings)
                        plan = planner.plan(transcript, source_lang=source_lang, target_lang=target_lang)
                        # Build a map: index -> planned item; fallback to translated if missing
                        planned = {p.index: p for p in plan.segments}
                    except Exception:
                        planned = {}
                else:
                    planned = {}

                for i, seg in enumerate(translated.segments):
                    seg_wav = deterministic_artifact(self.settings.tmp_dir, key + f"_seg{i}", f"seg_{i}.wav")
                    if not Path(seg_wav).exists() or force:
                        p = planned.get(i)
                        req = TTSRequest(
                            text=(p.text if p and p.text else seg.text),
                            voice=tts_voice,
                            language=tts_language or target_lang,
                            speaker_wav_path=ref_wav,
                        )
                        if p:
                            req.style = p.style
                            req.emotion = p.emotion
                            req.speed = p.speed
                            req.pitch = p.pitch
                        self.tts.synthesize_to_file(req, seg_wav)
                    # Stretch to match original segment duration for tighter lip-sync
                    target_dur = max(0.2, seg.end - seg.start)
                    cur_dur = get_audio_duration("ffprobe", seg_wav)
                    stretch = target_dur / cur_dur if cur_dur > 0 else 1.0
                    if abs(1 - stretch) > 0.05:
                        stretched = deterministic_artifact(
                            self.settings.tmp_dir, key + f"_seg{i}", f"seg_{i}_stretch.wav"
                        )
                        time_stretch_wav(self.settings.ffmpeg_bin, seg_wav, stretched, stretch)
                        seg_wav = stretched
                    # Light voice polishing
                    polished = deterministic_artifact(self.settings.tmp_dir, key + f"_seg{i}", f"seg_{i}_vox.wav")
                    polish_voice(self.settings.ffmpeg_bin, seg_wav, polished)
                    seg_wav = polished
                    # Normalize to PCM s16le for concat
                    norm_wav = deterministic_artifact(self.settings.tmp_dir, key + f"_seg{i}", f"seg_{i}_norm.wav")
                    if not Path(norm_wav).exists() or force:
                        normalize_wav(self.settings.ffmpeg_bin, seg_wav, norm_wav)
                    wavs.append(norm_wav)
                tts_audio = deterministic_artifact(self.settings.tmp_dir, key, f"tts_{target_lang}.wav")
                concat_wavs(self.settings.ffmpeg_bin, wavs, tts_audio)
            else:
                tts_audio = deterministic_artifact(self.settings.tmp_dir, key, f"tts_{target_lang}.wav")
                if not Path(tts_audio).exists() or force:
                    self.tts.synthesize_to_file(
                        TTSRequest(
                            text=translated.text,
                            voice=tts_voice,
                            language=tts_language or target_lang,
                            speaker_wav_path=ref_wav,
                        ),
                        tts_audio,
                    )
                polished = deterministic_artifact(self.settings.tmp_dir, key, f"tts_{target_lang}_vox.wav")
                polish_voice(self.settings.ffmpeg_bin, tts_audio, polished)
                tts_audio = polished
            # Build music/SFX bed from original audio and duck under TTS, then mux
            # Prefer 'final' clean bed via Demucs; fallback to fast ms gate.
            if bed_mode == "final":
                try:
                    _voc, acc, _rep = build_stems_demucs(
                        self.settings,
                        input_video=input_video,
                        out_dir=self.settings.tmp_dir,
                        model="htdemucs_ft",
                        use_cpu=True,
                        force=force,
                    )
                    bed_wav = acc
                except Exception:
                    bed_wav = deterministic_artifact(self.settings.tmp_dir, key, f"bed_{target_lang}.wav")
                    make_music_bed(self.settings.ffmpeg_bin, audio_path, bed_wav)
                    # Light extra smoothing using quick vocals estimate
                    try:
                        voc_fast, _acc_fast, _rep2 = build_stems_ms(
                            self.settings, input_video=input_video, out_dir=self.settings.tmp_dir, force=force
                        )
                        from ..utils.ffmpeg import smooth_bed_with_vocals

                        cleaned = deterministic_artifact(self.settings.tmp_dir, key, f"bed_{target_lang}_clean.wav")
                        smooth_bed_with_vocals(self.settings.ffmpeg_bin, bed_wav, voc_fast, cleaned)
                        bed_wav = cleaned
                    except Exception:
                        pass
            else:
                bed_wav = deterministic_artifact(self.settings.tmp_dir, key, f"bed_{target_lang}.wav")
                if not Path(bed_wav).exists() or force:
                    make_music_bed(self.settings.ffmpeg_bin, audio_path, bed_wav)
                    try:
                        voc_fast, _acc_fast, _rep2 = build_stems_ms(
                            self.settings, input_video=input_video, out_dir=self.settings.tmp_dir, force=force
                        )
                        from ..utils.ffmpeg import smooth_bed_with_vocals

                        cleaned = deterministic_artifact(self.settings.tmp_dir, key, f"bed_{target_lang}_clean.wav")
                        smooth_bed_with_vocals(self.settings.ffmpeg_bin, bed_wav, voc_fast, cleaned)
                        bed_wav = cleaned
                    except Exception:
                        pass
            mixed_wav = deterministic_artifact(self.settings.tmp_dir, key, f"mix_{target_lang}.wav")
            if not Path(mixed_wav).exists() or force:
                # Target bed loudness closer to original, but below voice by ~3 LU
                voice_lufs = -18.0
                bed_target = min(orig_lufs + 1.0, voice_lufs + 3.0)
                duck_mix_tts_with_bed(
                    self.settings.ffmpeg_bin, tts_audio, bed_wav, mixed_wav, voice_lufs=voice_lufs, bed_lufs=bed_target
                )
            if not Path(muxed_video).exists() or force:
                mux_audio_video(self.settings.ffmpeg_bin, input_video, mixed_wav, muxed_video, reencode=False)

            # 7. Optional overlays
            current_out = muxed_video
            if overlay_text:
                draw = build_drawtext_filter(overlay_text)
                with_text = deterministic_artifact(self.settings.tmp_dir, key, f"text_{target_lang}.mp4")
                render_text_overlay(self.settings.ffmpeg_bin, current_out, with_text, draw)
                current_out = with_text
            if overlay_png:
                with_png = deterministic_artifact(self.settings.tmp_dir, key, f"png_{target_lang}.mp4")
                overlay_image(self.settings.ffmpeg_bin, current_out, overlay_png, with_png)
                current_out = with_png

            # 8. Save to requested output path
            Path(output_video).parent.mkdir(parents=True, exist_ok=True)
            if Path(current_out) != Path(output_video):
                # Copy file efficiently
                Path(output_video).write_bytes(Path(current_out).read_bytes())
            # 9. Cleanup tmp unless requested to keep
            if not keep_tmp:
                try:
                    self._cleanup_tmp_dirs(key, input_video)
                except Exception:
                    pass
            return output_video
        except Exception as e:  # noqa: BLE001
            raise OrchestratorError(str(e)) from e

    def _artifact_dir(self, key: str) -> Path:
        # Mirror deterministic_artifact hashing (sha256 of key, first 16 chars)
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return Path(self.settings.tmp_dir) / h

    def _cleanup_tmp_dirs(self, key: str, input_video: str) -> None:
        import shutil

        # Remove hashed artifact directory for this run
        art_dir = self._artifact_dir(key)
        if art_dir.exists():
            shutil.rmtree(art_dir, ignore_errors=True)

        # Remove Demucs residual folders for this video
        stem = Path(input_video).stem
        base = Path(self.settings.tmp_dir)
        for p in base.glob(f"demucs_{stem}"):
            shutil.rmtree(p, ignore_errors=True)

        # Remove auxiliary wavs produced during stems (acc/vocals/intermediate)
        patterns = [
            f"{stem}_audio.wav",
            f"{stem}_acc.wav",
            f"{stem}_acc_clean*.wav",
            f"{stem}_acc_refill.wav",
            f"{stem}_acc_final.wav",
            f"{stem}_acc_16k.wav",
            f"{stem}_vocals.wav",
            f"{stem}_vocals_16k.wav",
            f"{stem}_stems_report.json",
        ]
        for pat in patterns:
            for p in base.glob(pat):
                try:
                    p.unlink()
                except Exception:
                    pass
