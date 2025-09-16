# AI Video Ad Localization — Hybrid Orchestrator (MVP)

Lightweight, modular Python orchestrator for localizing video ads using best-of-breed APIs:

- STT (Whisper via Hugging Face Inference API)
- MT (Hugging Face translation models; optional glossaries)
- TTS (Hugging Face TTS models)
- Subtitles (pysubs2 for SRT/VTT import/export and normalization)
- Simple text overlay and PNG overlay
- Safe FFmpeg muxing and rendering via preset utilities

Architectural highlights: modular monolith, adapters layer, typed DTOs, retries/timeouts, idempotent steps, 12-factor config via env.

## Quick Start

Prerequisites:

- Python 3.10+
- FFmpeg installed and available in `PATH`
- (Optional) Hugging Face API token for STT/MT/TTS: set `HF_API_TOKEN`

Install deps:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Environment (12-factor style):

```bash
export HF_API_TOKEN=hf_xxx
export STT_MODEL=openai/whisper-small
export MT_MODEL=facebook/nllb-200-distilled-600M
export TTS_MODEL=suno/bark
export FFMPEG_BIN=ffmpeg  # or absolute path
export TMP_DIR=.ai_loca_tmp
export HTTP_TIMEOUT=60
export HTTP_RETRIES=3
# Post-processing knobs for Google text detection results
export MIN_CONFIDENCE=0.55            # drop segments below this confidence (if present)
export MIN_AREA_RATIO=0.0005          # drop tiny polygons (< this fraction of frame area)
export MASK_PADDING_PX=3              # dilate mask polygons by N pixels
export NMS_IOU=0.7                    # deduplicate polygons at same timestamp if AABB IoU >= this
export MIN_SEGMENT_DURATION_SEC=0     # drop very short segments (< seconds)
## Optional: custom Inference Endpoints (instead of HF API)
# export STT_ENDPOINT="https://<your-stt-endpoint>"
# export MT_ENDPOINT="https://<your-mt-endpoint>"
# export TTS_ENDPOINT="https://<your-tts-endpoint>"

# Provider selection (defaults: HF)
# For OpenAI Whisper STT
# export STT_PROVIDER=openai
# export OPENAI_API_KEY=sk-...
# export OPENAI_BASE_URL=https://api.openai.com/v1  # optional
# For ElevenLabs TTS
# export TTS_PROVIDER=elevenlabs
# export ELEVENLABS_API_KEY=eleven_...

Recommended premium models:
- STT: Systran/faster-whisper-large-v3 (Endpoint)
- MT: facebook/nllb-200-distilled-600M (serverless often OK)
- TTS: FunAudioLLM/CosyVoice-2 (0.5B or 1B) (Endpoint)
```

Run CLI help (no install):

```bash
python run.py --help
```

Health check and offline smoke test:

```bash
# Verify environment (FFmpeg, tmp dir, token presence)
python run.py doctor

# Optionally ping HF endpoints or your custom endpoints (requires network + token)
python run.py doctor --check-hf

# Offline smoke test (generates short sample video, overlays text, burns subs)
python run.py smoke --out ./output/smoke_test.mp4 --with-png
```

Example end-to-end localization:

```bash
python run.py localize \
  --input-video ./samples/input.mp4 \
  --source-lang en \
  --target-lang ru \
  --output-video ./output/localized_ru.mp4 \
  --tts-language ru \
  --tts-speaker-wav ./assets/voice_ref.wav \
  --tts-segmented \
  --overlay-text "Limited offer" \
  --overlay-png ./assets/logo.png \
  --burn-subtitles
```

Notes:

- Without `HF_API_TOKEN`, adapter calls will fail gracefully with a clear error.
- Steps are idempotent where possible: intermediate artifacts reuse TMP_DIR paths based on content hash.

## Project Layout

```
src/ai_loca
├── adapters
│   ├── __init__.py
│   ├── mt_hf.py
│   ├── stt_hf.py
│   └── tts_hf.py
├── cli.py
├── config.py
├── dto.py
├── errors.py
├── ffmpeg_presets.py
├── http_client.py
├── logging.py
├── pipeline
│   └── orchestrator.py
├── subtitles
│   └── processor.py
└── utils
    ├── ffmpeg.py
    └── retry.py
```

## Dependencies

Minimal runtime dependencies:

- httpx — HTTP client
- tenacity — retries with exponential backoff
- pydantic — DTOs and validation
- pysubs2 — subtitle processing (SRT/VTT)

Dev/test dependencies are intentionally omitted to keep the MVP lean.

## Security & Reliability

- No shell string execution — FFmpeg invoked with arg lists.
- Logged FFmpeg commands for traceability.
- Timeouts and retries on HTTP calls; typed DTOs; explicit exceptions.

## License

MVP reference implementation. Provide your own license before distribution.
