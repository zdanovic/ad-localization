from __future__ import annotations

import json
from typing import List

import httpx

from ..config import Settings
from ..dto import Transcript, DubPlan, DubSegmentPlan
from ..errors import ConfigError


class ProsodyPlanner:
    """LLM-based dubbing planner that produces per-segment translation and prosody hints.

    Uses OpenAI chat completions to generate a JSON plan with fields:
    - text: translated line preserving tone/intent
    - emotion/style: coarse labels (e.g., excited, calm; conversational, announcer)
    - speed: float multiplier to aim for target duration (1.0 default)
    - pitch: optional relative adjustment
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ConfigError("OPENAI_API_KEY required for prosody planning")
        self._api_key = settings.openai_api_key
        self._base = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
        self._model = "gpt-4o-mini"
        self._timeout = settings.http_timeout

    def plan(self, transcript: Transcript, source_lang: str | None, target_lang: str) -> DubPlan:
        items = []
        for i, s in enumerate(transcript.segments):
            items.append({"index": i, "start": s.start, "end": s.end, "text": s.text})
        system = (
            "You are a professional dubbing director for advertising. "
            "Given segments with start/end seconds and source text, produce a JSON array where each item provides: \n"
            "- index (copy input)\n- text: translated line in the target language, preserving meaning, tone and persuasive impact; \n"
            "- emotion: one of [neutral, excited, urgent, calm, happy, sad, angry] — choose boldly to match intent;\n"
            "- style: one of [conversational, announcer, cinematic, playful, corporate];\n"
            "- speed: float speed multiplier to fit the same duration window (1.0 default);\n"
            "- pitch: optional float shift in semitones (0 default).\n"
            "Aim for natural, emotive delivery suitable for ad VO. Keep concise for lip sync. Output pure JSON."
        )
        user = {
            "task": "dub-plan",
            "source_language": source_lang or transcript.language or "auto",
            "target_language": target_lang,
            "segments": items,
        }
        payload = {
            "model": self._model,
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=self._timeout) as client:
            r = client.post(f"{self._base}/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"].strip()
        try:
            parsed = json.loads(content)
            arr = parsed.get("segments") or parsed
            segs: List[DubSegmentPlan] = []
            if isinstance(arr, list):
                for it in arr:
                    segs.append(
                        DubSegmentPlan(
                            index=int(it.get("index", 0)),
                            start=float(it.get("start", 0.0)),
                            end=float(it.get("end", 0.0)),
                            text=str(it.get("text", "")).strip(),
                            style=(it.get("style") or None),
                            emotion=(it.get("emotion") or None),
                            speed=(float(it["speed"]) if (it.get("speed") is not None) else None),
                            pitch=(float(it["pitch"]) if (it.get("pitch") is not None) else None),
                        )
                    )
            return DubPlan(language=target_lang, segments=segs)
        except Exception as e:  # noqa: BLE001
            # Fallback: neutral plan mirroring text
            segs = [
                DubSegmentPlan(index=i, start=s.start, end=s.end, text=s.text, style="conversational", emotion="neutral", speed=1.0)
                for i, s in enumerate(transcript.segments)
            ]
            return DubPlan(language=target_lang, segments=segs)
