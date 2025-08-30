from __future__ import annotations

import logging
from typing import Dict, Optional

from ..dto import TranslationRequest, TranslationResponse
from ..errors import AIAdapterError, ConfigError
from ..http_client import HttpClient

logger = logging.getLogger(__name__)


def _apply_glossary(text: str, glossary: Dict[str, str]) -> str:
    # Simple pre-replacement to steer MT; non-overlapping naive approach
    for k, v in glossary.items():
        text = text.replace(k, v)
    return text


def _deapply_glossary(text: str, glossary: Dict[str, str]) -> str:
    # No-op placeholder for bidirectional glossaries; keep simple for MVP
    return text


class HFTranslationAdapter:
    """Hugging Face translation adapter using `translation` pipeline models.

    Example:

        from ai_loca.adapters.mt_hf import HFTranslationAdapter
        adapter = HFTranslationAdapter(api_token="hf_xxx", model="Helsinki-NLP/opus-mt-en-ru")
        out = adapter.translate(TranslationRequest(source_lang="en", target_lang="ru", text="Hello"))
        print(out.text)

    With glossary:

        req = TranslationRequest(source_lang="en", target_lang="ru", text="ACME Cloud", glossary={"ACME": "Эйкми"})
        out = adapter.translate(req)

    """

    def __init__(self, api_token: Optional[str], model: str, endpoint_url: Optional[str] = None, timeout: float = 60.0, retries: int = 3) -> None:
        if not model:
            raise ConfigError("MT model must be provided")
        self.model = model
        self.endpoint_url = endpoint_url
        self._client = HttpClient(timeout=timeout, retries=retries)
        self._headers = {"Accept": "application/json"}
        if api_token:
            self._headers["Authorization"] = f"Bearer {api_token}"

    def _nllb_code(self, iso2: str) -> Optional[str]:
        mapping = {
            "en": "eng_Latn",
            "es": "spa_Latn",
            "ru": "rus_Cyrl",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "it": "ita_Latn",
            "pt": "por_Latn",
            "pl": "pol_Latn",
            "tr": "tur_Latn",
            "uk": "ukr_Cyrl",
            "ar": "arb_Arab",
            "hi": "hin_Deva",
            "zh": "zho_Hans",
            "ja": "jpn_Jpan",
            "ko": "kor_Hang",
        }
        return mapping.get((iso2 or "").lower())

    def translate(self, req: TranslationRequest) -> TranslationResponse:
        url = self.endpoint_url or f"https://api-inference.huggingface.co/models/{self.model}"
        text = req.text
        if req.glossary:
            text = _apply_glossary(text, req.glossary)
        payload = {"inputs": text}
        model_l = self.model.lower()
        if "nllb" in model_l:
            params: Dict[str, str] = {}
            if req.source_lang:
                code = self._nllb_code(req.source_lang)
                if code:
                    params["src_lang"] = code
            if req.target_lang:
                code = self._nllb_code(req.target_lang)
                if code:
                    params["tgt_lang"] = code
            if params:
                payload["parameters"] = params
        elif "m2m100" in model_l:
            params = {}
            if req.source_lang:
                params["src_lang"] = req.source_lang
            if req.target_lang:
                params["tgt_lang"] = req.target_lang
            if params:
                payload["parameters"] = params
        try:
            resp = self._client.request("POST", url, headers=self._headers, json=payload)
            data = resp.json()
            # HF returns list of dicts: [{"translation_text": "..."}] or a dict
            if isinstance(data, list) and data and isinstance(data[0], dict):
                out_text = data[0].get("translation_text") or data[0].get("generated_text") or ""
            elif isinstance(data, dict):
                out_text = data.get("translation_text") or data.get("generated_text") or ""
            else:
                out_text = ""
            if req.glossary:
                out_text = _deapply_glossary(out_text, req.glossary)
            return TranslationResponse(text=out_text)
        except Exception as e:  # noqa: BLE001
            raise AIAdapterError(f"MT failed: {e}") from e
