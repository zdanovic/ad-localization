from __future__ import annotations

import json
import mimetypes
import os
from datetime import timedelta
import time
from typing import Optional

from google.cloud import storage

from ..config import get_settings


class GCSClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = storage.Client(project=self.settings.gcp_project)
        self.bucket = self.client.bucket(self.settings.gcs_bucket)

    def blob(self, blob_name: str):
        return self.bucket.blob(blob_name)

    def upload_file(self, local_path: str, blob_name: str, content_type: Optional[str] = None) -> str:
        if content_type is None:
            content_type = mimetypes.guess_type(local_path)[0] or "application/octet-stream"
        blob = self.blob(blob_name)
        # simple retries for GCS 429 rate limits
        delay = 0.5
        for attempt in range(6):
            try:
                blob.upload_from_filename(local_path, content_type=content_type)
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "429" in msg or "rateLimitExceeded" in msg:
                    time.sleep(delay)
                    delay = min(delay * 2, 8.0)
                    continue
                raise
        return f"gs://{self.settings.gcs_bucket}/{blob_name}"

    def upload_bytes(self, data: bytes, blob_name: str, content_type: str) -> str:
        blob = self.blob(blob_name)
        delay = 0.5
        for attempt in range(6):
            try:
                blob.upload_from_string(data, content_type=content_type)
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "429" in msg or "rateLimitExceeded" in msg:
                    time.sleep(delay)
                    delay = min(delay * 2, 8.0)
                    continue
                raise
        return f"gs://{self.settings.gcs_bucket}/{blob_name}"

    def upload_json(self, payload: dict, blob_name: str, pretty: bool = True) -> str:
        data = json.dumps(payload, indent=2 if pretty else None, ensure_ascii=False).encode("utf-8")
        return self.upload_bytes(data, blob_name, content_type="application/json; charset=utf-8")

    def signed_url(self, blob_name: str) -> Optional[str]:
        try:
            blob = self.blob(blob_name)
            url = blob.generate_signed_url(expiration=timedelta(seconds=self.settings.gcs_signed_url_ttl))
            return url
        except Exception:
            return None
