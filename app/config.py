import os
from functools import lru_cache
from pydantic import BaseModel, Field, field_validator


class Settings(BaseModel):
    app_name: str = Field(default="video-text-masker")
    environment: str = Field(default=os.getenv("ENV", "dev"))

    gcp_project: str | None = Field(default=os.getenv("GCP_PROJECT"))
    gcs_bucket: str = Field(default=os.getenv("GCS_BUCKET", "video-text-masker"))
    gcp_credentials: str | None = Field(default=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

    # Signed URL expiration in seconds
    gcs_signed_url_ttl: int = Field(default=int(os.getenv("GCS_SIGNED_URL_TTL", "86400")))

    # Local temp storage
    tmp_dir: str = Field(default=os.getenv("TMP_DIR", "./.tmp"))

    # Post-processing knobs
    min_confidence: float = Field(default=float(os.getenv("MIN_CONFIDENCE", "0.55")))
    min_area_ratio: float = Field(default=float(os.getenv("MIN_AREA_RATIO", "0.0005")))
    mask_padding_px: int = Field(default=int(os.getenv("MASK_PADDING_PX", "3")))
    nms_iou_threshold: float = Field(default=float(os.getenv("NMS_IOU", "0.7")))
    min_segment_duration_sec: float = Field(default=float(os.getenv("MIN_SEGMENT_DURATION_SEC", "0")))

    @field_validator("gcs_signed_url_ttl")
    @classmethod
    def _positive_ttl(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("GCS_SIGNED_URL_TTL must be > 0")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    os.makedirs(s.tmp_dir, exist_ok=True)
    return s
