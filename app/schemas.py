from __future__ import annotations

from enum import StrEnum
from typing import List, Optional
from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    error = "error"


class CreateVideoResponse(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.in_progress


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    message: Optional[str] = None
    result_gcs_uri: Optional[str] = None
    result_url: Optional[str] = None


class DetectionFrame(BaseModel):
    time_offset_sec: float
    bounding_poly: list[dict]
    mask_gcs_uri: str
    mask_url: str | None = None


class DetectionSegment(BaseModel):
    start_time_sec: float | None = None
    end_time_sec: float | None = None
    confidence: float | None = None
    frames: list[DetectionFrame] = Field(default_factory=list)


class DetectionItem(BaseModel):
    text: str
    segments: list[DetectionSegment]


class VideoInfo(BaseModel):
    filename: str
    gcs_uri: str
    duration_sec: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None


class ResultPayload(BaseModel):
    job_id: str
    language_hints: List[str] = Field(default_factory=list)
    video: VideoInfo
    detections: List[DetectionItem]

