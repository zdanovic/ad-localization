from __future__ import annotations

import json
import os
import shutil
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .config import get_settings
from .jobs import job_store
from .schemas import CreateVideoResponse, JobInfo, JobStatus
from .services.gcs import GCSClient
from .services.video_intel import VideoIntelClient
from .services.processing import process_annotations


app = FastAPI(title="Video Text Masker", version="1.0.0")


def _save_upload_to_tmp(job_id: str, upload: UploadFile) -> str:
    settings = get_settings()
    job_dir = os.path.join(settings.tmp_dir, job_id, "input")
    os.makedirs(job_dir, exist_ok=True)
    filename = upload.filename or "upload.bin"
    local_path = os.path.join(job_dir, filename)
    with open(local_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return local_path


def _parse_languages(languages_str: Optional[str] | None) -> List[str]:
    if not languages_str:
        return []
    # Accept JSON string or comma-separated
    try:
        value = json.loads(languages_str)
        if isinstance(value, list):
            return [str(x) for x in value]
    except Exception:
        pass
    return [x.strip() for x in languages_str.split(",") if x.strip()]


@app.post("/videos", response_model=CreateVideoResponse)
async def create_video_job(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    languages: Optional[str] = Form(default=None, description="JSON array or comma-separated list"),
):
    # Create job
    job = job_store.create()

    # Persist upload locally
    try:
        local_video_path = _save_upload_to_tmp(job.job_id, file)
    finally:
        try:
            await file.close()
        except Exception:
            pass

    langs = _parse_languages(languages)

    # Enqueue background processing
    background.add_task(_process_job, job.job_id, local_video_path, langs)

    return CreateVideoResponse(job_id=job.job_id, status=JobStatus.in_progress)


def _process_job(job_id: str, local_video_path: str, language_hints: List[str]):
    settings = get_settings()
    gcs = GCSClient()
    vi = VideoIntelClient(project=settings.gcp_project)

    try:
        # Upload original video to GCS
        base_prefix = f"jobs/{job_id}"
        input_blob_name = f"{base_prefix}/input/{os.path.basename(local_video_path)}"
        input_gcs_uri = gcs.upload_file(local_video_path, input_blob_name)

        # Request annotation
        result = vi.annotate_text(input_gcs_uri, language_hints=language_hints)

        # Process frames and masks, upload result JSON
        out = process_annotations(
            job_id=job_id,
            local_video_path=local_video_path,
            input_gcs_uri=input_gcs_uri,
            annotation_result=result,
            language_hints=language_hints,
        )

        job_store.set_result(job_id, gcs_uri=out["result_gcs_uri"], url=out.get("result_url"))
        job_store.set_status(job_id, JobStatus.completed, message="done")
    except Exception as e:
        job_store.set_status(job_id, JobStatus.error, message=str(e))
    finally:
        # Optionally cleanup local files here
        pass


@app.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobInfo(
        job_id=job.job_id,
        status=job.status,
        message=job.message,
        result_gcs_uri=job.result_gcs_uri,
        result_url=job.result_url,
    )


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

