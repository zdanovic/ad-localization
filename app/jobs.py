from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from .schemas import JobStatus


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.in_progress
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    message: Optional[str] = None
    result_gcs_uri: Optional[str] = None
    result_url: Optional[str] = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()

    def create(self) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id, status=JobStatus.in_progress)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def set_status(self, job_id: str, status: JobStatus, message: Optional[str] = None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = status
            job.message = message
            job.updated_at = time.time()

    def set_result(self, job_id: str, gcs_uri: str, url: Optional[str]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.result_gcs_uri = gcs_uri
            job.result_url = url
            job.updated_at = time.time()


job_store = JobStore()

