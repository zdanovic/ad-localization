from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class VideoMeta:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


class VideoReader:
    def __init__(self, path: str) -> None:
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        self.meta = self._read_meta()

    def _read_meta(self) -> VideoMeta:
        fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 25.0
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if frame_count and fps else 0.0
        return VideoMeta(fps=fps, frame_count=frame_count, width=width, height=height, duration_sec=duration)

    def frame_at_time(self, t_sec: float) -> np.ndarray:
        # Clamp time to valid range
        t = max(0.0, min(t_sec, max(0.0, self.meta.duration_sec - 1e-3)))
        idx = int(round(t * self.meta.fps))
        idx = max(0, min(idx, max(0, self.meta.frame_count - 1)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame at {t_sec}s (idx={idx}) from {self.path}")
        return frame

    def release(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass

