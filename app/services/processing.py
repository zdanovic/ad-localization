from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ..config import get_settings
from ..schemas import DetectionFrame, DetectionItem, DetectionSegment, ResultPayload, VideoInfo
from ..utils.video_utils import VideoReader
from .gcs import GCSClient
from .postprocess import postprocess_detections


def _time_offset_to_seconds(offset: Any) -> float:
    # offset is a google.protobuf.duration_pb2.Duration or similar with seconds and nanos
    seconds = getattr(offset, "seconds", 0)
    nanos = getattr(offset, "nanos", 0)
    return float(seconds) + float(nanos) / 1e9


def _vertices_to_points(vertices: List[Any], width: int, height: int) -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    for v in vertices:
        # Support attributes or dict-like
        x = getattr(v, "x", None)
        y = getattr(v, "y", None)
        if x is None and isinstance(v, dict):
            x = v.get("x")
            y = v.get("y")
        if x is None or y is None:
            continue
        # Coordinates are normalized [0,1]
        px = max(0, min(int(round(float(x) * width)), width - 1))
        py = max(0, min(int(round(float(y) * height)), height - 1))
        pts.append((px, py))
    return pts


def _mask_from_polygon(width: int, height: int, polygon: List[Tuple[int, int]], padding_px: int = 0) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if polygon:
        pts = np.array(polygon, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], color=255)
        if padding_px and padding_px > 0:
            k = int(max(1, padding_px))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, 2 * k + 1))
            mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def process_annotations(
    job_id: str,
    local_video_path: str,
    input_gcs_uri: str,
    annotation_result: Any,
    language_hints: List[str] | None,
) -> Dict[str, Any]:
    settings = get_settings()
    gcs = GCSClient()

    # Prepare readers and meta
    vr = VideoReader(local_video_path)
    width, height = vr.meta.width, vr.meta.height

    detections: List[DetectionItem] = []
    base_prefix = f"jobs/{job_id}"
    masks_prefix = f"{base_prefix}/masks"

    # Video intelligence response structure:
    # annotation_result.annotation_results[0].text_annotations -> List[TextAnnotation]
    annotation_results = getattr(annotation_result, "annotation_results", None) or []
    if not annotation_results:
        # Some SDKs return a simple list
        annotation_results = [annotation_result]

    # First pass: parse raw detections from annotation into a lightweight structure
    raw_items: List[Dict[str, Any]] = []
    for ar in annotation_results:
        text_annotations = getattr(ar, "text_annotations", [])
        for ta in text_annotations:
            text = getattr(ta, "text", "")
            raw_segs: List[Dict[str, Any]] = []
            for seg in getattr(ta, "segments", []):
                seg_start = _time_offset_to_seconds(getattr(seg, "segment").start_time_offset) if getattr(seg, "segment", None) else None
                seg_end = _time_offset_to_seconds(getattr(seg, "segment").end_time_offset) if getattr(seg, "segment", None) else None
                seg_conf = getattr(seg, "confidence", None)

                raw_frames: List[Dict[str, Any]] = []
                for fr in getattr(seg, "frames", []):
                    t = _time_offset_to_seconds(getattr(fr, "time_offset", 0))
                    rbb = getattr(fr, "rotated_bounding_box", None)
                    vertices = getattr(rbb, "vertices", []) if rbb is not None else []
                    polygon = _vertices_to_points(vertices, width, height)
                    if not polygon:
                        continue
                    raw_frames.append({"t": t, "poly": polygon})

                raw_segs.append({
                    "start": seg_start,
                    "end": seg_end,
                    "confidence": seg_conf,
                    "frames": raw_frames,
                })
            raw_items.append({"text": text, "segments": raw_segs})

    # Second pass: apply validation and post-processing filters
    processed_items = postprocess_detections(raw_items, width=width, height=height)

    # Third pass: generate masks, upload and build schema DTOs
    for item_idx, it in enumerate(processed_items):
        text = it.get("text", "")
        segments_out: List[DetectionSegment] = []
        for seg_idx, seg in enumerate(it.get("segments", [])):
            seg_start = seg.get("start")
            seg_end = seg.get("end")
            seg_conf = seg.get("confidence")
            frames_out: List[DetectionFrame] = []
            for frame_idx, fr in enumerate(seg.get("frames", [])):
                t = float(fr.get("t", 0.0))
                polygon = fr.get("poly", [])

                # Extract frame (ensures bounds/timing are valid and reusable later if needed)
                _ = vr.frame_at_time(t)
                # Create mask with optional padding
                mask = _mask_from_polygon(width, height, polygon, padding_px=getattr(settings, "mask_padding_px", 0))

                # Save mask to local temp then upload
                local_mask_dir = os.path.join(settings.tmp_dir, job_id, "masks")
                os.makedirs(local_mask_dir, exist_ok=True)
                local_mask_path = os.path.join(local_mask_dir, f"{seg_idx:04d}_{frame_idx:04d}.png")

                cv2.imwrite(local_mask_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])

                mask_blob_name = f"{masks_prefix}/{seg_idx:04d}_{frame_idx:04d}.png"
                mask_gcs_uri = gcs.upload_file(local_mask_path, mask_blob_name, content_type="image/png")
                mask_url = gcs.signed_url(mask_blob_name)

                frames_out.append(
                    DetectionFrame(
                        time_offset_sec=t,
                        bounding_poly=[{"x": x, "y": y} for (x, y) in polygon],
                        mask_gcs_uri=mask_gcs_uri,
                        mask_url=mask_url,
                    )
                )

            if frames_out:
                segments_out.append(
                    DetectionSegment(
                        start_time_sec=seg_start,
                        end_time_sec=seg_end,
                        confidence=seg_conf,
                        frames=frames_out,
                    )
                )

        if segments_out:
            detections.append(DetectionItem(text=text, segments=segments_out))

    vr.release()

    video_info = VideoInfo(
        filename=os.path.basename(local_video_path),
        gcs_uri=input_gcs_uri,
        duration_sec=vr.meta.duration_sec,
        fps=vr.meta.fps,
        width=width,
        height=height,
    )

    result = ResultPayload(
        job_id=job_id,
        language_hints=language_hints or [],
        video=video_info,
        detections=detections,
    )

    # Upload result JSON
    result_blob_name = f"{base_prefix}/result.json"
    result_gcs_uri = gcs.upload_json(result.model_dump(), result_blob_name)
    result_url = gcs.signed_url(result_blob_name)

    return {"result_gcs_uri": result_gcs_uri, "result_url": result_url}
