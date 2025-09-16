from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..config import get_settings


Point = Tuple[int, int]


def _polygon_area(poly: List[Point]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _aabb(poly: List[Point]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _iou_aabb(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


def _dedup_same_timestamp(frames: List[Dict[str, Any]], iou_thresh: float) -> List[Dict[str, Any]]:
    # Group by exact timestamp value; keep one poly per time using NMS over AABB
    out: List[Dict[str, Any]] = []
    groups: Dict[float, List[Dict[str, Any]]] = {}
    for fr in frames:
        t = float(fr.get("t", 0.0))
        groups.setdefault(t, []).append(fr)
    for t, group in groups.items():
        # Simple NMS: sort by area desc, suppress high IoU
        with_boxes = [(fr, _aabb(fr.get("poly", [])), _polygon_area(fr.get("poly", []))) for fr in group]
        with_boxes.sort(key=lambda x: x[2], reverse=True)
        kept: List[Tuple[Dict[str, Any], Tuple[int, int, int, int], float]] = []
        for fr, box, area in with_boxes:
            suppress = False
            for kfr, kbox, _ in kept:
                if _iou_aabb(box, kbox) >= iou_thresh:
                    suppress = True
                    break
            if not suppress:
                kept.append((fr, box, area))
        out.extend([fr for fr, _, _ in kept])
    # Preserve chronological order
    out.sort(key=lambda fr: float(fr.get("t", 0.0)))
    return out


def postprocess_detections(raw_items: List[Dict[str, Any]], *, width: int, height: int) -> List[Dict[str, Any]]:
    """Apply light validation and clean-up to raw detections.

    - Drop segments below MIN_CONFIDENCE (if present)
    - Remove frames with polygon area below MIN_AREA_RATIO of the frame
    - NMS within the same timestamp per segment (AABB IoU >= NMS_IOU)
    - Drop segments with no frames left
    - Optionally drop very short segments (MIN_SEGMENT_DURATION_SEC > 0)
    """
    s = get_settings()
    min_conf = float(getattr(s, "min_confidence", 0.0))
    min_area_ratio = float(getattr(s, "min_area_ratio", 0.0))
    nms_iou = float(getattr(s, "nms_iou_threshold", 0.7))
    min_seg_dur = float(getattr(s, "min_segment_duration_sec", 0.0))

    frame_area = float(max(1, width * height))

    cleaned_items: List[Dict[str, Any]] = []
    for it in raw_items:
        text = it.get("text", "")
        out_segs: List[Dict[str, Any]] = []
        for seg in it.get("segments", []):
            conf = seg.get("confidence")
            if conf is not None and conf < min_conf:
                continue

            # area filter
            good_frames: List[Dict[str, Any]] = []
            for fr in seg.get("frames", []):
                poly = fr.get("poly", [])
                if not poly:
                    continue
                area = _polygon_area(poly)
                if min_area_ratio > 0.0 and (area / frame_area) < min_area_ratio:
                    continue
                good_frames.append(fr)

            if not good_frames:
                continue

            # dedup by timestamp within segment
            good_frames = _dedup_same_timestamp(good_frames, iou_thresh=nms_iou)

            start = seg.get("start")
            end = seg.get("end")
            if min_seg_dur > 0.0 and (start is not None and end is not None):
                if (end - start) < min_seg_dur:
                    # drop too short segments
                    continue

            out_segs.append({
                "start": start,
                "end": end,
                "confidence": conf,
                "frames": good_frames,
            })

        if out_segs:
            cleaned_items.append({"text": text.strip(), "segments": out_segs})

    return cleaned_items

