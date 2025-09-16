from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class Poly:
    points: List[Tuple[int, int]]
    label: str | None = None


def _read_json(path_or_url: str) -> dict:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        import urllib.request

        with urllib.request.urlopen(path_or_url) as r:  # nosec - signed url expected
            return json.loads(r.read().decode("utf-8"))
    elif path_or_url.startswith("gs://"):
        # Download to temp and read
        bucket, blob = _parse_gs_uri(path_or_url)
        tmp = tempfile.mktemp(suffix=".json")
        _gcs_download(bucket, blob, tmp)
        with open(tmp, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(path_or_url, "r", encoding="utf-8") as f:
            return json.load(f)


def _parse_gs_uri(uri: str) -> Tuple[str, str]:
    m = re.match(r"^gs://([^/]+)/(.+)$", uri)
    if not m:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return m.group(1), m.group(2)


def _gcs_download(bucket: str, blob_name: str, dest_path: str) -> None:
    # Lazy import so script works without GCS for local files
    from google.cloud import storage

    client = storage.Client()
    b = client.bucket(bucket)
    bl = b.blob(blob_name)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    bl.download_to_filename(dest_path)


def _collect_overlays(result: dict, width: int, height: int, fps: float) -> Dict[int, List[Poly]]:
    overlays: Dict[int, List[Poly]] = {}
    detections = result.get("detections", [])
    for item in detections:
        label = item.get("text")
        for seg in item.get("segments", []):
            for fr in seg.get("frames", []):
                t = float(fr.get("time_offset_sec", 0.0))
                idx = int(round(t * fps))
                pts = [(int(p.get("x")), int(p.get("y"))) for p in fr.get("bounding_poly", [])]
                if not pts:
                    continue
                poly = Poly(points=pts, label=label)
                overlays.setdefault(idx, []).append(poly)
    return overlays


def _draw_overlays(frame: np.ndarray, polys: List[Poly], alpha: float, color: Tuple[int, int, int], thickness: int, outline_only: bool) -> np.ndarray:
    out = frame.copy()
    if not polys:
        return out
    overlay = frame.copy()
    for poly in polys:
        pts = np.array(poly.points, dtype=np.int32).reshape((-1, 1, 2))
        if not outline_only:
            cv2.fillPoly(overlay, [pts], color=color)
        # outline
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=thickness)

        if poly.label:
            # Put label near the first point
            x, y = poly.points[0]
            cv2.putText(
                overlay,
                poly.label,
                (x + 3, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
    # blend
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a preview video with detected text masks/polygons overlaid")
    ap.add_argument("--result", required=True, help="Path/URL/GCS URI to result.json produced by the API")
    ap.add_argument("--out", required=True, help="Output path for preview mp4")
    ap.add_argument("--alpha", type=float, default=0.35, help="Overlay transparency [0..1]")
    ap.add_argument("--thickness", type=int, default=2, help="Polygon outline thickness")
    ap.add_argument("--outline-only", action="store_true", help="Do not fill polygons, only draw outlines")
    ap.add_argument("--color", default="0,0,255", help="Overlay BGR color, e.g. '0,0,255' (red)")
    args = ap.parse_args()

    color = tuple(int(x) for x in args.color.split(","))  # B,G,R

    result = _read_json(args.result)
    video_uri = result.get("video", {}).get("gcs_uri")
    if not video_uri:
        print("result.video.gcs_uri missing in result json", file=sys.stderr)
        sys.exit(2)

    # Download video locally
    bucket, blob = _parse_gs_uri(video_uri)
    tmp_dir = tempfile.mkdtemp(prefix="marked_video_")
    local_video = os.path.join(tmp_dir, os.path.basename(blob))
    _gcs_download(bucket, blob, local_video)

    cap = cv2.VideoCapture(local_video)
    if not cap.isOpened():
        print(f"Failed to open video: {local_video}", file=sys.stderr)
        sys.exit(3)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    overlays = _collect_overlays(result, width, height, fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Failed to open writer: {args.out}", file=sys.stderr)
        sys.exit(4)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        polys = overlays.get(idx, [])
        if polys:
            frame = _draw_overlays(frame, polys, alpha=args.alpha, color=color, thickness=args.thickness, outline_only=args.outline_only)
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    print(args.out)


if __name__ == "__main__":
    main()

