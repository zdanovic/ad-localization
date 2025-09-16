from __future__ import annotations

from typing import List

from google.cloud import videointelligence_v1 as vi


class VideoIntelClient:
    def __init__(self, project: str | None = None) -> None:
        self.client = vi.VideoIntelligenceServiceClient()

    def annotate_text(self, gcs_uri: str, language_hints: List[str]):
        features = [vi.Feature.TEXT_DETECTION]
        context = vi.VideoContext(
            text_detection_config=vi.TextDetectionConfig(language_hints=language_hints or [])
        )
        operation = self.client.annotate_video(
            request={
                "features": features,
                "input_uri": gcs_uri,
                "video_context": context,
            }
        )
        return operation.result()

