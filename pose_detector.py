# ───────────────────────────────────────────────────────────────
# pose_detector.py  –  YOLOv8 Pose Estimation Wrapper
# ───────────────────────────────────────────────────────────────

import numpy as np
from ultralytics import YOLO

from config import CONFIDENCE_THRESHOLD, MODEL_NAME, INPUT_SIZE, KEYPOINT_NAMES


class PoseDetector:
    """Wraps the YOLOv8-pose model for person detection + keypoint extraction."""

    def __init__(self, model_path: str = MODEL_NAME, conf_thresh: float = CONFIDENCE_THRESHOLD):
        """
        Load the YOLOv8-pose model.

        Args:
            model_path:  Name or path of the .pt weights file.
                         On first run the weights are auto-downloaded.
            conf_thresh: Minimum keypoint confidence to treat as valid.
        """
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    # ── public API ────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run pose estimation on a single BGR frame.

        Returns:
            List of dicts, one per detected person:
            {
                "bbox":       (x1, y1, x2, y2, conf),
                "keypoints":  np.ndarray  shape (17, 3)  – x, y, conf per kp,
                "bbox_area":  float,
            }
        """
        results = self.model(frame, imgsz=INPUT_SIZE, verbose=False)
        detections = []

        for result in results:
            if result.keypoints is None:
                continue

            boxes = result.boxes
            kps   = result.keypoints

            for i in range(len(boxes)):
                # Bounding box
                xyxy = boxes.xyxy[i].cpu().numpy()          # (4,)
                box_conf = float(boxes.conf[i].cpu().numpy())
                x1, y1, x2, y2 = xyxy
                bbox = (float(x1), float(y1), float(x2), float(y2), box_conf)

                # Keypoints – shape (17, 3) with x, y, conf
                kp_data = kps.data[i].cpu().numpy()         # (17, 3)

                detections.append({
                    "bbox":      bbox,
                    "keypoints": kp_data,
                    "bbox_area": float((x2 - x1) * (y2 - y1)),
                })

        return detections

    def get_primary_player(self, detections: list[dict]) -> dict | None:
        """Return the detection with the largest bounding box (assumed to be
        the main player closest to camera)."""
        if not detections:
            return None
        return max(detections, key=lambda d: d["bbox_area"])

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def keypoint_valid(kp_row: np.ndarray, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
        """Check if a single keypoint (x, y, conf) has sufficient confidence."""
        return float(kp_row[2]) >= threshold

    @staticmethod
    def get_point(keypoints: np.ndarray, index: int) -> tuple[int, int] | None:
        """Return (x, y) as ints for keypoint *index*, or None if below threshold."""
        if float(keypoints[index][2]) < CONFIDENCE_THRESHOLD:
            return None
        return (int(keypoints[index][0]), int(keypoints[index][1]))

    @staticmethod
    def get_keypoint_name(index: int) -> str:
        return KEYPOINT_NAMES.get(index, f"KP-{index}")
