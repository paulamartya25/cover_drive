# ───────────────────────────────────────────────────────────────
# angle_calculator.py  –  Joint Angle Computation
# ───────────────────────────────────────────────────────────────

import math
import numpy as np

from config import CONFIDENCE_THRESHOLD


class AngleCalculator:
    """Computes joint angles from detected keypoints for biomechanical analysis."""

    # ── generic angle math ────────────────────────────────────

    @staticmethod
    def calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
        """
        Calculate the angle (in degrees) at vertex **b** formed by
        the line segments a→b and b→c.

        Args:
            a, b, c: each is (x, y).
        Returns:
            Angle in degrees [0, 180].
        """
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)

        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        return round(angle, 1)

    @staticmethod
    def angle_to_horizontal(a: tuple, b: tuple) -> float:
        """
        Angle (degrees) of the line a→b relative to the horizontal axis.
        Returns a value in [0, 90].
        """
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle > 90:
            angle = 180 - angle
        return round(angle, 1)

    # ── cricket-specific joint angles ─────────────────────────

    def compute_all_angles(self, keypoints: np.ndarray) -> dict:
        """
        Compute the 6 key biomechanical angles for cricket cover-drive
        analysis.

        Args:
            keypoints: shape (17, 3) — x, y, conf per COCO keypoint.

        Returns:
            Dict mapping angle name → value (float degrees) or None
            if required keypoints are missing.
        """
        angles = {}

        # Helper: extract (x,y) only if confidence is sufficient
        def pt(idx):
            if keypoints[idx][2] < CONFIDENCE_THRESHOLD:
                return None
            return (float(keypoints[idx][0]), float(keypoints[idx][1]))

        # ── Left-side points ──
        l_shoulder = pt(5)
        l_elbow    = pt(7)
        l_wrist    = pt(9)
        l_hip      = pt(11)
        l_knee     = pt(13)
        l_ankle    = pt(15)

        # ── Right-side points ──
        r_shoulder = pt(6)
        r_elbow    = pt(8)
        r_wrist    = pt(10)
        r_hip      = pt(12)
        r_knee     = pt(14)
        r_ankle    = pt(16)

        # Determine batting side heuristic:
        # In a right-handed cover drive the LEFT side is the "front" side
        # (left foot leads toward the bowler).  We try left-first; if not
        # visible we fall back to right.

        # 1. Front Elbow  (leading arm — left for RHB)
        angles["Front Elbow"] = self._safe_angle(l_shoulder, l_elbow, l_wrist) \
                                 or self._safe_angle(r_shoulder, r_elbow, r_wrist)

        # 2. Back Elbow  (trailing arm — right for RHB)
        angles["Back Elbow"] = self._safe_angle(r_shoulder, r_elbow, r_wrist) \
                                or self._safe_angle(l_shoulder, l_elbow, l_wrist)

        # 3. Front Knee
        angles["Front Knee"] = self._safe_angle(l_hip, l_knee, l_ankle) \
                                or self._safe_angle(r_hip, r_knee, r_ankle)

        # 4. Back Knee
        angles["Back Knee"] = self._safe_angle(r_hip, r_knee, r_ankle) \
                               or self._safe_angle(l_hip, l_knee, l_ankle)

        # 5. Hip Angle  (shoulder → hip → knee on front side)
        angles["Hip Angle"] = self._safe_angle(l_shoulder, l_hip, l_knee) \
                               or self._safe_angle(r_shoulder, r_hip, r_knee)

        # 6. Shoulder Line  (angle of the shoulder line to horizontal)
        if l_shoulder and r_shoulder:
            angles["Shoulder Line"] = self.angle_to_horizontal(l_shoulder, r_shoulder)
        else:
            angles["Shoulder Line"] = None

        return angles

    # ── internal helpers ──────────────────────────────────────

    def _safe_angle(self, a, b, c) -> float | None:
        """Return angle at b if all three points are valid, else None."""
        if a is None or b is None or c is None:
            return None
        return self.calculate_angle(a, b, c)
