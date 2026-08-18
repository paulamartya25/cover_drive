# ───────────────────────────────────────────────────────────────
# cover_drive_analyzer.py  –  Cricket Cover Drive Phase & Scoring
# ───────────────────────────────────────────────────────────────

import numpy as np

from config import (
    CONFIDENCE_THRESHOLD,
    IDEAL_ANGLES,
    PHASE_NAMES,
)
from angle_calculator import AngleCalculator


class CoverDriveAnalyzer:
    """Detects the current phase of a cover drive, scores posture,
    and generates coaching tips based on keypoints + angles."""

    def __init__(self):
        self.angle_calc = AngleCalculator()

    # ── main entry ────────────────────────────────────────────

    def analyze(self, keypoints: np.ndarray) -> dict:
        """
        Full analysis pipeline for one detected player.

        Args:
            keypoints: shape (17, 3).

        Returns:
            {
                "angles":   {name: value, ...},
                "phase":    str,
                "score":    int   (0-100),
                "ratings":  {name: "ideal" | "okay" | "bad", ...},
                "tips":     [str, ...],
            }
        """
        angles  = self.angle_calc.compute_all_angles(keypoints)
        phase   = self._detect_phase(keypoints, angles)
        ratings = self._rate_angles(angles)
        score   = self._compute_score(ratings)
        tips    = self._generate_tips(angles, ratings, phase)

        return {
            "angles":  angles,
            "phase":   phase,
            "score":   score,
            "ratings": ratings,
            "tips":    tips,
        }

    # ── phase detection ───────────────────────────────────────

    def _detect_phase(self, kp: np.ndarray, angles: dict) -> str:
        """
        Heuristic phase classification based on keypoint geometry.

        Rules (simplified — works well for single-frame analysis):
        1. **Follow-through** — wrists above shoulders (bat finishing high)
        2. **Downswing & Impact** — front knee well flexed AND wrists
           roughly at shoulder height
        3. **Backswing & Stride** — back elbow tightly flexed (< 100°)
        4. **Stance** — default / neutral position
        """

        def pt_y(idx):
            """Return y-coordinate if confident, else None."""
            if kp[idx][2] < CONFIDENCE_THRESHOLD:
                return None
            return float(kp[idx][1])

        l_wrist_y    = pt_y(9)
        r_wrist_y    = pt_y(10)
        l_shoulder_y = pt_y(5)
        r_shoulder_y = pt_y(6)

        front_knee = angles.get("Front Knee")
        back_elbow = angles.get("Back Elbow")

        # ── Follow-through: at least one wrist well above its shoulder
        if l_wrist_y and l_shoulder_y and r_wrist_y and r_shoulder_y:
            avg_wrist    = (l_wrist_y + r_wrist_y) / 2
            avg_shoulder = (l_shoulder_y + r_shoulder_y) / 2
            if avg_wrist < avg_shoulder - 40:          # wrists above shoulders (y-axis inverted)
                return PHASE_NAMES[3]                  # Follow-through

        # ── Downswing & Impact: front knee flexed
        if front_knee is not None and front_knee < 160:
            # Also check wrists are roughly at torso level
            if l_wrist_y and l_shoulder_y:
                if l_wrist_y > l_shoulder_y:           # wrists at or below shoulder
                    return PHASE_NAMES[2]              # Downswing & Impact

        # ── Backswing & Stride: back elbow tightly flexed
        if back_elbow is not None and back_elbow < 100:
            return PHASE_NAMES[1]                      # Backswing & Stride

        # ── Default
        return PHASE_NAMES[0]                          # Stance

    # ── angle rating ──────────────────────────────────────────

    @staticmethod
    def _rate_angles(angles: dict) -> dict:
        """Rate each angle as 'ideal', 'okay', or 'bad'."""
        ratings = {}
        for name, value in angles.items():
            if value is None:
                ratings[name] = "bad"    # missing keypoints → flag
                continue
            ideal = IDEAL_ANGLES.get(name)
            if ideal is None:
                ratings[name] = "okay"
                continue
            min_i, max_i, min_a, max_a = ideal
            if min_i <= value <= max_i:
                ratings[name] = "ideal"
            elif min_a <= value <= max_a:
                ratings[name] = "okay"
            else:
                ratings[name] = "bad"
        return ratings

    # ── scoring ───────────────────────────────────────────────

    @staticmethod
    def _compute_score(ratings: dict) -> int:
        """0-100 composite posture score based on angle ratings."""
        points = {"ideal": 100, "okay": 60, "bad": 20}
        values = [points.get(r, 0) for r in ratings.values()]
        if not values:
            return 0
        return int(round(sum(values) / len(values)))

    # ── coaching tips ─────────────────────────────────────────

    @staticmethod
    def _generate_tips(angles: dict, ratings: dict, phase: str) -> list[str]:
        """Return a list of short coaching feedback strings."""
        tips = []

        # Front Elbow
        fe = angles.get("Front Elbow")
        if ratings.get("Front Elbow") == "bad" and fe is not None:
            if fe < 120:
                tips.append("Extend your front elbow more — arm too bent for control.")
            else:
                tips.append("Front elbow hyper-extended — keep a soft bend.")

        # Back Elbow
        be = angles.get("Back Elbow")
        if ratings.get("Back Elbow") == "bad" and be is not None:
            if be > 130:
                tips.append("Tuck your back elbow in — keep it compact for power.")
            else:
                tips.append("Back elbow too tight — allow some natural swing.")

        # Front Knee
        fk = angles.get("Front Knee")
        if ratings.get("Front Knee") == "bad" and fk is not None:
            if fk > 175:
                tips.append("Bend your front knee more — too straight for stability.")
            elif fk < 110:
                tips.append("Front knee over-bent — you're crouching too low.")

        # Back Knee
        bk = angles.get("Back Knee")
        if ratings.get("Back Knee") == "bad" and bk is not None:
            tips.append("Straighten your back leg — pivot needs a stable base.")

        # Hip Angle
        ha = angles.get("Hip Angle")
        if ratings.get("Hip Angle") == "bad" and ha is not None:
            tips.append("Improve hip rotation — trunk needs more turn for power.")

        # Shoulder Line
        sl = angles.get("Shoulder Line")
        if ratings.get("Shoulder Line") == "bad" and sl is not None:
            tips.append("Level your shoulders — excessive tilt reduces bat control.")

        # Phase-specific
        if phase == PHASE_NAMES[0]:
            tips.append("You appear to be in Stance — relax and stay balanced.")

        if not tips:
            tips.append("Great form! Maintain this posture through the shot.")

        return tips
