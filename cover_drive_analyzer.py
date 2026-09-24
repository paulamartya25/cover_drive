# ───────────────────────────────────────────────────────────────
# cover_drive_analyzer.py  –  Cricket Posture Analysis Engine
# ───────────────────────────────────────────────────────────────

import math
import collections
import numpy as np

from config import (
    CONFIDENCE_THRESHOLD,
    IDEAL_ANGLES,
    PHASE_NAMES,
    HEIGHT_ANGLE_MAP,
    JOINT_QUALITY_PARAMS,
    HEIGHT_QUALITY_ADJUSTMENTS,
    SHOT_SIGNATURES,
)
from angle_calculator import AngleCalculator
from height_estimator import estimate_height_category


class CoverDriveAnalyzer:
    """
    Full biomechanical analysis engine.

    Features:
    1. Height-Adaptive Angle Thresholds   — adjusts ideal angles for SHORT/MEDIUM/TALL
    2. Generalized Gaussian Quality Score — smooth formula, not just pass/fail buckets
    3. Shot Classification                — Cover Drive / Pull Shot / Sweep / Defensive Push
    4. Temporal Smoothing                 — rolling average over last 10 frames (no flicker)
    5. Phase Detection                    — Stance / Backswing / Impact / Follow-through
    """

    SMOOTH_WINDOW = 10

    def __init__(self):
        self.angle_calc      = AngleCalculator()
        self._score_queue    = collections.deque(maxlen=self.SMOOTH_WINDOW)
        self._last_shot_type = ""   # memory: holds shot name through Follow-through

    # ── Main Entry ────────────────────────────────────────────

    def analyze(self, keypoints: np.ndarray, player_meta: dict = None) -> dict:
        """
        Full analysis pipeline for one frame.

        Shot Classification Logic:
          - ONLY classify during "Downswing & Impact" phase.
          - HOLD the last confirmed shot name during Follow-through.
          - RESET the shot name during Stance (fresh shot incoming).
          This prevents misclassification during body-recovery frames.
        """
        # Step 1: Estimate player height for adaptive thresholds
        height_cat  = estimate_height_category(keypoints)
        ideal_table = HEIGHT_ANGLE_MAP.get(height_cat, IDEAL_ANGLES)

        # Step 2: Compute joint angles
        angles = self.angle_calc.compute_all_angles(keypoints)

        # Step 3: Detect current shot phase
        phase = self._detect_phase(keypoints, angles)

        # Step 4: Rate angles using height-adaptive table
        ratings = self._rate_angles(angles, ideal_table)

        # Step 5: Generalized Gaussian quality score
        raw_score = self._gaussian_score(angles, height_cat)

        # Step 6: Temporal smoothing
        self._score_queue.append(raw_score)
        smooth_score = int(round(sum(self._score_queue) / len(self._score_queue)))

        # Step 7: Phase-locked shot classification
        locked = player_meta.get("locked_on_striker", True) if player_meta else True

        if phase == PHASE_NAMES[0]:
            # Stance → reset: new shot is starting
            self._last_shot_type = ""
            shot_type = ""

        elif phase == PHASE_NAMES[2] and locked:
            # Downswing & Impact → CLASSIFY NOW (only valid moment)
            new_shot = self._classify_shot(angles)
            if new_shot:
                self._last_shot_type = new_shot   # commit to memory
            shot_type = self._last_shot_type

        else:
            # Backswing or Follow-through → HOLD last confirmed shot name
            # Do NOT re-classify: body angles are wrong during these phases
            shot_type = self._last_shot_type if locked else ""

        # Step 8: Coaching tips
        tips = self._generate_tips(angles, ratings, phase)

        return {
            "angles":          angles,
            "phase":           phase,
            "raw_score":       raw_score,
            "score":           smooth_score,
            "ratings":         ratings,
            "tips":            tips,
            "shot_type":       shot_type,
            "height_category": height_cat,
        }

    # ── Phase Detection ───────────────────────────────────────

    def _detect_phase(self, kp: np.ndarray, angles: dict) -> str:
        def pt_y(idx):
            if kp[idx][2] < CONFIDENCE_THRESHOLD:
                return None
            return float(kp[idx][1])

        l_wrist_y    = pt_y(9)
        r_wrist_y    = pt_y(10)
        l_shoulder_y = pt_y(5)
        r_shoulder_y = pt_y(6)

        front_knee  = angles.get("Front Knee")
        back_elbow  = angles.get("Back Elbow")
        front_elbow = angles.get("Front Elbow")

        # Follow-through: wrists well above shoulders
        if l_wrist_y and l_shoulder_y and r_wrist_y and r_shoulder_y:
            avg_wrist    = (l_wrist_y + r_wrist_y) / 2
            avg_shoulder = (l_shoulder_y + r_shoulder_y) / 2
            if avg_wrist < avg_shoulder - 40:
                return PHASE_NAMES[3]

        # Downswing & Impact: wrists low + elbow extended OR knee bent
        is_wrists_low = (l_wrist_y and l_shoulder_y and l_wrist_y > l_shoulder_y)
        if is_wrists_low:
            if front_elbow is not None and front_elbow > 130:
                return PHASE_NAMES[2]
            elif front_knee is not None and front_knee < 165:
                return PHASE_NAMES[2]

        # Backswing: back elbow tightly flexed
        if back_elbow is not None and back_elbow < 100:
            return PHASE_NAMES[1]

        return PHASE_NAMES[0]

    # ── Height-Adaptive Angle Rating ──────────────────────────

    @staticmethod
    def _rate_angles(angles: dict, ideal_table: dict) -> dict:
        """Rate each angle as 'ideal', 'okay', or 'bad' using the height-adapted table."""
        ratings = {}
        for name, value in angles.items():
            if value is None:
                ratings[name] = "bad"
                continue
            ideal = ideal_table.get(name)
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

    # ── Generalized Gaussian Quality Score ────────────────────

    @staticmethod
    def _gaussian_score(angles: dict, height_cat: str) -> int:
        """
        The Generalized Shot Quality Formula (works for ALL players):

            score_per_joint = exp(-0.5 * ((angle - center) / sigma)^2) * 100

        Where:
          - center: the anatomically perfect angle for that joint
          - sigma:  tolerance window (how far from center before score drops)

        Height adjustments shift the 'center' value for knee/hip joints
        to match the player's natural proportions.

        Final score = weighted mean of all visible joint scores.
        Weights: Front Elbow & Front Knee are most critical (weight=2).
        """
        adjustments = HEIGHT_QUALITY_ADJUSTMENTS.get(height_cat, {})
        weights     = {
            "Front Elbow":   2.0,
            "Back Elbow":    1.0,
            "Front Knee":    2.0,
            "Back Knee":     1.0,
            "Hip Angle":     1.5,
            "Shoulder Line": 1.5,
        }

        total_score  = 0.0
        total_weight = 0.0

        for name, params in JOINT_QUALITY_PARAMS.items():
            value = angles.get(name)
            if value is None:
                continue
            center = params["center"] + adjustments.get(name, 0)
            sigma  = params["sigma"]
            w      = weights.get(name, 1.0)

            # Gaussian scoring function
            joint_score = math.exp(-0.5 * ((value - center) / sigma) ** 2) * 100
            total_score  += joint_score * w
            total_weight += w

        if total_weight == 0:
            return 0
        return int(round(total_score / total_weight))

    # ── Shot Classification ────────────────────────────────────

    @staticmethod
    def _classify_shot(angles: dict) -> str:
        """
        Match observed angles against known shot signatures.

        Scoring:
          - Each shot accumulates one point per matching rule.
          - Requires at least 2 matches (or all rules if < 2 rules exist).
          - Tiebreak: SHOT_SIGNATURES insertion order is the priority list
            (Cover Drive > Pull Shot > Sweep Shot > Defensive Push).
            If two shots tie on match count, the higher-priority one wins.

        Returns the shot name, or "" if nothing matches confidently.
        """
        # Priority order follows SHOT_SIGNATURES dict order (Python 3.7+)
        PRIORITY = {name: i for i, name in enumerate(SHOT_SIGNATURES)}

        best_shot    = ""
        best_matches = 0
        best_priority = 999

        for shot_name, rules in SHOT_SIGNATURES.items():
            matched = 0
            total   = len(rules)
            for angle_name, (lo, hi) in rules.items():
                val = angles.get(angle_name)
                if val is not None and lo <= val <= hi:
                    matched += 1
            min_required = min(2, total)
            if matched >= min_required:
                priority = PRIORITY[shot_name]
                # Accept if: more matches, OR same matches but higher priority
                if matched > best_matches or (matched == best_matches and priority < best_priority):
                    best_matches  = matched
                    best_shot     = shot_name
                    best_priority = priority

        return best_shot   # empty string "" if no shot matched confidently


    # ── Coaching Tips ─────────────────────────────────────────

    @staticmethod
    def _generate_tips(angles: dict, ratings: dict, phase: str) -> list:
        tips = []

        fe = angles.get("Front Elbow")
        if ratings.get("Front Elbow") == "bad" and fe is not None:
            if fe < 120:
                tips.append("Extend your front elbow more — arm too bent for control.")
            else:
                tips.append("Front elbow hyper-extended — keep a soft bend.")

        be = angles.get("Back Elbow")
        if ratings.get("Back Elbow") == "bad" and be is not None:
            if be > 130:
                tips.append("Tuck your back elbow in — keep it compact for power.")
            else:
                tips.append("Back elbow too tight — allow some natural swing.")

        fk = angles.get("Front Knee")
        if ratings.get("Front Knee") == "bad" and fk is not None:
            if fk > 175:
                tips.append("Bend your front knee more — too straight for stability.")
            elif fk < 110:
                tips.append("Front knee over-bent — you're crouching too low.")

        bk = angles.get("Back Knee")
        if ratings.get("Back Knee") == "bad" and bk is not None:
            tips.append("Straighten your back leg — pivot needs a stable base.")

        ha = angles.get("Hip Angle")
        if ratings.get("Hip Angle") == "bad" and ha is not None:
            tips.append("Improve hip rotation — trunk needs more turn for power.")

        sl = angles.get("Shoulder Line")
        if ratings.get("Shoulder Line") == "bad" and sl is not None:
            tips.append("Level your shoulders — excessive tilt reduces bat control.")

        if phase == PHASE_NAMES[0]:
            tips.append("You appear to be in Stance — relax and stay balanced.")

        if not tips:
            tips.append("Great form! Maintain this posture through the shot.")

        return tips
