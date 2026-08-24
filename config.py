# ───────────────────────────────────────────────────────────────
# config.py  –  Constants for Cricket Cover Drive Pose Analysis
# ───────────────────────────────────────────────────────────────

# ── 17 COCO Keypoint Indices ──────────────────────────────────
KEYPOINT_NAMES = {
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "Left Ear",
    4: "Right Ear",
    5: "Left Shoulder",
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Hip",
    12: "Right Hip",
    13: "Left Knee",
    14: "Right Knee",
    15: "Left Ankle",
    16: "Right Ankle",
}

# Short labels for on-screen display
KEYPOINT_SHORT = {
    0: "NOS", 1: "LEY", 2: "REY", 3: "LEA", 4: "REA",
    5: "LSH", 6: "RSH", 7: "LEL", 8: "REL", 9: "LWR", 10: "RWR",
    11: "LHP", 12: "RHP", 13: "LKN", 14: "RKN", 15: "LAN", 16: "RAN",
}

# ── Skeleton Connection Pairs ─────────────────────────────────
# Each tuple = (keypoint_A_index, keypoint_B_index)
SKELETON_PAIRS = [
    # Face
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Upper body
    (5, 6),   # shoulder → shoulder
    (5, 7),   # L shoulder → L elbow
    (7, 9),   # L elbow → L wrist
    (6, 8),   # R shoulder → R elbow
    (8, 10),  # R elbow → R wrist
    # Torso
    (5, 11),  # L shoulder → L hip
    (6, 12),  # R shoulder → R hip
    (11, 12), # hip → hip
    # Lower body
    (11, 13), # L hip → L knee
    (13, 15), # L knee → L ankle
    (12, 14), # R hip → R knee
    (14, 16), # R knee → R ankle
]

# ── Color Palette (BGR for OpenCV) ────────────────────────────
COLORS = {
    "face":      (255, 200, 55),   # light blue
    "arm_left":  (0, 255, 128),    # green
    "arm_right": (0, 200, 255),    # orange-yellow
    "torso":     (255, 255, 0),    # cyan
    "leg_left":  (255, 0, 128),    # magenta
    "leg_right": (128, 0, 255),    # purple
    "keypoint":  (0, 255, 255),    # yellow
    "bbox":      (255, 128, 0),    # blue-ish
    "text_bg":   (40, 40, 40),     # dark grey
    "ideal":     (0, 220, 0),      # green  — ideal range
    "okay":      (0, 220, 220),    # yellow — acceptable
    "bad":       (0, 0, 255),      # red    — needs correction
    "hud_bg":    (30, 30, 30),     # dashboard background
    "hud_border":(100, 100, 100),  # dashboard border
}

# Map each skeleton pair to a color region
PAIR_COLORS = {
    (0, 1): "face", (0, 2): "face", (1, 3): "face", (2, 4): "face",
    (5, 6): "torso",
    (5, 7): "arm_left", (7, 9): "arm_left",
    (6, 8): "arm_right", (8, 10): "arm_right",
    (5, 11): "torso", (6, 12): "torso", (11, 12): "torso",
    (11, 13): "leg_left", (13, 15): "leg_left",
    (12, 14): "leg_right", (14, 16): "leg_right",
}

# ── Detection Settings ────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.35     # lowered from 0.5 to account for batting pads/helmets
MODEL_NAME = "yolov8n-pose.pt"  # nano model (fast); change to 's'/'m' for accuracy
INPUT_SIZE = 640                # inference resolution

# ── Cricket Cover Drive — Ideal Angle Ranges (degrees) ────────
# Format: (min_ideal, max_ideal, min_acceptable, max_acceptable)
# These are the MEDIUM height defaults (used as base)
IDEAL_ANGLES = {
    "Front Elbow":    (140, 180, 120, 180),
    "Back Elbow":     (60, 110, 40, 130),
    "Front Knee":     (130, 165, 110, 175),
    "Back Knee":      (140, 180, 120, 180),
    "Hip Angle":      (80, 140, 60, 160),
    "Shoulder Line":  (0, 35, 0, 50),
}

# ── Height-Adaptive Angle Thresholds ──────────────────────────
# Taller players have a naturally wider stance → more extended knees
# Shorter players crouch more → more flexed knees
IDEAL_ANGLES_TALL = {
    "Front Elbow":    (140, 180, 120, 180),
    "Back Elbow":     (60, 110, 40, 130),
    "Front Knee":     (140, 175, 120, 180),   # more extended for tall players
    "Back Knee":      (150, 180, 130, 180),
    "Hip Angle":      (90, 150, 70, 165),
    "Shoulder Line":  (0, 30, 0, 45),
}

IDEAL_ANGLES_MEDIUM = IDEAL_ANGLES  # same as the base

IDEAL_ANGLES_SHORT = {
    "Front Elbow":    (140, 180, 120, 180),
    "Back Elbow":     (60, 110, 40, 130),
    "Front Knee":     (115, 155, 100, 170),   # more flexed for shorter players
    "Back Knee":      (125, 165, 110, 175),
    "Hip Angle":      (70, 130, 50, 150),
    "Shoulder Line":  (0, 40, 0, 55),
}

HEIGHT_ANGLE_MAP = {
    "TALL":   IDEAL_ANGLES_TALL,
    "MEDIUM": IDEAL_ANGLES_MEDIUM,
    "SHORT":  IDEAL_ANGLES_SHORT,
}

# ── Generalized Shot Quality Formula ──────────────────────────
# Each joint has an "ideal center" angle and a "sigma" (tolerance in degrees).
# Score per joint = exp(-0.5 * ((angle - center) / sigma)^2) * 100
# This is a Gaussian scoring function — it gives 100% at perfect form
# and drops smoothly as the angle deviates, working for any player height.
JOINT_QUALITY_PARAMS = {
    "Front Elbow":   {"center": 155, "sigma": 20},
    "Back Elbow":    {"center": 85,  "sigma": 22},
    "Front Knee":    {"center": 145, "sigma": 22},
    "Back Knee":     {"center": 160, "sigma": 22},
    "Hip Angle":     {"center": 110, "sigma": 28},
    "Shoulder Line": {"center": 15,  "sigma": 18},
}

# Height-based center adjustments (added to JOINT_QUALITY_PARAMS centers)
HEIGHT_QUALITY_ADJUSTMENTS = {
    "TALL":   {"Front Knee": +10, "Back Knee": +10, "Hip Angle": +5},
    "MEDIUM": {},  # no adjustment
    "SHORT":  {"Front Knee": -12, "Back Knee": -12, "Hip Angle": -8},
}

# ── Shot Classification Signatures ────────────────────────────
# Each shot is defined by angle range rules.
# Format: {angle_name: (min, max)} — all rules must match for the shot.
SHOT_SIGNATURES = {
    "Cover Drive": {
        "Front Elbow":  (130, 180),
        "Front Knee":   (115, 170),
        "Shoulder Line":(0, 45),
    },
    "Pull Shot": {
        "Front Elbow":  (90, 160),
        "Shoulder Line":(30, 90),
        "Hip Angle":    (60, 130),
    },
    "Sweep Shot": {
        "Front Knee":   (70, 115),   # extreme knee bend
        "Hip Angle":    (50, 110),
    },
    "Defensive Push": {
        "Front Elbow":  (90, 140),   # arm compact
        "Front Knee":   (155, 180),  # nearly straight — minimal footwork
    },
}

# ── Cover Drive Phase Thresholds ──────────────────────────────
PHASE_NAMES = ["Stance", "Backswing & Stride", "Downswing & Impact", "Follow-through"]

