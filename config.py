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
IDEAL_ANGLES = {
    "Front Elbow":    (140, 180, 120, 180),  # near-full extension at impact
    "Back Elbow":     (60, 110, 40, 130),    # compact and loaded
    "Front Knee":     (130, 165, 110, 175),  # flexed for stability (~146°)
    "Back Knee":      (140, 180, 120, 180),  # relatively straight on pivot
    "Hip Angle":      (80, 140, 60, 160),    # trunk rotation
    "Shoulder Line":  (0, 35, 0, 50),        # degrees from horizontal
}

# ── Cover Drive Phase Thresholds ──────────────────────────────
# Used by the analyzer to classify the current phase
PHASE_NAMES = ["Stance", "Backswing & Stride", "Downswing & Impact", "Follow-through"]
