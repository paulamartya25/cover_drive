# ───────────────────────────────────────────────────────────────
# visualizer_3d.py  –  3D Stick Figure Renderer
# ───────────────────────────────────────────────────────────────
"""
Renders a 3D stick-figure model of the batsman's posture using Matplotlib.
Triggered when the user presses 'p' to freeze-frame during analysis.

How the Z-depth estimation works:
    YOLO gives us 2D (X, Y) pixel coordinates.
    We ESTIMATE Z (depth) using body symmetry rules:
      - "Front" side joints (closer to camera) → Z = -depth
      - "Back" side joints (further from camera) → Z = +depth
    This creates a plausible 3D model without a depth sensor.
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # Use TkAgg backend so it opens a proper window
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from config import CONFIDENCE_THRESHOLD


# COCO keypoint index → (name, side)
# side: "front", "back", "center"
KEYPOINT_SIDES = {
    0:  "center",   # Nose
    5:  "front",    # Left Shoulder   (front arm for RHB)
    6:  "back",     # Right Shoulder
    7:  "front",    # Left Elbow
    8:  "back",     # Right Elbow
    9:  "front",    # Left Wrist
    10: "back",     # Right Wrist
    11: "front",    # Left Hip
    12: "back",     # Right Hip
    13: "front",    # Left Knee
    14: "back",     # Right Knee
    15: "front",    # Left Ankle
    16: "back",     # Right Ankle
}

DEPTH_MAP = {"front": -0.15, "center": 0.0, "back": 0.15}

SKELETON_3D = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

SEGMENT_COLORS = {
    (5, 6): "#00FFFF",   # shoulder bar — cyan
    (5, 7): "#00FF80",   # left arm — green
    (7, 9): "#00FF80",
    (6, 8): "#FFB300",   # right arm — amber
    (8, 10): "#FFB300",
    (5, 11): "#FFFF00",  # torso — yellow
    (6, 12): "#FFFF00",
    (11, 12): "#FFFF00",
    (11, 13): "#FF0080", # left leg — magenta
    (13, 15): "#FF0080",
    (12, 14): "#8000FF", # right leg — purple
    (14, 16): "#8000FF",
}


class Visualizer3D:
    """Renders an interactive 3D stick-figure model of detected keypoints."""

    def render(self, keypoints: np.ndarray, angles: dict, score: int, shot_type: str = ""):
        """
        Opens a Matplotlib 3D window showing the batsman's skeleton.

        Args:
            keypoints: (17, 3) YOLO keypoints array.
            angles:    dict of joint angle values.
            score:     posture score (0-100).
            shot_type: classified shot name string.
        """
        # ── Build 3D points ───────────────────────────────────
        pts = {}  # idx → (x, y, z)
        for idx in range(17):
            if keypoints[idx][2] < CONFIDENCE_THRESHOLD:
                continue
            px = float(keypoints[idx][0])
            py = float(keypoints[idx][1])
            side = KEYPOINT_SIDES.get(idx, "center")
            pz = DEPTH_MAP[side]
            # Normalise X and Y to [-1, 1] range for clean display
            pts[idx] = (px, py, pz)

        if len(pts) < 4:
            print("[3D] Not enough visible keypoints for 3D render.")
            return

        # Normalise to centre the skeleton
        xs = [p[0] for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        cx, cy = np.mean(xs), np.mean(ys)
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1)

        def norm(p):
            return ((p[0] - cx) / span, -(p[1] - cy) / span, p[2])

        pts_n = {k: norm(v) for k, v in pts.items()}

        # ── Plot ─────────────────────────────────────────────
        fig = plt.figure(figsize=(7, 8), facecolor="#1a1a2e")
        ax = fig.add_subplot(111, projection="3d", facecolor="#16213e")
        ax.set_title(
            f"3D Posture Model  |  Score: {score}/100"
            + (f"  |  Shot: {shot_type}" if shot_type else ""),
            color="white", fontsize=11, pad=12
        )

        # Draw bones
        for (a, b) in SKELETON_3D:
            if a not in pts_n or b not in pts_n:
                continue
            pa, pb = pts_n[a], pts_n[b]
            color = SEGMENT_COLORS.get((a, b), SEGMENT_COLORS.get((b, a), "#FFFFFF"))
            ax.plot(
                [pa[0], pb[0]],
                [pa[2], pb[2]],   # Z as the "width" axis for side-on view
                [pa[1], pb[1]],   # Y is up
                color=color, linewidth=3, solid_capstyle="round"
            )

        # Draw joint dots
        for idx, (x, y, z) in pts_n.items():
            ax.scatter(x, z, y, color="#FFFF00", s=40, zorder=5)

        # Annotate key angles
        angle_labels = {
            7: ("Front Elbow", angles.get("Front Elbow")),
            8: ("Back Elbow", angles.get("Back Elbow")),
            13: ("F.Knee", angles.get("Front Knee")),
        }
        for idx, (label, val) in angle_labels.items():
            if idx in pts_n and val is not None:
                x, y, z = pts_n[idx]
                ax.text(x + 0.05, z, y, f"{label}\n{val:.0f}°",
                        color="white", fontsize=7)

        # Style axes
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_tick_params(colors="gray")
            axis.label.set_color("gray")
        ax.set_xlabel("←  Left    Right  →", color="#888888", fontsize=8)
        ax.set_ylabel("Depth", color="#888888", fontsize=8)
        ax.set_zlabel("↑  Up    Down  ↓", color="#888888", fontsize=8)
        ax.grid(True, color="#333355", linewidth=0.5)
        ax.set_facecolor("#0d1117")

        # Set a good initial view angle (side-on, like a cricket broadcaster)
        ax.view_init(elev=10, azim=-80)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
