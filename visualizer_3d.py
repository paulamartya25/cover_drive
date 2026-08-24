# ───────────────────────────────────────────────────────────────
# visualizer_3d.py  –  3D Stick Figure Renderer (OpenCV window)
# ───────────────────────────────────────────────────────────────
"""
Renders the 3D posture model using Matplotlib's Agg backend (off-screen),
converts the result to a NumPy image, and displays it in a standard
resizable OpenCV window.

WHY THIS APPROACH:
  - matplotlib TkAgg backend blocks the entire Python process, killing
    the video playback FPS.
  - The Agg backend renders silently off-screen into a memory buffer.
  - We grab that buffer as a NumPy array and hand it to cv2.imshow(),
    which is non-blocking and plays nicely with the rest of the pipeline.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # CRITICAL: off-screen, non-blocking, no GUI event loop
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import cv2
from config import CONFIDENCE_THRESHOLD


KEYPOINT_SIDES = {
    0: "center", 5: "front",  6: "back",
    7: "front",  8: "back",   9: "front", 10: "back",
    11: "front", 12: "back", 13: "front", 14: "back",
    15: "front", 16: "back",
}
DEPTH_MAP = {"front": -0.15, "center": 0.0, "back": 0.15}

SKELETON_3D = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

SEGMENT_COLORS = {
    (5, 6):   "#00FFFF",
    (5, 7):   "#00FF80", (7, 9):   "#00FF80",
    (6, 8):   "#FFB300", (8, 10):  "#FFB300",
    (5, 11):  "#FFFF00", (6, 12):  "#FFFF00", (11, 12): "#FFFF00",
    (11, 13): "#FF0080", (13, 15): "#FF0080",
    (12, 14): "#8000FF", (14, 16): "#8000FF",
}

WIN_NAME = "3D Posture Model"


class Visualizer3D:

    def render(self, keypoints: np.ndarray, angles: dict,
               score: int, shot_type: str = ""):
        """
        Render a 3D stick figure and show it in a resizable OpenCV window.
        Completely non-blocking — does not slow the video pipeline at all.
        """
        pts = {}
        for idx in range(17):
            if keypoints[idx][2] < CONFIDENCE_THRESHOLD:
                continue
            px = float(keypoints[idx][0])
            py = float(keypoints[idx][1])
            pz = DEPTH_MAP.get(KEYPOINT_SIDES.get(idx, "center"), 0.0)
            pts[idx] = (px, py, pz)

        if len(pts) < 4:
            return

        # Normalise skeleton to centre
        xs   = [p[0] for p in pts.values()]
        ys   = [p[1] for p in pts.values()]
        cx_  = np.mean(xs)
        cy_  = np.mean(ys)
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1)

        def norm(p):
            return ((p[0] - cx_) / span, -(p[1] - cy_) / span, p[2])

        pts_n = {k: norm(v) for k, v in pts.items()}

        # ── Off-screen Matplotlib render ──────────────────────
        fig = plt.figure(figsize=(6, 7), dpi=100, facecolor="#1a1a2e")
        ax  = fig.add_subplot(111, projection="3d", facecolor="#0d1117")

        title = f"3D Posture Model  |  Score: {score}/100"
        if shot_type:
            title += f"  |  {shot_type}"
        ax.set_title(title, color="white", fontsize=10, pad=10)

        for (a, b) in SKELETON_3D:
            if a not in pts_n or b not in pts_n:
                continue
            pa, pb = pts_n[a], pts_n[b]
            col = SEGMENT_COLORS.get((a, b), SEGMENT_COLORS.get((b, a), "#FFFFFF"))
            ax.plot(
                [pa[0], pb[0]], [pa[2], pb[2]], [pa[1], pb[1]],
                color=col, linewidth=3, solid_capstyle="round"
            )

        for idx, (x, y, z) in pts_n.items():
            ax.scatter(x, z, y, color="#FFFF00", s=35, zorder=5, depthshade=False)

        # Key angle labels
        for idx, (label, val) in {
            7:  ("F.Elbow", angles.get("Front Elbow")),
            8:  ("B.Elbow", angles.get("Back Elbow")),
            13: ("F.Knee",  angles.get("Front Knee")),
        }.items():
            if idx in pts_n and val is not None:
                x, y, z = pts_n[idx]
                ax.text(x + 0.05, z, y, f"{label}\n{val:.0f}°",
                        color="white", fontsize=7)

        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False
            axis.set_tick_params(colors="#555555", labelsize=6)
        ax.set_xlabel("Left / Right", color="#666666", fontsize=7)
        ax.set_ylabel("Depth",        color="#666666", fontsize=7)
        ax.set_zlabel("Up / Down",    color="#666666", fontsize=7)
        ax.grid(True, color="#223344", linewidth=0.4)
        ax.view_init(elev=10, azim=-80)

        plt.tight_layout(pad=1.0)

        # ── Convert to OpenCV image ───────────────────────────
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img_rgba = buf.reshape(h, w, 4)
        img_bgr  = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
        plt.close(fig)   # free memory immediately

        # ── Display in resizable OpenCV window ────────────────
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 600, 700)
        cv2.imshow(WIN_NAME, img_bgr)
