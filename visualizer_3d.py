# ───────────────────────────────────────────────────────────────
# visualizer_3d.py  –  Interactive 3D Stick Figure (Separate Thread)
# ───────────────────────────────────────────────────────────────
"""
Renders the 3D posture model in a SEPARATE DAEMON THREAD using
matplotlib TkAgg backend. This gives a fully interactive, rotatable
3D window while the OpenCV video loop continues uninterrupted.

The previous approach (Agg → OpenCV image) was a static screenshot.
This approach gives a live, draggable, zoomable 3D window.
"""

import threading
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from config import CONFIDENCE_THRESHOLD


KEYPOINT_SIDES = {
    0: "center",
    5: "front",  6: "back",
    7: "front",  8: "back",
    9: "front",  10: "back",
    11: "front", 12: "back",
    13: "front", 14: "back",
    15: "front", 16: "back",
}
DEPTH_MAP = {"front": -0.2, "center": 0.0, "back": 0.2}

SKELETON_3D = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

SEGMENT_COLORS = {
    (5, 6):   "#00FFFF",
    (5, 7):   "#00FF80", (7, 9):   "#00FF80",
    (6, 8):   "#FFB300", (8, 10):  "#FFB300",
    (5, 11):  "#FFFF00", (6, 12):  "#FFFF00", (11, 12): "#FFFF00",
    (11, 13): "#FF0080", (13, 15): "#FF0080",
    (12, 14): "#8000FF", (14, 16): "#8000FF",
}


def _build_pts(keypoints):
    """Convert YOLO 2D keypoints to estimated 3D coordinates."""
    pts = {}
    for idx in range(17):
        if keypoints[idx][2] < CONFIDENCE_THRESHOLD:
            continue
        px = float(keypoints[idx][0])
        py = float(keypoints[idx][1])
        pz = DEPTH_MAP.get(KEYPOINT_SIDES.get(idx, "center"), 0.0)
        pts[idx] = (px, py, pz)
    return pts


def _normalise(pts):
    """Centre and scale the skeleton to fit a unit cube."""
    xs   = [p[0] for p in pts.values()]
    ys   = [p[1] for p in pts.values()]
    cx_  = np.mean(xs)
    cy_  = np.mean(ys)
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1)

    def _n(p):
        return ((p[0] - cx_) / span,
                -(p[1] - cy_) / span,   # flip Y so head is up
                p[2])
    return {k: _n(v) for k, v in pts.items()}


def _draw_figure(keypoints, angles, score, shot_type):
    """
    Runs inside a daemon thread.
    Creates a full interactive TkAgg matplotlib window the user can rotate.
    """
    # Must set backend HERE inside the thread to avoid conflicts
    matplotlib.use("TkAgg")

    pts = _build_pts(keypoints)
    if len(pts) < 4:
        return

    pts_n = _normalise(pts)

    fig = plt.figure(figsize=(7, 8), facecolor="#1a1a2e")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#0d1117")

    title = f"3D Posture Model  |  Score: {score}/100"
    if shot_type:
        title += f"  |  Shot: {shot_type}"
    ax.set_title(title, color="white", fontsize=11, pad=12)

    # Draw skeleton bones
    for (a, b) in SKELETON_3D:
        if a not in pts_n or b not in pts_n:
            continue
        pa, pb = pts_n[a], pts_n[b]
        col = SEGMENT_COLORS.get((a, b), SEGMENT_COLORS.get((b, a), "#FFFFFF"))
        ax.plot(
            [pa[0], pb[0]],
            [pa[2], pb[2]],   # Z as side-axis
            [pa[1], pb[1]],   # Y as vertical
            color=col, linewidth=3.5, solid_capstyle="round"
        )

    # Draw joint dots
    for idx, (x, y, z) in pts_n.items():
        ax.scatter(x, z, y, color="#FFFF00", s=50, zorder=5, depthshade=False)

    # Key angle annotations
    annotations = {
        7:  ("F.Elbow", angles.get("Front Elbow")),
        8:  ("B.Elbow", angles.get("Back Elbow")),
        13: ("F.Knee",  angles.get("Front Knee")),
        11: ("Hip",     angles.get("Hip Angle")),
    }
    for idx, (label, val) in annotations.items():
        if idx in pts_n and val is not None:
            x, y, z = pts_n[idx]
            ax.text(x + 0.04, z, y + 0.04,
                    f"{label}\n{val:.0f}°",
                    color="white", fontsize=8, fontweight="bold")

    # Styling
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.set_tick_params(colors="#555555", labelsize=7)
    ax.set_xlabel("← Left  |  Right →", color="#666666", fontsize=8)
    ax.set_ylabel("Depth",               color="#666666", fontsize=8)
    ax.set_zlabel("↑ Up  |  Down ↓",    color="#666666", fontsize=8)
    ax.grid(True, color="#223344", linewidth=0.4)

    # Start with a good side-on cricket view
    ax.view_init(elev=12, azim=-75)

    # Hint text
    fig.text(0.5, 0.01,
             "🖱️  Left-drag to rotate  |  Scroll to zoom  |  Right-drag to pan",
             ha="center", color="#888888", fontsize=9)

    plt.tight_layout(pad=1.5)
    plt.show()   # blocks inside the thread only — OpenCV loop is unaffected


class Visualizer3D:
    """Spawns an interactive 3D viewer in a daemon thread."""

    def render(self, keypoints: np.ndarray, angles: dict,
               score: int, shot_type: str = ""):
        """
        Opens the interactive 3D window in a background thread.
        The main OpenCV loop continues at full speed.
        """
        t = threading.Thread(
            target=_draw_figure,
            args=(keypoints, angles, score, shot_type),
            daemon=True   # thread dies automatically when main program exits
        )
        t.start()
