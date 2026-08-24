# ───────────────────────────────────────────────────────────────
# visualizer_3d.py  –  Interactive 3D Stick Figure (Separate Process)
# ───────────────────────────────────────────────────────────────
"""
Uses multiprocessing.Process instead of threading.Thread.

WHY:
  Tkinter (and matplotlib TkAgg) MUST run on the MAIN thread.
  On Windows, threading does NOT give you a new main thread —
  it gives you a secondary thread, which crashes Tkinter.

  multiprocessing.Process spawns a brand new Python process.
  That process has its OWN main thread where Tkinter runs perfectly.

RESULT:
  - Fully interactive, rotatable, zoomable 3D window
  - OpenCV video loop runs at full speed, completely unaffected
"""

import multiprocessing
import numpy as np
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
DEPTH_MAP   = {"front": -0.2, "center": 0.0, "back": 0.2}
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


def _draw_figure(keypoints_list, angles, score, shot_type):
    """
    Runs in a SEPARATE PROCESS with its own main thread.
    TkAgg/Tkinter works perfectly here.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    keypoints = np.array(keypoints_list)

    # Build 3D points from keypoints
    pts = {}
    for idx in range(min(17, len(keypoints))):
        if keypoints[idx][2] < CONFIDENCE_THRESHOLD:
            continue
        px = float(keypoints[idx][0])
        py = float(keypoints[idx][1])
        pz = DEPTH_MAP.get(KEYPOINT_SIDES.get(idx, "center"), 0.0)
        pts[idx] = (px, py, pz)

    if len(pts) < 4:
        return

    # Normalise skeleton to fit unit cube
    xs   = [p[0] for p in pts.values()]
    ys   = [p[1] for p in pts.values()]
    cx_  = np.mean(xs)
    cy_  = np.mean(ys)
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1)

    def norm(p):
        return ((p[0] - cx_) / span,
                -(p[1] - cy_) / span,
                p[2])

    pts_n = {k: norm(v) for k, v in pts.items()}

    # Draw the figure
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
            [pa[2], pb[2]],
            [pa[1], pb[1]],
            color=col, linewidth=3.5, solid_capstyle="round"
        )

    # Draw joint dots
    for idx, (x, y, z) in pts_n.items():
        ax.scatter(x, z, y, color="#FFFF00", s=55, zorder=5, depthshade=False)

    # Annotate key joint angles
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

    # Style
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.set_tick_params(colors="#555555", labelsize=7)
    ax.set_xlabel("← Left  |  Right →", color="#666666", fontsize=8)
    ax.set_ylabel("Depth",               color="#666666", fontsize=8)
    ax.set_zlabel("↑ Up  |  Down ↓",    color="#666666", fontsize=8)
    ax.grid(True, color="#223344", linewidth=0.4)
    ax.view_init(elev=12, azim=-75)

    fig.text(0.5, 0.01,
             "🖱️  Left-drag to rotate  |  Scroll to zoom  |  Right-drag to pan",
             ha="center", color="#888888", fontsize=9)

    plt.tight_layout(pad=1.5)
    plt.show()


class Visualizer3D:
    """Spawns an interactive 3D viewer in a completely separate process."""

    def render(self, keypoints: np.ndarray, angles: dict,
               score: int, shot_type: str = ""):
        """
        Opens the interactive 3D window in a new process.
        The main OpenCV loop continues at full speed.
        Keypoints are converted to a plain list for safe inter-process transfer.
        """
        kp_list = keypoints.tolist()   # numpy arrays can't be pickled directly on Windows

        p = multiprocessing.Process(
            target=_draw_figure,
            args=(kp_list, angles, score, shot_type),
            daemon=True
        )
        p.start()
