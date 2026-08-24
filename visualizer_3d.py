# ───────────────────────────────────────────────────────────────
# visualizer_3d.py  –  Single Persistent Interactive 3D Window
# ───────────────────────────────────────────────────────────────
"""
Architecture:
  - ONE long-running viewer process is spawned the first time render() is called.
  - It keeps a matplotlib window open indefinitely.
  - Subsequent calls to render() send new keypoint data via a Queue.
  - The viewer process reads from the queue and UPDATES the existing axes in-place.
  - Result: single rotatable window that refreshes automatically.
"""

import multiprocessing
import numpy as np
from config import CONFIDENCE_THRESHOLD


# ── Skeleton geometry ─────────────────────────────────────────

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


# ── Shared helpers ────────────────────────────────────────────

def _build_and_normalise(keypoints_list):
    """Convert raw keypoint list → normalised 3D dict."""
    keypoints = np.array(keypoints_list)
    pts = {}
    for idx in range(min(17, len(keypoints))):
        if keypoints[idx][2] < CONFIDENCE_THRESHOLD:
            continue
        px = float(keypoints[idx][0])
        py = float(keypoints[idx][1])
        pz = DEPTH_MAP.get(KEYPOINT_SIDES.get(idx, "center"), 0.0)
        pts[idx] = (px, py, pz)

    if len(pts) < 4:
        return None

    xs   = [p[0] for p in pts.values()]
    ys   = [p[1] for p in pts.values()]
    cx_  = np.mean(xs)
    cy_  = np.mean(ys)
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1)

    def norm(p):
        return ((p[0] - cx_) / span, -(p[1] - cy_) / span, p[2])

    return {k: norm(v) for k, v in pts.items()}


def _populate_axes(ax, pts_n, angles, score, shot_type):
    """Clear the axes and redraw the skeleton + annotations."""
    ax.cla()
    ax.set_facecolor("#0d1117")

    title = f"3D Posture Model  |  Score: {score}/100"
    if shot_type and shot_type != "Unknown Shot":
        title += f"  |  {shot_type}"
    ax.set_title(title, color="white", fontsize=11, pad=10)

    for (a, b) in SKELETON_3D:
        if a not in pts_n or b not in pts_n:
            continue
        pa, pb = pts_n[a], pts_n[b]
        col = SEGMENT_COLORS.get((a, b), SEGMENT_COLORS.get((b, a), "#FFFFFF"))
        ax.plot(
            [pa[0], pb[0]], [pa[2], pb[2]], [pa[1], pb[1]],
            color=col, linewidth=3.5, solid_capstyle="round"
        )

    for idx, (x, y, z) in pts_n.items():
        ax.scatter(x, z, y, color="#FFFF00", s=55, zorder=5, depthshade=False)

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

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.set_tick_params(colors="#555555", labelsize=7)
    ax.set_xlabel("← Left  |  Right →", color="#666666", fontsize=8)
    ax.set_ylabel("Depth",               color="#666666", fontsize=8)
    ax.set_zlabel("↑ Up  |  Down ↓",    color="#666666", fontsize=8)
    ax.grid(True, color="#223344", linewidth=0.4)


# ── Persistent viewer process ─────────────────────────────────

def _viewer_loop(queue: multiprocessing.Queue):
    """
    Runs in a separate process.
    Keeps ONE matplotlib window open.
    Polls the queue for new skeleton data and redraws in-place.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    plt.ion()   # interactive mode — window stays open while we update

    fig = plt.figure(figsize=(7, 8), facecolor="#1a1a2e")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#0d1117")
    ax.set_title("3D Posture Model  |  Waiting for analysis...",
                 color="white", fontsize=11)
    fig.text(0.5, 0.01,
             "🖱️  Left-drag: Rotate  |  Scroll: Zoom  |  Right-drag: Pan",
             ha="center", color="#888888", fontsize=9)

    # Remember the user's current view angle across updates
    current_elev = 12
    current_azim = -75
    ax.view_init(elev=current_elev, azim=current_azim)

    plt.tight_layout(pad=1.5)
    plt.show(block=False)
    plt.pause(0.05)

    while plt.fignum_exists(fig.number):
        if not queue.empty():
            try:
                kp_list, angles, score, shot_type = queue.get_nowait()
            except Exception:
                plt.pause(0.1)
                continue

            pts_n = _build_and_normalise(kp_list)
            if pts_n is None:
                plt.pause(0.1)
                continue

            # Save the user's current rotation before clearing
            current_elev = ax.elev
            current_azim = ax.azim

            _populate_axes(ax, pts_n, angles, score, shot_type)

            # Restore the view angle so the user's rotation is preserved
            ax.view_init(elev=current_elev, azim=current_azim)

            fig.canvas.draw()
            fig.canvas.flush_events()

        plt.pause(0.1)   # 10 Hz poll — lightweight


# ── Public class ──────────────────────────────────────────────

class Visualizer3D:
    """
    Manages a single persistent 3D viewer process.
    Calling render() updates the existing window instead of opening a new one.
    """

    def __init__(self):
        self._process = None
        self._queue   = None

    def render(self, keypoints: np.ndarray, angles: dict,
               score: int, shot_type: str = ""):
        """Send new skeleton data to the persistent window."""
        kp_list = keypoints.tolist()
        data    = (kp_list, angles, score, shot_type)

        # If window is already open, just push new data to it
        if self._process is not None and self._process.is_alive():
            # Drain stale frames so only the latest is shown
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Exception:
                    break
            self._queue.put(data)
            return

        # First call — spawn the persistent viewer process
        self._queue   = multiprocessing.Queue()
        self._queue.put(data)
        self._process = multiprocessing.Process(
            target=_viewer_loop,
            args=(self._queue,),
            daemon=True
        )
        self._process.start()
