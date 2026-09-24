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
    """
    Draw a mannequin-style dummy person using thick rounded lines.

    Thick matplotlib 3D lines render as cylinders — at linewidth 10–14
    they look like actual limb tubes, giving a clear human-dummy shape
    that is immediately recognisable from any viewing angle.

    Rendering layers (back → front):
      1. Thick coloured limb lines  (look like 3D cylinders)
      2. White joint spheres        (mark every articulation point)
      3. Gold head sphere           (large, proportional to body)
      4. Angle text annotations
    """
    ax.cla()
    ax.set_facecolor("#0d1117")

    title = f"3D Posture Model  |  Score: {score}/100"
    if shot_type and shot_type != "Unknown Shot":
        title += f"  |  {shot_type}"
    ax.set_title(title, color="white", fontsize=12, pad=12)

    # ── Coordinate helper ──────────────────────────────────────
    # pts_n[i] = (x, y, z)  where x=horizontal, y=vertical(inverted), z=depth
    # matplotlib axes:  plot(X, Y, Z) → X=horiz, Y=depth, Z=up/down
    def mpl(idx):
        p = pts_n[idx]
        return p[0], p[2], p[1]   # (mpl_X, mpl_Y, mpl_Z)

    # ── Limb styles: (color_hex, linewidth) ───────────────────
    # Thicker = looks more like a solid cylinder limb.
    # NOTE: torso SIDES (5,11) and (6,12) are intentionally OMITTED —
    #   at lw=12 they form a giant cyan box. The torso is instead implied
    #   by the shoulder and hip bars at thin linewidth.
    LIMB_STYLE = {
        (5,  6): ("#00FFFF",  6),   # shoulder bar     — thin connector
        (11,12): ("#00FFFF",  6),   # hip bar          — thin connector
        (5,  7): ("#00FF80", 13),   # L upper arm      — green
        (7,  9): ("#00FF80", 11),   # L forearm        — green
        (6,  8): ("#FFB300", 13),   # R upper arm      — amber
        (8, 10): ("#FFB300", 11),   # R forearm        — amber
        (11,13): ("#FF2080", 14),   # L thigh          — pink
        (13,15): ("#FF2080", 12),   # L shin           — pink
        (12,14): ("#9933FF", 14),   # R thigh          — purple
        (14,16): ("#9933FF", 12),   # R shin           — purple
    }


    # ── 1. Thick limb lines ───────────────────────────────────
    for (a, b), (col, lw) in LIMB_STYLE.items():
        if a not in pts_n or b not in pts_n:
            continue
        xa, ya, za = mpl(a)
        xb, yb, zb = mpl(b)
        ax.plot([xa, xb], [ya, yb], [za, zb],
                color=col, linewidth=lw,
                solid_capstyle="round", solid_joinstyle="round",
                zorder=3)

    # ── 2. Joint spheres (white with dark outline) ────────────
    for idx in pts_n:
        if idx in (0, 1, 2, 3, 4):   # face pts — covered by head sphere
            continue
        x, y, z = mpl(idx)
        ax.scatter(x, y, z,
                   color="#FFFFFF", s=220, zorder=5,
                   depthshade=False,
                   edgecolors="#222222", linewidths=1.5)

    # ── 3. Head sphere ────────────────────────────────────────
    # Try nose first, fall back to ear keypoints
    head_idx = next((i for i in (0, 3, 4, 1, 2) if i in pts_n), None)
    if head_idx is not None:
        hx, hy, hz = mpl(head_idx)
        # Outer dark ring for contrast against any background
        ax.scatter(hx, hy, hz,
                   color="#8B6914", s=3000, zorder=5,
                   depthshade=False, edgecolors="#000000", linewidths=0)
        # Gold fill
        ax.scatter(hx, hy, hz,
                   color="#FFD700", s=2400, zorder=6,
                   depthshade=False, edgecolors="#AA8800", linewidths=2)

    # ── 4. Angle annotations ──────────────────────────────────
    annotations = {
        7:  ("F.Elbow", angles.get("Front Elbow")),
        8:  ("B.Elbow", angles.get("Back Elbow")),
        13: ("F.Knee",  angles.get("Front Knee")),
        11: ("Hip",     angles.get("Hip Angle")),
    }
    for idx, (label, val) in annotations.items():
        if idx in pts_n and val is not None:
            x, y, z = mpl(idx)
            ax.text(x + 0.06, y, z + 0.06,
                    f"{label}\n{val:.0f}°",
                    color="white", fontsize=9, fontweight="bold", zorder=7)

    # ── 5. Axis styling ───────────────────────────────────────
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.set_tick_params(colors="#555555", labelsize=7)
    ax.set_xlabel("← Left  |  Right →", color="#666666", fontsize=8)
    ax.set_ylabel("Depth",               color="#666666", fontsize=8)
    ax.set_zlabel("↑ Up  |  Down ↓",    color="#666666", fontsize=8)
    ax.grid(True, color="#1a2a3a", linewidth=0.5)


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
