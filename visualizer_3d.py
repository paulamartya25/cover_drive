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
    """Clear the axes and redraw a human-figure 3D skeleton with filled segments."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax.cla()
    ax.set_facecolor("#0d1117")

    title = f"3D Posture Model  |  Score: {score}/100"
    if shot_type and shot_type != "Unknown Shot":
        title += f"  |  {shot_type}"
    ax.set_title(title, color="white", fontsize=11, pad=10)

    # ── Helper: map pts_n → matplotlib (X, Y, Z) coords ──────
    # pts_n stores (x, y, z) where x=horiz, y=vert(inverted), z=depth
    # matplotlib plot call: ax.plot(X, Y, Z) = ax.plot(x, z, y)
    # So matplotlib X=pts[0], matplotlib Y=pts[2], matplotlib Z=pts[1]
    def mpl(idx):
        p = pts_n[idx]
        return (p[0], p[2], p[1])   # (mpl_x, mpl_y, mpl_z)

    # ── Limb widths (in normalised coords) ───────────────────
    LIMB_W = {
        (5,  6): 0.09,   # shoulder girdle
        (5,  7): 0.05,   # L upper arm
        (7,  9): 0.038,  # L forearm
        (6,  8): 0.05,   # R upper arm
        (8, 10): 0.038,  # R forearm
        (11,12): 0.09,   # hip girdle
        (11,13): 0.065,  # L thigh
        (13,15): 0.048,  # L shin
        (12,14): 0.065,  # R thigh
        (14,16): 0.048,  # R shin
    }

    def ribbon(p1, p2, half_w):
        """
        Return a flat horizontal ribbon (Poly3DCollection vertices list)
        for the limb from p1→p2 in matplotlib (X,Y,Z) coords.
        The ribbon is offset in the X direction (left/right) so it faces
        the default camera angle.
        """
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        hw1 = half_w            # proximal half-width
        hw2 = half_w * 0.65     # distal  half-width (tapered)
        return [
            (x1 - hw1, y1, z1),
            (x1 + hw1, y1, z1),
            (x2 + hw2, y2, z2),
            (x2 - hw2, y2, z2),
        ]

    # ── 1. Filled Torso Quad ──────────────────────────────────
    if all(k in pts_n for k in (5, 6, 11, 12)):
        ls, rs = mpl(5), mpl(6)
        lh, rh = mpl(11), mpl(12)
        torso_verts = [[ls, rs, rh, lh]]
        torso_col = Poly3DCollection(torso_verts, alpha=0.75, zorder=2)
        torso_col.set_facecolor("#00FFFF")
        torso_col.set_edgecolor("#000000")
        ax.add_collection3d(torso_col)

    # ── 2. Filled Limb Ribbons ────────────────────────────────
    # Draw order: legs first (behind), then arms (in front)
    draw_order = [
        (12,14), (14,16),   # R thigh, R shin
        (11,13), (13,15),   # L thigh, L shin
        (6, 8),  (8,10),    # R upper arm, R forearm
        (5, 7),  (7, 9),    # L upper arm, L forearm
        (5, 6),             # shoulder bar
        (11,12),            # hip bar
    ]
    for (a, b) in draw_order:
        if a not in pts_n or b not in pts_n:
            continue
        col_hex = SEGMENT_COLORS.get((a, b), SEGMENT_COLORS.get((b, a), "#FFFFFF"))
        half_w  = LIMB_W.get((a, b), LIMB_W.get((b, a), 0.04))
        verts   = [ribbon(mpl(a), mpl(b), half_w)]
        poly    = Poly3DCollection(verts, alpha=0.85, zorder=3)
        poly.set_facecolor(col_hex)
        poly.set_edgecolor("#111111")
        ax.add_collection3d(poly)

    # ── 3. Head Sphere ────────────────────────────────────────
    head_idx = next((i for i in (0, 3, 4, 1, 2) if i in pts_n), None)
    if head_idx is not None:
        hx, hy, hz = mpl(head_idx)
        ax.scatter(hx, hy, hz,
                   color="#FFD700", s=1200,
                   zorder=6, depthshade=False, edgecolors="#000000", linewidths=1.5)

    # ── 4. Joint Spheres ──────────────────────────────────────
    for idx in pts_n:
        if idx in (0, 1, 2, 3, 4):   # face keypoints — skip (head sphere covers these)
            continue
        x, y, z = mpl(idx)
        ax.scatter(x, y, z,
                   color="#FFFFFF", s=160,
                   zorder=5, depthshade=False, edgecolors="#333333", linewidths=1)

    # ── 5. Angle Annotations ──────────────────────────────────
    annotations = {
        7:  ("F.Elbow", angles.get("Front Elbow")),
        8:  ("B.Elbow", angles.get("Back Elbow")),
        13: ("F.Knee",  angles.get("Front Knee")),
        11: ("Hip",     angles.get("Hip Angle")),
    }
    for idx, (label, val) in annotations.items():
        if idx in pts_n and val is not None:
            x, y, z = mpl(idx)
            ax.text(x + 0.05, y, z + 0.05,
                    f"{label}\n{val:.0f}°",
                    color="white", fontsize=8, fontweight="bold", zorder=7)

    # ── 6. Axes styling ───────────────────────────────────────
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
