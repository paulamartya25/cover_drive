# ───────────────────────────────────────────────────────────────
# visualizer.py  –  Skeleton Drawing, Angle Overlays & HUD
# ───────────────────────────────────────────────────────────────

import cv2
import math
import numpy as np

from config import (
    COLORS,
    CONFIDENCE_THRESHOLD,
    KEYPOINT_SHORT,
    PAIR_COLORS,
    SKELETON_PAIRS,
)


class Visualizer:
    """Draws pose skeletons, angle annotations, and a coaching HUD dashboard."""

    HUD_WIDTH = 320  # width of the side panel in pixels

    # ── Limb width fractions (relative to player bounding-box height) ─────────
    # Each tuple → (a, b) keypoint pair: fraction of player height used as width
    _LIMB_WIDTHS = {
        (5,  6): 0.11,   # shoulder girdle
        (5,  7): 0.055,  # L upper arm
        (7,  9): 0.040,  # L forearm
        (6,  8): 0.055,  # R upper arm
        (8, 10): 0.040,  # R forearm
        (5, 11): 0.090,  # L torso side  (drawn separately as quad, but kept for fallback)
        (6, 12): 0.090,  # R torso side
        (11,12): 0.110,  # hip girdle
        (11,13): 0.075,  # L thigh
        (13,15): 0.055,  # L shin
        (12,14): 0.075,  # R thigh
        (14,16): 0.055,  # R shin
    }
    # Face links are drawn as thin lines only (not filled segments)
    _FACE_PAIRS = {(0,1),(0,2),(1,3),(2,4)}

    # ── skeleton + keypoints ──────────────────────────────────

    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray,
                      bbox: tuple = None) -> np.ndarray:
        """
        Draw a human-figure skeleton using filled body-segment polygons.

        Each limb (upper arm, forearm, thigh, shin, torso) is rendered as a
        tapered filled trapezoid — wider at the body end, narrower at the
        extremity — giving a realistic silhouette instead of a stick figure.

        Rendering order (back-to-front):
          1. Filled torso quadrilateral
          2. Filled limb trapezoids (thighs, shins, arms)
          3. Shoulder & hip girdle bars
          4. Head ellipse
          5. Joint circles (white with dark outline)
          6. Bounding box + label

        Args:
            frame:     BGR image (mutated in-place and returned).
            keypoints: (17, 3) array — x, y, confidence per keypoint.
            bbox:      optional (x1, y1, x2, y2, conf) bounding box.
        """
        # ── Estimate player scale from bounding box ────────────────
        if bbox is not None:
            x1_b, y1_b, x2_b, y2_b, _ = bbox
            player_h = max(float(y2_b - y1_b), 80.0)
        else:
            # Fallback: estimate from nose-to-ankle pixel distance
            nose_y   = float(keypoints[0,  1]) if keypoints[0,  2] >= CONFIDENCE_THRESHOLD else None
            ankle_ys = [float(keypoints[i, 1]) for i in (15, 16)
                        if keypoints[i, 2] >= CONFIDENCE_THRESHOLD]
            if nose_y is not None and ankle_ys:
                player_h = max(ankle_ys) - nose_y
            else:
                player_h = frame.shape[0] * 0.55
            player_h = max(player_h, 80.0)

        overlay = frame.copy()

        # Helper: get (x, y) as float array or None
        def pt(idx):
            if keypoints[idx, 2] < CONFIDENCE_THRESHOLD:
                return None
            return np.array([float(keypoints[idx, 0]), float(keypoints[idx, 1])])

        # ── 1. Filled Torso Quadrilateral ──────────────────────────
        ls, rs, lh, rh = pt(5), pt(6), pt(11), pt(12)
        if ls is not None and rs is not None and lh is not None and rh is not None:
            torso_pts = np.array([
                [int(ls[0]), int(ls[1])],
                [int(rs[0]), int(rs[1])],
                [int(rh[0]), int(rh[1])],
                [int(lh[0]), int(lh[1])],
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [torso_pts], COLORS["torso"])
            cv2.polylines(overlay, [torso_pts], True, (0, 0, 0), 1, cv2.LINE_AA)

        # ── 2. Filled Limb Trapezoids ─────────────────────────────
        # Draw in order: legs first (behind torso), then arms (in front)
        draw_order = [
            # Legs (drawn first so torso sits on top)
            (12, 14), (14, 16),   # R thigh, R shin
            (11, 13), (13, 15),   # L thigh, L shin
            # Arms
            (6,  8),  (8, 10),    # R upper arm, R forearm
            (5,  7),  (7,  9),    # L upper arm, L forearm
            # Girdles (shoulder bar, hip bar)
            (5,  6),
            (11, 12),
        ]
        for (a, b) in draw_order:
            if (a, b) in self._FACE_PAIRS:
                continue
            p1, p2 = pt(a), pt(b)
            if p1 is None or p2 is None:
                continue
            color = COLORS.get(PAIR_COLORS.get((a, b), "torso"), COLORS["torso"])
            frac  = self._LIMB_WIDTHS.get((a, b), self._LIMB_WIDTHS.get((b, a), 0.05))
            width = int(frac * player_h)
            width = max(5, min(width, 70))
            self._draw_limb(overlay, p1, p2, width, color)

        # ── 3. Head Ellipse ───────────────────────────────────────
        # Try nose first, then ears as fallback
        head_pt = None
        for idx in (0, 3, 4, 1, 2):
            if keypoints[idx, 2] >= CONFIDENCE_THRESHOLD:
                head_pt = (int(keypoints[idx, 0]), int(keypoints[idx, 1]))
                break
        if head_pt is not None:
            # Radius proportional to player height; typical head ≈ 13 % of body
            r_y = max(10, min(int(player_h * 0.13), 55))
            r_x = max(8,  min(int(player_h * 0.09), 40))
            face_col = COLORS.get("face", (255, 200, 55))
            cv2.ellipse(overlay, head_pt, (r_x, r_y), 0, 0, 360, face_col, -1, cv2.LINE_AA)
            cv2.ellipse(overlay, head_pt, (r_x, r_y), 0, 0, 360, (0, 0, 0),  2, cv2.LINE_AA)

        # ── 4. Face skeleton lines (thin, not filled) ─────────────
        for (a, b) in self._FACE_PAIRS:
            p1, p2 = pt(a), pt(b)
            if p1 is None or p2 is None:
                continue
            cv2.line(overlay, tuple(p1.astype(int)), tuple(p2.astype(int)),
                     COLORS.get("face", (255, 200, 55)), 1, cv2.LINE_AA)

        # ── 5. Joint circles ──────────────────────────────────────
        for i in range(17):
            p = pt(i)
            if p is None:
                continue
            cx, cy = int(p[0]), int(p[1])
            joint_r = max(4, min(int(player_h * 0.025), 10))
            cv2.circle(overlay, (cx, cy), joint_r + 2, (20, 20, 20), -1, cv2.LINE_AA)   # dark ring
            cv2.circle(overlay, (cx, cy), joint_r,     (255, 255, 255), -1, cv2.LINE_AA) # white fill

        # ── 6. Bounding box + label ───────────────────────────────
        if bbox is not None:
            x1, y1, x2, y2, conf = bbox
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)),
                          COLORS["bbox"], 2)
            label = f"Player {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (int(x1), int(y1) - th - 8),
                          (int(x1) + tw + 6, int(y1)), COLORS["bbox"], -1)
            cv2.putText(overlay, label, (int(x1) + 3, int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Blend: 55 % skeleton overlay, 45 % original frame ─────
        # 45 % of the original frame bleeds through → player is clearly
        # visible underneath the coloured body segments.
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        return frame


    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _draw_limb(frame: np.ndarray,
                   pt1: np.ndarray, pt2: np.ndarray,
                   width: int, color: tuple) -> None:
        """
        Draw a filled, tapered trapezoid representing one body segment.

        The trapezoid is wider at pt1 (proximal / body end) and slightly
        narrower at pt2 (distal / extremity end), which mimics how real
        limbs taper from shoulder → wrist and hip → ankle.

        Args:
            frame:  BGR image to draw on (mutated in-place).
            pt1:    Proximal keypoint as float (x, y) array.
            pt2:    Distal  keypoint as float (x, y) array.
            width:  Width of the limb in pixels at the proximal end.
            color:  BGR fill color tuple.
        """
        direction = pt2 - pt1
        length = float(np.linalg.norm(direction))
        if length < 2.0:
            return

        # Unit perpendicular vector (90° from limb direction)
        perp = np.array([-direction[1], direction[0]]) / length

        half_w1 = width / 2.0          # proximal half-width
        half_w2 = width / 2.0 * 0.65   # distal half-width (tapered ~35%)

        corners = np.array([
            pt1 + perp * half_w1,    # proximal left
            pt1 - perp * half_w1,    # proximal right
            pt2 - perp * half_w2,    # distal right
            pt2 + perp * half_w2,    # distal left
        ], dtype=np.int32)

        cv2.fillPoly(frame, [corners], color)
        # Thin dark outline for depth / separation between adjacent limbs
        cv2.polylines(frame, [corners], isClosed=True,
                      color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    # ── angle arcs ────────────────────────────────────────────

    def draw_angle_arcs(self, frame: np.ndarray, keypoints: np.ndarray,
                        angles: dict, ratings: dict) -> np.ndarray:
        """Draw small arc indicators at each measured joint."""

        angle_vertex_map = {
            "Front Elbow": 7,
            "Back Elbow":  8,
            "Front Knee":  13,
            "Back Knee":   14,
            "Hip Angle":   11,
        }

        for name, vertex_idx in angle_vertex_map.items():
            value = angles.get(name)
            if value is None:
                continue
            if keypoints[vertex_idx][2] < CONFIDENCE_THRESHOLD:
                continue

            cx = int(keypoints[vertex_idx][0])
            cy = int(keypoints[vertex_idx][1])
            rating = ratings.get(name, "okay")
            color = COLORS.get(rating, COLORS["okay"])

            # Small arc
            cv2.ellipse(frame, (cx, cy), (22, 22), 0, 0, int(value),
                        color, 2, cv2.LINE_AA)

            # Angle label with background
            label = f"{value:.0f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            lx, ly = cx + 24, cy - 2
            cv2.rectangle(frame, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 3),
                          COLORS["text_bg"], -1)
            cv2.putText(frame, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # Shoulder line
        sl_val = angles.get("Shoulder Line")
        if sl_val is not None:
            if keypoints[5][2] >= CONFIDENCE_THRESHOLD and keypoints[6][2] >= CONFIDENCE_THRESHOLD:
                ls = (int(keypoints[5][0]), int(keypoints[5][1]))
                rs = (int(keypoints[6][0]), int(keypoints[6][1]))
                rating = ratings.get("Shoulder Line", "okay")
                color = COLORS.get(rating, COLORS["okay"])
                mid_x = (ls[0] + rs[0]) // 2
                mid_y = (ls[1] + rs[1]) // 2
                label = f"SL {sl_val:.0f}"
                cv2.putText(frame, label, (mid_x - 20, mid_y - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        return frame

    # ── HUD dashboard (separate side panel) ───────────────────

    def _create_hud_panel(self, height: int, analysis: dict) -> np.ndarray:
        """
        Create a standalone dark HUD panel image.

        Args:
            height: must match the video frame height.
            analysis: dict from CoverDriveAnalyzer.analyze().

        Returns:
            BGR image of shape (height, HUD_WIDTH, 3).
        """
        pw = self.HUD_WIDTH
        panel = np.zeros((height, pw, 3), dtype=np.uint8)
        panel[:] = (25, 25, 25)  # dark background

        # Draw a left border accent line
        cv2.line(panel, (0, 0), (0, height), (0, 200, 255), 3)

        cx = 18  # left margin
        cy = 30

        # ── Title ──
        cv2.putText(panel, "POSTURE EXPERT", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cy += 22
        cv2.putText(panel, "Cricket Cover Drive Analysis", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)
        cy += 18
        cv2.line(panel, (cx, cy), (pw - 18, cy), (60, 60, 60), 1)

        # ── Phase ──
        cy += 28
        phase = analysis.get("phase", "—")
        cv2.putText(panel, "PHASE", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        cy += 24
        cv2.putText(panel, phase, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # ── Shot Type ──
        shot_type   = analysis.get("shot_type", "")
        height_cat  = analysis.get("height_category", "")
        if shot_type:
            cy += 22
            cv2.putText(panel, "SHOT TYPE", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
            cy += 20
            cv2.putText(panel, shot_type, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 180, 0), 2, cv2.LINE_AA)

        # ── Height Category badge ──
        if height_cat:
            badge_colors = {
                "TALL":   (0, 200, 100),
                "MEDIUM": (0, 180, 255),
                "SHORT":  (200, 100, 255),
            }
            cy += 20
            badge_col = badge_colors.get(height_cat, (180, 180, 180))
            cv2.putText(panel, f"BUILD: {height_cat}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, badge_col, 1, cv2.LINE_AA)

        # ── Score ──
        cy += 35
        score = analysis.get("score", 0)
        if score >= 75:
            score_color = COLORS["ideal"]
        elif score >= 50:
            score_color = COLORS["okay"]
        else:
            score_color = COLORS["bad"]

        cv2.putText(panel, "POSTURE SCORE", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        cy += 28
        cv2.putText(panel, f"{score}", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, score_color, 3, cv2.LINE_AA)
        cv2.putText(panel, "/ 100", (cx + 55, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

        # Score bar
        cy += 14
        bar_x1, bar_x2 = cx, pw - 20
        bar_w = bar_x2 - bar_x1
        cv2.rectangle(panel, (bar_x1, cy), (bar_x2, cy + 8), (50, 50, 50), -1)
        fill_w = int(bar_w * score / 100)
        if fill_w > 0:
            cv2.rectangle(panel, (bar_x1, cy), (bar_x1 + fill_w, cy + 8), score_color, -1)
        cv2.rectangle(panel, (bar_x1, cy), (bar_x2, cy + 8), (70, 70, 70), 1)

        # ── Joint Angles ──
        cy += 28
        cv2.line(panel, (cx, cy), (pw - 18, cy), (60, 60, 60), 1)
        cy += 22
        cv2.putText(panel, "JOINT ANGLES", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        angles  = analysis.get("angles", {})
        ratings = analysis.get("ratings", {})

        for name in ["Front Elbow", "Back Elbow", "Front Knee",
                     "Back Knee", "Hip Angle", "Shoulder Line"]:
            cy += 26
            value  = angles.get(name)
            rating = ratings.get(name, "okay")
            color  = COLORS.get(rating, COLORS["okay"])

            # Status dot
            cv2.circle(panel, (cx + 4, cy - 4), 4, color, -1, cv2.LINE_AA)

            # Name
            cv2.putText(panel, name, (cx + 16, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1, cv2.LINE_AA)

            # Value — right-aligned
            val_str = f"{value:.0f} deg" if value is not None else "N/A"
            (tw, _), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(panel, val_str, (pw - 22 - tw, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # ── Coaching Tips ──
        cy += 22
        cv2.line(panel, (cx, cy), (pw - 18, cy), (60, 60, 60), 1)
        cy += 22
        cv2.putText(panel, "COACHING TIPS", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        tips = analysis.get("tips", [])
        for tip in tips[:3]:  # max 3 tips
            cy += 24
            # Wrap long text
            max_chars = 30
            if len(tip) <= max_chars:
                cv2.putText(panel, f"- {tip}", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 210, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(panel, f"- {tip[:max_chars]}", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 210, 255), 1, cv2.LINE_AA)
                cy += 18
                cv2.putText(panel, f"  {tip[max_chars:]}", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 210, 255), 1, cv2.LINE_AA)

        # ── Legend ──
        legend_y = height - 60
        cv2.line(panel, (cx, legend_y), (pw - 18, legend_y), (60, 60, 60), 1)
        legend_y += 18
        for label_text, clr_key in [("Ideal", "ideal"), ("Okay", "okay"), ("Fix", "bad")]:
            cv2.circle(panel, (cx + 4, legend_y - 3), 4, COLORS[clr_key], -1)
            cv2.putText(panel, label_text, (cx + 14, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 160, 160), 1, cv2.LINE_AA)
            cx += 70
        cx = 18  # reset

        # ── Footer ──
        cv2.putText(panel, "YOLOv8-Pose | Posture Expert v1.0",
                    (cx, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1, cv2.LINE_AA)

        return panel

    # ── combined render ───────────────────────────────────────

    def render(self, frame: np.ndarray, detection: dict,
               analysis: dict) -> np.ndarray:
        """
        Full render pipeline: skeleton → angle arcs → HUD (as side panel).

        The HUD is placed as a separate panel to the RIGHT of the video frame
        so it never obscures the player.
        """
        kp = detection["keypoints"]

        # Draw skeleton + arcs on the video frame
        frame = self.draw_skeleton(frame, kp, bbox=detection.get("bbox"))
        frame = self.draw_angle_arcs(frame, kp,
                                      analysis["angles"], analysis["ratings"])

        # Create the HUD as a side panel and concatenate
        h = frame.shape[0]
        hud_panel = self._create_hud_panel(h, analysis)
        combined = np.hstack([frame, hud_panel])

        return combined
