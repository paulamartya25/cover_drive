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

    # ── skeleton + keypoints ──────────────────────────────────

    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray,
                      bbox: tuple = None) -> np.ndarray:
        """
        Draw skeleton lines + keypoint dots on the frame.

        Args:
            frame:     BGR image (mutated in-place and returned).
            keypoints: (17, 3) array.
            bbox:      optional (x1,y1,x2,y2,conf) bounding box.
        """
        overlay = frame.copy()

        # Bounding box
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

        # Skeleton lines
        for (a, b) in SKELETON_PAIRS:
            if keypoints[a][2] < CONFIDENCE_THRESHOLD or keypoints[b][2] < CONFIDENCE_THRESHOLD:
                continue
            pt1 = (int(keypoints[a][0]), int(keypoints[a][1]))
            pt2 = (int(keypoints[b][0]), int(keypoints[b][1]))
            color = COLORS.get(PAIR_COLORS.get((a, b), "torso"), COLORS["torso"])
            cv2.line(overlay, pt1, pt2, color, 3, cv2.LINE_AA)

        # Keypoint dots + labels
        for i in range(17):
            if keypoints[i][2] < CONFIDENCE_THRESHOLD:
                continue
            cx, cy = int(keypoints[i][0]), int(keypoints[i][1])
            # Outer ring + filled center
            cv2.circle(overlay, (cx, cy), 7, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(overlay, (cx, cy), 5, COLORS["keypoint"], -1, cv2.LINE_AA)

            # We removed the tiny text labels here (like 'RWr') because they clutter 
            # the screen. The angle labels (drawn later) are much more important.

        # Blend overlay
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        return frame

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
