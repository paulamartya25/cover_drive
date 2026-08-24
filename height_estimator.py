# ───────────────────────────────────────────────────────────────
# height_estimator.py  –  Estimate player height category from keypoints
# ───────────────────────────────────────────────────────────────
"""
Uses the pixel distance from nose (kp[0]) to ankle (kp[15] or kp[16])
as a proxy for the player's relative height in the frame.

Returns one of: "SHORT" | "MEDIUM" | "TALL"

This drives the height-adaptive angle thresholds in config.py.
The beauty of this approach: it doesn't need the player's actual height in cm.
It uses their proportional size in the current video frame, which naturally
accounts for camera distance differences between videos.
"""

import numpy as np
from config import CONFIDENCE_THRESHOLD


def estimate_height_category(keypoints: np.ndarray) -> str:
    """
    Estimate height category from pixel distance between nose and ankle.

    The Generalized Height Ratio Formula:
        pixel_height = |nose_y - ankle_y|
        normalized   = pixel_height / frame_height (estimated from bbox)

    We use the ankle-to-nose span rather than the full bounding box
    because the bounding box can include empty space above the head.

    Returns: "SHORT" | "MEDIUM" | "TALL"
    """
    nose_y   = _get_y(keypoints, 0)
    l_ankle  = _get_y(keypoints, 15)
    r_ankle  = _get_y(keypoints, 16)

    # Use whichever ankle is visible
    ankle_y = None
    if l_ankle is not None and r_ankle is not None:
        ankle_y = max(l_ankle, r_ankle)   # use the lower (larger y-value)
    elif l_ankle is not None:
        ankle_y = l_ankle
    elif r_ankle is not None:
        ankle_y = r_ankle

    if nose_y is None or ankle_y is None:
        return "MEDIUM"  # fallback — can't estimate

    pixel_height = abs(ankle_y - nose_y)

    # Thresholds are in raw pixels (assumes 640px tall frame).
    # Taller players occupy more pixels in the frame.
    if pixel_height > 380:
        return "TALL"
    elif pixel_height > 240:
        return "MEDIUM"
    else:
        return "SHORT"


def _get_y(keypoints: np.ndarray, idx: int):
    """Return y-coordinate if confidence is sufficient, else None."""
    if keypoints[idx][2] < CONFIDENCE_THRESHOLD:
        return None
    return float(keypoints[idx][1])
