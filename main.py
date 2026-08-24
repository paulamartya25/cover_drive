# ───────────────────────────────────────────────────────────────
# main.py  –  Cricket Cover Drive Pose Analysis — Entry Point
# ───────────────────────────────────────────────────────────────
"""
Usage:
    python main.py                                     # opens file picker
    python main.py --image  path/to/cover_drive.jpg
    python main.py --video  path/to/batting_clip.mp4
    python main.py --webcam
    python main.py --webcam --save output.mp4

Keys during live view:
    q / ESC  — quit
    s        — save current frame as screenshot
    p        — pause / resume (video & webcam)
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from pose_detector import PoseDetector
from cover_drive_analyzer import CoverDriveAnalyzer
from visualizer import Visualizer
from visualizer_3d import Visualizer3D


# ── File picker (tkinter) ─────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

def browse_file() -> tuple[str, str]:
    """
    Open a file-picker dialog and return (path, kind).
    kind is 'image' or 'video' based on the chosen file extension.
    Returns (None, None) if the user cancels.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("[ERROR] tkinter not available — please pass --image or --video instead.")
        sys.exit(1)

    root = tk.Tk()
    root.withdraw()                # hide the root window
    root.attributes("-topmost", True)

    filetypes = [
        ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
        ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
        ("All files",   "*.*"),
    ]

    path = filedialog.askopenfilename(
        title="Posture Expert — Select a cricket video or image",
        filetypes=filetypes,
    )
    root.destroy()

    if not path:
        return None, None

    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return path, "image"
    else:
        return path, "video"        # treat anything non-image as video


# ── Frame Processing ──────────────────────────────────────────

def process_frame(frame: np.ndarray, detector: PoseDetector,
                  analyzer: CoverDriveAnalyzer, viz: Visualizer) -> np.ndarray:
    """Run the full pipeline on a single frame and return the annotated image."""
    detections = detector.detect(frame)
    player = detector.get_primary_player(detections, frame.shape)

    if player is None:
        # No person detected — show placeholder text
        cv2.putText(frame, "No player detected — step into frame",
                    (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2, cv2.LINE_AA)
        return frame

    analysis = analyzer.analyze(player["keypoints"], player_meta=player)
    frame = viz.render(frame, player, analysis)
    return frame


# ── Image Mode ────────────────────────────────────────────────

def run_image(path: str, detector, analyzer, viz):
    """Analyze a single image and display the result."""
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {path}")
        sys.exit(1)

    print(f"[INFO] Analyzing image: {path}")
    raw_h, raw_w = frame.shape[:2]
    
    # Standardize height to 640px to ensure the HUD fonts scale perfectly and look clear
    TARGET_H = 640
    scale = TARGET_H / raw_h if raw_h > 0 else 1.0
    w = int(raw_w * scale)
    h = TARGET_H
    
    frame = cv2.resize(frame, (w, h))
    result = process_frame(frame, detector, analyzer, viz)

    # Save output
    out_path = path.rsplit(".", 1)[0] + "_analyzed.jpg"
    cv2.imwrite(out_path, result)
    print(f"[INFO] Saved annotated image → {out_path}")

    cv2.imshow("Posture Expert — Cover Drive Analysis", result)
    print("[INFO] Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Video / Webcam Mode ──────────────────────────────────────

def run_video(source, detector, analyzer, viz, save_path: str = None):
    """
    Process video file or webcam feed frame-by-frame.

    Args:
        source: file path (str) or camera index (int).
        save_path: optional output file path.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    # Compute how many ms to wait per frame to match original video speed.
    # e.g. 30 FPS → wait 33ms per frame. Max 33ms so we never go slower than 30 FPS.
    frame_delay_ms = max(1, int(1000 / fps))

    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Standardize height to 640px to ensure the HUD fonts scale perfectly and look clear
    TARGET_H = 640
    scale = TARGET_H / raw_h if raw_h > 0 else 1.0
    w = int(raw_w * scale)
    h = TARGET_H

    hud_w = 320  # must match Visualizer.HUD_WIDTH
    total_w = w + hud_w  # combined frame width (video + HUD panel)

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (total_w, h))
        print(f"[INFO] Recording output → {save_path}")

    # Auto-save analyzed video next to the original
    if save_path is None and isinstance(source, str):
        base, ext = os.path.splitext(source)
        save_path = f"{base}_analyzed{ext}"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (total_w, h))
        print(f"[INFO] Auto-saving analyzed video → {save_path}")

    # Create the window BEFORE the loop and force it to the front
    win_name = "Posture Expert - Cover Drive Analysis"
    # Use KEEPRATIO to completely prevent squashing if manually resized
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(win_name, total_w, h)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)  # bring to front

    src_label = "webcam" if isinstance(source, int) else source
    print(f"[INFO] Processing: {src_label}  |  Resolution: {w}x{h} (scaled)  |  FPS: {fps:.1f}")
    print("[INFO] Keys:  q/ESC=quit  s=screenshot  p=pause")

    paused = False
    has_auto_paused = False
    display_frame = None
    frame_count = 0
    t_start = time.time()
    viz3d = Visualizer3D()   # ONE persistent 3D window for the whole session

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video.")
                break
            
            # Resize frame to our standard 640p height so fonts aren't squashed
            frame = cv2.resize(frame, (w, h))
            frame_count += 1

            # 1. Prepare clean frame (padded to maintain video writer dimensions)
            blank_hud = np.zeros((h, hud_w, 3), dtype=np.uint8)
            blank_hud[:] = (25, 25, 25)
            cv2.putText(blank_hud, "Analyzing live...", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            clean_display = np.hstack([frame, blank_hud])

            # 2. Detect in background to find the batsman on strike
            detections = detector.detect(frame)
            player = detector.get_primary_player(detections, frame.shape)

            trigger_pause = False
            if player is not None:
                analysis = analyzer.analyze(player["keypoints"], player_meta=player)
                current_phase = analysis.get("phase")

                # Reset tracker if they go back to Stance
                if current_phase == "Stance":
                    has_auto_paused = False
                # If they reach Impact, trigger the analysis freeze-frame
                elif current_phase == "Downswing & Impact" and not has_auto_paused:
                    trigger_pause = True
                    has_auto_paused = True

            if trigger_pause:
                paused = True
                display_frame = viz.render(frame.copy(), player, analysis)
                # ── Update the single persistent 3D window ──
                try:
                    viz3d.render(
                        player["keypoints"],
                        analysis.get("angles", {}),
                        analysis.get("score", 0),
                        analysis.get("shot_type", ""),
                    )
                except Exception as e:
                    print(f"[3D] Could not render 3D model: {e}")
                print("[INFO] Impact detected! Pausing for analysis. Press 'p' to resume.")
            else:
                display_frame = clean_display

            # FPS counter
            elapsed = time.time() - t_start
            live_fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(display_frame, f"FPS: {live_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            if writer:
                writer.write(display_frame)

            cv2.imshow(win_name, display_frame)

        key = cv2.waitKey(frame_delay_ms) & 0xFF
        if key in (ord("q"), 27):        # q or ESC
            break
        elif key == ord("s") and display_frame is not None:  # screenshot
            ts = time.strftime("%Y%m%d_%H%M%S")
            snap_path = f"screenshot_{ts}.jpg"
            cv2.imwrite(snap_path, display_frame)
            print(f"[INFO] Screenshot saved → {snap_path}")
        elif key == ord("p"):            # pause / resume
            paused = not paused
            if paused and 'frame' in locals() and frame is not None:
                # Force analysis display on manual pause!
                detections = detector.detect(frame)
                player = detector.get_primary_player(detections, frame.shape)
                if player:
                    analysis = analyzer.analyze(player["keypoints"], player_meta=player)
                    display_frame = viz.render(frame.copy(), player, analysis)
                    cv2.imshow(win_name, display_frame)
                    # ── Update the single persistent 3D window ──
                    try:
                        viz3d.render(
                            player["keypoints"],
                            analysis.get("angles", {}),
                            analysis.get("score", 0),
                            analysis.get("shot_type", ""),
                        )
                    except Exception as e:
                        print(f"[3D] Could not render 3D model: {e}")
                    print("[INFO] Paused manually. Displaying frame analysis + 3D model.")
            else:
                print("[INFO] Resumed")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Processed {frame_count} frames in {elapsed:.1f}s "
          f"({live_fps:.1f} avg FPS)")


# ── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Posture Expert — Cricket Cover Drive Keypoint Analysis (YOLOv8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=False)  # not required — browse is default
    group.add_argument("--image",  type=str, help="Path to an image file")
    group.add_argument("--video",  type=str, help="Path to a video file")
    group.add_argument("--browse", action="store_true",
                       help="Open a file picker to select a video or image (default)")

    parser.add_argument("--save",  type=str, default=None,
                        help="Save output video to this path (video only)")
    parser.add_argument("--model", type=str, default=None,
                        help="YOLOv8-pose model path (default: yolov8n-pose.pt)")

    args = parser.parse_args()

    # ── Determine input mode ──
    # If nothing specified, default to browse
    use_browse = args.browse or (not args.image and not args.video)

    if use_browse:
        print("[INFO] Opening file picker — select your cricket video or image...")
        path, kind = browse_file()
        if path is None:
            print("[INFO] No file selected. Exiting.")
            sys.exit(0)
        if kind == "image":
            args.image = path
        else:
            args.video = path
        print(f"[INFO] Selected: {path}  ({kind})")

    # Initialise components
    model_name = args.model or "yolov8n-pose.pt"
    print(f"[INFO] Loading YOLOv8 pose model: {model_name}")
    detector = PoseDetector(model_path=model_name)
    analyzer = CoverDriveAnalyzer()
    viz      = Visualizer()
    print("[INFO] Model loaded successfully ✓")

    if args.image:
        run_image(args.image, detector, analyzer, viz)
    elif args.video:
        run_video(args.video, detector, analyzer, viz, save_path=args.save)


import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()   # Required for Windows multiprocessing
    main()
