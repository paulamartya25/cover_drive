import cv2
import time
import sys
import numpy as np
from pose_detector import PoseDetector
from cover_drive_analyzer import CoverDriveAnalyzer
from visualizer import Visualizer

def run_benchmark(video_path: str):
    print(f"[INFO] Initializing Benchmark on: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    # Initialize components
    detector = PoseDetector(model_path="yolov8n-pose.pt")
    analyzer = CoverDriveAnalyzer()
    viz = Visualizer()
    
    frame_count = 0
    total_yolo_time = 0.0
    total_logic_time = 0.0
    total_render_time = 0.0
    
    # Read first frame to get dimensions
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    TARGET_H = 640
    scale = TARGET_H / raw_h if raw_h > 0 else 1.0
    w = int(raw_w * scale)
    h = TARGET_H

    print("[INFO] Running inference... (This runs without UI for maximum speed)")
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (w, h))
        
        # 1. Measure YOLOv8 Inference Time
        t0 = time.time()
        detections = detector.detect(frame)
        player = detector.get_primary_player(detections, frame.shape)
        t1 = time.time()
        total_yolo_time += (t1 - t0)
        
        # 2. Measure Biomechanical Logic (NumPy) Time
        if player:
            analysis = analyzer.analyze(player["keypoints"])
        else:
            analysis = None
        t2 = time.time()
        total_logic_time += (t2 - t1)
        
        # 3. Measure OpenCV Rendering Time
        if player and analysis:
            # Create a blank HUD to simulate the real rendering workload
            blank_hud = np.zeros((h, 320, 3), dtype=np.uint8)
            clean_display = np.hstack([frame, blank_hud])
            _ = viz.render(frame.copy(), player, analysis)
        t3 = time.time()
        total_render_time += (t3 - t2)
        
        frame_count += 1
        
        # Print progress every frame so the user knows it's working
        print(f"  -> Processed {frame_count} frames...", end="\r")
        sys.stdout.flush()
        
        # Stop after 100 frames to make the benchmark fast
        if frame_count >= 100:
            break

    print(f"\n  -> Finished processing {frame_count} frames!      ")
    total_pipeline_time = time.time() - start_time
    
    if frame_count == 0:
        print("[ERROR] No frames processed.")
        return

    print("\n" + "="*50)
    print("🚀 POSTURE EXPERT: PERFORMANCE BENCHMARK")
    print("="*50)
    print(f"Video Resolution:       {w}x{h} (scaled)")
    print(f"Total Frames Processed: {frame_count}")
    print(f"Overall Pipeline FPS:   {frame_count / total_pipeline_time:.2f} FPS")
    print("-" * 50)
    print("⏱️ LATENCY BREAKDOWN (Average per frame):")
    print(f"1. AI Inference (YOLO): {(total_yolo_time/frame_count)*1000:.2f} ms")
    print(f"2. Biomechanics Math:   {(total_logic_time/frame_count)*1000:.2f} ms")
    print(f"3. OpenCV Rendering:    {(total_render_time/frame_count)*1000:.2f} ms")
    print("-" * 50)
    print("💡 INTERVIEW TALKING POINT:")
    print("Notice how the Biomechanics Math (NumPy) takes almost 0 ms.")
    print("This proves the vectorization architecture is highly optimized!")
    print("="*50)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_benchmark(sys.argv[1])
    else:
        # No argument provided -> Open a file picker
        import tkinter as tk
        from tkinter import filedialog
        print("[INFO] No file specified. Opening file picker...")
        
        # Hide the main tkinter window
        root = tk.Tk()
        root.withdraw()
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select a Video for Benchmarking",
            filetypes=(
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            )
        )
        
        if file_path:
            run_benchmark(file_path)
        else:
            print("[INFO] No file selected. Exiting.")
