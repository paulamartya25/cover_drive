# 🏏 Posture Expert — Cricket Cover Drive Analysis

Full-body keypoint detection and biomechanical analysis for the cricket cover drive shot, powered by **YOLOv8 Pose Estimation**.

## Features

- **17 COCO Keypoints** — detects nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles
- **6 Joint Angles** — front/back elbow, front/back knee, hip rotation, shoulder alignment
- **4-Phase Detection** — Stance → Backswing & Stride → Downswing & Impact → Follow-through
- **Posture Score** — 0–100 composite score based on biomechanical ideals
- **Live Coaching Tips** — actionable feedback on form corrections
- **HUD Dashboard** — real-time on-screen overlay with all metrics
- **3 Input Modes** — single image, video file, or live webcam

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run

```bash
# Analyze a single image
python main.py --image cover_drive.jpg

# Analyze a video
python main.py --video batting_clip.mp4

# Live webcam feed
python main.py --webcam

# Save output video
python main.py --webcam --save output.mp4
```

### 3. Controls (Video / Webcam)

| Key       | Action              |
|-----------|---------------------|
| `q` / ESC | Quit                |
| `s`       | Save screenshot     |
| `p`       | Pause / Resume      |

## Project Structure

```
tech_s_yolo/
├── main.py                  # CLI entry point
├── pose_detector.py         # YOLOv8 model wrapper
├── angle_calculator.py      # Joint angle computation
├── cover_drive_analyzer.py  # Phase detection & scoring
├── visualizer.py            # Skeleton drawing & HUD
├── config.py                # Constants & thresholds
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Model Options

By default the **nano** model (`yolov8n-pose.pt`) is used for speed.  
For higher accuracy, pass a larger model:

```bash
python main.py --webcam --model yolov8s-pose.pt   # small
python main.py --webcam --model yolov8m-pose.pt   # medium
python main.py --webcam --model yolov8l-pose.pt   # large
```

## How It Works

1. **YOLOv8-Pose** detects the player and localises 17 body keypoints
2. **AngleCalculator** computes 6 cricket-specific joint angles
3. **CoverDriveAnalyzer** classifies the shot phase and scores posture
4. **Visualizer** renders the annotated skeleton + coaching HUD

---

*Built with [Ultralytics YOLOv8](https://docs.ultralytics.com/) and OpenCV.*
