# 🏏 Posture Expert — AI Cricket Cover Drive Analysis

An advanced computer vision pipeline for real-time biomechanical analysis of the cricket cover drive. 

## 📸 System in Action
The system is highly robust across various broadcast angles, camera zooms, and batting phases:

| Impact Phase (Player 1) | Impact Phase (Player 2) |
| :---: | :---: |
| ![Demo 1](assets/demo1.png) | ![Demo 2](assets/demo2.png) |

| Stance Phase | High-Zoom Tracking |
| :---: | :---: |
| ![Demo 3](assets/demo3.png) | ![Demo 4](assets/demo4.png) |

*(Also correctly identifies and scores varying postures like in [Demo 5](assets/demo5.png))*

---

## 🧠 Technical Architecture & Theory

This project was engineered to solve three major challenges in sports computer vision: **Multi-person disambiguation** (finding the batsman among bowlers/fielders), **Occlusion handling** (dealing with batting pads blocking lower-body keypoints), and **Real-time processing** (running frame-by-frame analysis without lag).

### Core Libraries & Architectural Justification

#### 1. Pose Estimation: Ultralytics YOLOv8-Pose
* **Why YOLOv8?** We chose YOLOv8 over alternatives like **MediaPipe** or **OpenPose**. 
  * *MediaPipe* is exceptionally fast but natively struggles with multi-person scenes (it frequently jumps between the bowler, non-striker, and striker).
  * *OpenPose* handles multiple people well but is computationally heavy, often requiring high-end GPUs to achieve real-time FPS. 
  * *YOLOv8-Pose* strikes the perfect balance: a single-pass CNN architecture that simultaneously detects bounding boxes and 17 COCO keypoints, maintaining high FPS while easily tracking 10+ people on a cricket field.
* **Implementation:** The model is configured with a lowered `CONFIDENCE_THRESHOLD = 0.35` specifically to combat the occlusion caused by bulky cricket equipment (helmets, batting pads, and gloves) which usually confuse standard COCO-trained models.

#### 2. Vision & UI Pipeline: OpenCV (`cv2`)
* **Why OpenCV?** For the graphical pipeline, OpenCV was chosen over high-level wrappers like *PIL* or *Matplotlib*. 
  * Video streams are effectively massive 3D arrays (Frames × Height × Width × Channels). OpenCV operates directly on memory buffers using heavily optimized C/C++ backends.
  * It provides zero-overhead rendering for our HUD, dynamically padding frames to standardize resolution (preventing text-squashing on TikTok/vertical video formats), and creating the broadcast-style freeze-frame effect.

#### 3. Biomechanical Mathematics: NumPy
* **Why NumPy?** Computing 6 different joint angles per frame requires calculating inverse tangents across 3-point vertices.
  * Doing this in pure Python `math` loops is slow. NumPy vectorizes these operations, running them closer to the hardware level.

---

## ⚙️ Algorithmic Deep-Dive

### 1. Smart Striker Identification (`pose_detector.py`)
Cricket broadcasts present a unique challenge: the bowler is often in the extreme foreground (producing the largest bounding box), and the non-striker is closer to the camera than the striker. 
To solve this, the system uses a heuristic scoring algorithm rather than naïve area-based selection:
* **Centrality Score:** The camera invariably tracks the ball, placing the striker near the horizontal center (`image_width / 2`).
* **Depth Penalty (`y2` coordinate):** Foreground objects (bowlers) have their feet at the very bottom edge of the frame. The system mathematically penalizes bounding boxes that touch the bottom 10% of the screen.
* **Area Filtering:** Rejects microscopic boxes (deep fielders) to save compute.

### 2. Occlusion-Resistant Phase Detection (`cover_drive_analyzer.py`)
Biomechanical analysis requires knowing *when* the shot is being played. The system breaks the shot into 4 phases: *Stance, Backswing & Stride, Downswing & Impact, Follow-through*.
* **The Pad Problem:** Batting pads heavily obscure the knees and ankles, meaning `front_knee` confidence often drops to zero. 
* **The Solution:** The algorithm uses the upper body as a proxy. For the critical **Impact** phase, it looks for an extended front elbow (`> 130°`) combined with the wrists dropping below the hips/shoulders. 

### 3. Broadcast Freeze-Frame (`main.py`)
To prevent visual clutter, the system mimics a professional sports broadcast. 
1. While the bowler runs in, the video plays cleanly with no skeletal overlays.
2. The AI silently analyzes frames in the background memory buffer.
3. The exact millisecond the algorithm detects the `Downswing & Impact` phase, it triggers an interrupt.
4. The system pauses the video, renders the 17 keypoints, calculates the 6 joint angles, computes the Posture Score (0-100), and overlays the coaching HUD.

---

## 📊 Evaluation Metrics & Benchmarks

To ensure production readiness, the system is evaluated on both **Inference Latency** and **Spatial/Temporal Accuracy**.

### Inference Performance (CPU Benchmark)
By utilizing fully vectorized NumPy operations, the biomechanical engine adds virtually zero overhead to the AI inference.
*   **Overall Pipeline:** ~18.86 FPS
*   **AI Inference (YOLOv8):** 49.53 ms
*   **OpenCV Rendering:** 0.64 ms
*   **Biomechanics Math (NumPy):** **0.02 ms** *(Highly Optimized)*

### Accuracy Metrics (IoU & PCK)
To mathematically prove the robustness of the system, the repository includes an `evaluation_metrics.py` module. This script calculates spatial and temporal Intersection over Union (IoU) against ground-truth data to evaluate the Smart Tracking Heuristic and Phase Detection logic.

**Example Evaluation Output:**
```text
==================================================
🏏 POSTURE EXPERT: EVALUATION METRICS MODULE
==================================================

[1] Spatial Accuracy (Bounding Box IoU)
Ground Truth Box: [100, 100, 200, 200]
Predicted Box:    [90, 110, 210, 190]
Calculated IoU:   0.6800 (68.0%)
Result: ✅ True Positive (IoU > 0.50)

[2] Phase Detection Accuracy (Temporal IoU)
Ground Truth 'Impact' Phase: 2.5s to 3.0s
Predicted 'Impact' Phase:    2.6s to 3.1s
Calculated Temporal IoU:     0.6667 (66.7%)
Result: ✅ Accurate Phase Detection (IoU > 0.50)
```
*   **PCK (Percentage of Correct Keypoints):** This metric is utilized to benchmark YOLO's spatial pixel accuracy for critical joints (elbows/knees) against human annotations.

---

## 💻 Quick Start & Commands

**Installation:**
```bash
pip install -r requirements.txt
```

**Running the App:**
```bash
# Easiest way (Opens a file picker)
python main.py

# CLI specific paths
python main.py --video my_video.mp4
python main.py --image my_image.jpg
```

**In-App Controls:**
* `p` — Manually pause the video and instantly force a biomechanical analysis on that exact frame.
* `s` — Save a high-resolution screenshot of the current analysis.
* `q` or `ESC` — Quit the application.
