# 🚶 Footfall Counter using Computer Vision

A real-time computer vision system that detects, tracks, and counts people crossing a virtual line in video streams. This solution uses object detection (YOLO/HOG) and centroid tracking to accurately count entries and exits.

## 📋 Table of Contents
- [Overview]
- [Features]
- [Approach & Methodology]
- [Installation]
- [Usage]
- [Counting Logic]
- [Project Structure]
- [Example Output]
- [Limitations & Future Improvements]

---

## 🎯 Overview

This project implements a footfall counter that:
1. **Detects** people in video frames using YOLO or HOG detector
2. **Tracks** individuals across frames using centroid tracking
3. **Counts** entries and exits based on crossing a virtual line
4. **Visualizes** results with bounding boxes, trajectories, and real-time counts

**Video Source**: This system can work with any video file or live webcam feed. For testing, I recommend using:
- YouTube videos of mall entrances, corridors, or doorways
- Custom recorded videos using a phone/webcam


---

## ✨ Features

- ✅ Real-time people detection using YOLOv4-tiny or HOG
- ✅ Centroid-based tracking with unique ID assignment
- ✅ Configurable virtual counting line
- ✅ Separate entry and exit counters
- ✅ Trajectory visualization for tracked objects
- ✅ Works with video files or live camera feeds
- ✅ Handles multiple people simultaneously
- ✅ Robust to brief occlusions
- ✅ Video output with overlaid information

---

## 🧠 Approach & Methodology

### 1. **Detection Stage**
**Primary Method**: YOLOv4-tiny via OpenCV DNN
- Fast and accurate person detection
- Uses pre-trained weights on COCO dataset (class 0 = person)
- Confidence threshold filtering to reduce false positives
- Non-Maximum Suppression (NMS) to eliminate duplicate detections

**Fallback Method**: HOG (Histogram of Oriented Gradients)
- Used if YOLO weights are not available
- Built-in OpenCV people detector
- Less accurate but doesn't require external model files

### 2. **Tracking Stage**
**Centroid Tracking Algorithm**:
- Calculates the center point (centroid) of each detected bounding box
- Assigns unique IDs to new detections
- Matches detections across frames using Euclidean distance
- Maintains object identity even during brief disappearances
- Stores trajectory history for visualization

**Key Parameters**:
- `max_disappeared`: Number of frames an object can be missing before deregistration (default: 40)
- `max_distance`: Maximum pixel distance for matching centroids between frames (default: 100)

### 3. **Counting Logic**
The counting mechanism works by:

1. **Virtual Line Definition**: A horizontal line is drawn at a configurable position (default: middle of frame)

2. **Position Tracking**: For each tracked object, we store its previous Y-coordinate

3. **Crossing Detection**:
   - **Entry (Downward)**: When previous_y < line_y AND current_y >= line_y
   - **Exit (Upward)**: When previous_y > line_y AND current_y <= line_y

4. **Duplicate Prevention**: Each object can only trigger one count per crossing using a `crossed` dictionary

5. **Direction Awareness**: The system distinguishes between entries and exits based on crossing direction

**Visual Representation**:
```
        Frame Top (y=0)
    ┌─────────────────────┐
    │   Person moving ↓   │  previous_y < line_y
    │─────────────────────│  ← COUNTING LINE (line_y)
    │   Person moved ↓    │  current_y >= line_y
    │                     │  → Entry Count +1
    └─────────────────────┘
        Frame Bottom
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Webcam (optional, for live feed)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/footfall-counter.git
cd footfall-counter
```

### Step 2: Install Dependencies
```bash
pip install opencv-python numpy
```

### Step 3: Download YOLO Weights (Recommended)
For better accuracy, download YOLOv4-tiny model files:

```bash
# Download weights (245 MB)
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

# Download config
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
```

Place both files in the same directory as `footfall_counter.py`.

**Note**: If you skip this step, the system will automatically use HOG detector (less accurate but works out of the box).

---

## 🚀 Usage

### Basic Usage (Video File)
```bash
python footfall_counter.py --video path/to/video.mp4
```

### Save Output Video
```bash
python footfall_counter.py --video input.mp4 --output result.mp4
```

### Use Webcam (Real-time)
```bash
python footfall_counter.py --video 0
```

### Adjust Counting Line Position
```bash
# Line at 30% from top
python footfall_counter.py --video input.mp4 --line 0.3

# Line at 70% from top
python footfall_counter.py --video input.mp4 --line 0.7
```

### Change Detection Sensitivity
```bash
python footfall_counter.py --video input.mp4 --confidence 0.6
```

### Process Without Display (Faster)
```bash
python footfall_counter.py --video input.mp4 --output result.mp4 --no-display
```

### Command-line Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video` | str | Required | Path to video file or camera index (0, 1, etc.) |
| `--output` | str | None | Path to save output video |
| `--line` | float | 0.5 | Counting line position (0.0-1.0, where 0 is top) |
| `--confidence` | float | 0.5 | Detection confidence threshold (0.0-1.0) |
| `--no-display` | flag | False | Disable display window for headless processing |

---

## 📊 Counting Logic (Detailed)

### Algorithm Flowchart
```
Start
  ↓
[Detect People] → Bounding boxes
  ↓
[Calculate Centroids] → (x, y) coordinates
  ↓
[Update Tracker] → Match with previous frame
  ↓
[Check Each Tracked Object]
  ↓
  ├─ New object? → Store initial position
  │
  ├─ Has previous position?
  │   ↓
  │   Check crossing:
  │   ├─ prev_y < line_y AND curr_y >= line_y? → Entry +1
  │   └─ prev_y > line_y AND curr_y <= line_y? → Exit +1
  │
  └─ Update previous position
  ↓
[Visualize Results]
  ↓
Next Frame / End
```

### Handling Edge Cases

1. **Multiple Crossings**: An object is only counted once per crossing direction
2. **Occlusions**: Tracker maintains IDs for up to 40 frames of disappearance
3. **Crowded Scenes**: NMS prevents counting the same person multiple times
4. **False Detections**: Confidence threshold and tracking persistence filter noise

---

## 📁 Project Structure

```
footfall-counter/
│
├── footfall_counter.py      # Main script with all functionality
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
├── models/                   # (Optional) Model weights directory
│   ├── yolov4-tiny.weights
│   └── yolov4-tiny.cfg
│
├── videos/                   # Sample videos
│   ├── test_input.mp4
│   └── output_result.mp4
│
└── screenshots/              # Example outputs
    ├── screenshot1.png
    └── screenshot2.png
```

---

## 🖼️ Example Output

The processed video shows:
- **Cyan horizontal line**: Virtual counting line
- **Green circles**: Tracked object centroids
- **Green trails**: Movement trajectories
- **Text overlay**: Object IDs, entry/exit counts, total people

**Sample Statistics**:
```
====================================================
FINAL RESULTS
====================================================
Total Frames Processed: 1247
Entry Count: 15
Exit Count: 12
Total Crossings: 27
====================================================
```

---

## ⚠️ Limitations & Future Improvements

### Current Limitations
- Centroid tracking can fail with severe occlusions or