# SoccerVision

An AI-powered soccer clip analyzer built with Python, OpenCV, YOLO, and Streamlit.

## Overview

SoccerVision processes soccer video clips to automatically detect and track players and the ball using the YOLOv8 object detection model. The application generates annotated output videos with bounding boxes and provides detailed detection statistics.

This project demonstrates the complete computer vision pipeline: from video I/O and preprocessing, through real-time object detection, to post-processing and statistical analysis.

## Features

- 📹 Upload soccer video clips in MP4, MOV, or AVI format
- 🤖 Automatic player and ball detection using YOLOv8
- 🎨 Real-time frame annotation with bounding boxes and confidence scores
- 📊 Detection statistics dashboard showing counts and average confidence
- 🎬 Browser-compatible annotated video output
- ⚡ Progress indicators and status updates

## Tech Stack

- **Video Processing**: OpenCV, moviepy
- **Object Detection**: YOLOv8 (ultralytics)
- **Web Interface**: Streamlit
- **Language**: Python 3.10+

## Project Structure

```
soccer-vision-analyzer/
├── src/
│   └── app.py              # Main Streamlit application
├── outputs/
│   ├── uploaded_videos/    # Saved uploaded videos
│   └── processed_videos/   # Annotated output videos
├── sample_videos/          # Example videos for testing
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── .gitignore
```

## Computer Vision Pipeline

The application implements a complete CV pipeline with five phases:

### Phase 1: Upload & Input
- User uploads soccer video via Streamlit UI
- Video validated for supported formats (MP4, MOV, AVI)
- File saved locally with metadata

### Phase 2: Video Processing
- Extract video metadata (FPS, resolution, frame count)
- Initialize OpenCV video reader and writer
- Prepare output video file with same specs as input
- Frame-by-frame processing ready

### Phase 3: YOLO Detection
- Load YOLOv8 nano model (~7MB, fast inference)
- Run per-frame object detection
- Filter detections for "person" and "sports ball" classes
- Draw bounding boxes: green for players, red for ball
- Add confidence labels to each detection

### Phase 4: Statistics Collection
- Track detection counts per frame
- Accumulate confidence scores
- Calculate average confidence per class
- Display metrics dashboard

### Phase 5: Output & Presentation
- Write annotated frames to output MP4
- Convert to H.264 codec for browser compatibility
- Display original and annotated videos side-by-side
- Show detection summary and detailed stats
- Save all files to outputs/ directory

## Installation

### Requirements
- Python 3.10 or higher
- ~2GB of free disk space (for YOLO model + outputs)

### Setup

```bash
# Clone repository
git clone https://github.com/greatnnaji/soccer-vision-analyzer.git
cd soccer-vision-analyzer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YOLO model (automatic on first run)
# Or pre-download: python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Usage

### Running the Application

```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate

# Start Streamlit app
streamlit run src/app.py

# App will open at http://localhost:8501
```

### Processing a Video

1. **Upload**: Click "Upload a soccer clip" and select your video file
2. **Wait**: App processes the video frame-by-frame (takes 30-60 seconds for typical clips)
3. **View Results**: 
   - Left column: original and metadata
   - Right column: annotated output video
   - Bottom: detection statistics

### Output Files

Processed videos are saved to `outputs/processed_videos/` with naming pattern:
- `{original_name}_processed.mp4` - annotated video
- `{original_name}_processed_browser.mp4` - H.264 version for browser playback

## Learning Outcomes

### Computer Vision Concepts Demonstrated

1. **Video I/O**: Reading/writing video files with OpenCV, handling frame sequences
2. **Object Detection**: Using pre-trained deep learning models (YOLO) for real-time inference
3. **Frame Annotation**: Drawing geometric shapes and text on video frames
4. **Statistical Analysis**: Aggregating and computing metrics across video sequences
5. **Performance Optimization**: Balancing accuracy vs. speed (YOLOv8 nano for efficiency)

### Key Insights

- **YOLO Efficiency**: YOLOv8 nano achieves real-time detection (~50 fps) on CPU
- **Confidence Thresholds**: YOLO's confidence scores help filter false positives
- **Class Filtering**: Post-detection filtering improves relevance (only "person" and "sports ball")
- **Frame-by-Frame Processing**: Enables tracking and temporal analysis in future extensions
- **Codec Compatibility**: H.264 ensures videos play in all modern browsers

## Code Quality

- **Modular Functions**: Separation of concerns (load, save, detect, aggregate)
- **Type Hints**: Full type annotations for clarity and error prevention
- **Error Handling**: Graceful fallbacks if model fails or file reading errors occur
- **Docstrings**: Comprehensive documentation for all functions
- **Progress Indicators**: Streamlit spinners and progress bars for UX

## Future Enhancements

- **Tracking**: Assign consistent IDs to players across frames using Kalman filtering
- **Temporal Analysis**: Track player movements and compute speed/distance metrics
- **Pose Estimation**: Detect player body parts to analyze running/passing postures
- **Zone Analysis**: Segment field into regions to track player positioning
- **Performance Metrics**: Compare model efficiency across YOLO variants (nano, small, medium)
- **Batch Processing**: Process multiple videos in sequence with aggregated reports
- **Export Options**: Save statistics to CSV, generate PDF reports

## Architecture Notes

### Why YOLOv8n (nano)?
- Fast: ~50 fps inference on CPU
- Small: 7MB model, suitable for deployment
- Accurate: Sufficient for soccer player/ball detection
- Trade-off: Slightly lower accuracy than larger variants, but real-time performance

### Codec Handling
- OpenCV writes MP4 using mp4v codec (wide support)
- Moviepy converts to H.264 for guaranteed browser playback
- Adds minimal overhead: ~5-10 seconds per video

### Memory Efficiency
- Processes one frame at a time (no full video in memory)
- Scales to videos of any length (only frame buffer in RAM)