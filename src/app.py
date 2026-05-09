"""
SoccerVision: AI-powered soccer clip analyzer using YOLO and OpenCV.

This application processes soccer video clips to detect players and the ball,
drawing bounding boxes and collecting detection statistics for analysis.

Pipeline stages:
1. Upload video file via Streamlit UI
2. Save video locally
3. Extract video metadata
4. Load YOLO model for object detection
5. Process video frame-by-frame, detecting players and balls
6. Create annotated output video with detection overlays
7. Display statistics dashboard with detection metrics
"""

from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO
from moviepy import VideoFileClip


BASE_OUTPUT_DIR = Path("outputs")
UPLOADED_VIDEO_DIR = BASE_OUTPUT_DIR / "uploaded_videos"
PROCESSED_VIDEO_DIR = BASE_OUTPUT_DIR / "processed_videos"


def convert_to_browser_mp4(input_path: Path) -> Path:
    """Convert video to H.264 codec for better browser compatibility.
    
    Args:
        input_path: Path to input video file
        
    Returns:
        Path to the converted browser-safe video file
    """
    browser_output_path = input_path.with_name(f"{input_path.stem}_browser.mp4")

    clip = VideoFileClip(str(input_path))
    clip.write_videofile(
        str(browser_output_path),
        codec="libx264",
        audio=False
    )
    clip.close()

    return browser_output_path

def save_uploaded_video(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Path:
    """Save uploaded video file to local disk.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Path to the saved video file
    """
    UPLOADED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    uploaded_path = UPLOADED_VIDEO_DIR / uploaded_file.name

    with uploaded_path.open("wb") as target_file:
        target_file.write(uploaded_file.getbuffer())

    return uploaded_path


def get_video_metadata(video_path: Path) -> dict[str, float | int]:
    """Extract video metadata using OpenCV.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with fps, width, height, and frame_count
    """
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    metadata = {
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

    capture.release()
    return metadata


def process_video(video_path: Path, model: YOLO | None = None) -> tuple[Path, dict]:
    """Process video frame-by-frame, run YOLO detection, and collect statistics.
    
    This function:
    1. Opens the video file with OpenCV
    2. Iterates through each frame
    3. Runs YOLO detection (if model provided) to find players and balls
    4. Draws bounding boxes and labels on each frame
    5. Writes the annotated frames to output video
    6. Tracks and aggregates detection statistics
    
    Args:
        video_path: Path to input video file
        model: Pre-loaded YOLO model or None
        
    Returns:
        Tuple of (processed_video_path, statistics_dict):
        - processed_video_path: Path to output annotated video
        - statistics_dict: Contains player count, ball count, and average confidences
    """
    PROCESSED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = PROCESSED_VIDEO_DIR / f"{video_path.stem}_processed.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise ValueError("Could not create output video writer.")

    # Prepare model class mapping if available
    model_names = {}
    if model is not None:
        try:
            model_names = model.model.names if hasattr(model, "model") else getattr(model, "names", {})
        except Exception:
            model_names = {}

    # Initialize stats for tracking detections
    stats = {
        "player_count": 0,
        "ball_count": 0,
        "player_confidences": [],
        "ball_confidences": [],
    }

    frame_index = 0
    progress = st.progress(0, text="Processing video frames...")

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_index += 1

        # Run detection if a YOLO model is provided
        if model is not None:
            try:
                results = model(frame)[0]

                if hasattr(results, "boxes") and len(results.boxes) > 0:
                    try:
                        xyxy = results.boxes.xyxy.cpu().numpy()
                        confs = results.boxes.conf.cpu().numpy()
                        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
                    except Exception:
                        xyxy = []
                        confs = []
                        cls_ids = []

                    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids):
                        class_name = model_names.get(int(cls_id), str(cls_id))
                        if class_name not in ("person", "sports ball"):
                            continue

                        # Update statistics
                        if class_name == "person":
                            stats["player_count"] += 1
                            stats["player_confidences"].append(float(conf))
                        elif class_name == "sports ball":
                            stats["ball_count"] += 1
                            stats["ball_confidences"].append(float(conf))

                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        color = (0, 255, 0) if class_name == "person" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(frame, label, (x1, max(y1 - 8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except Exception:
                # Ignore per-frame detection errors to keep processing
                pass

        # Default overlay (frame index)
        cv2.putText(
            frame,
            f"Frame {frame_index}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        if frame_count > 0:
            progress.progress(min(frame_index / frame_count, 1.0), text="Processing video frames...")

    capture.release()
    writer.release()
    progress.progress(1.0, text="Processing complete.")

    st.write(f"Processed file size: {output_path.stat().st_size} bytes")

    # Calculate average confidences
    stats["player_avg_confidence"] = (
        sum(stats["player_confidences"]) / len(stats["player_confidences"])
        if stats["player_confidences"]
        else 0.0
    )
    stats["ball_avg_confidence"] = (
        sum(stats["ball_confidences"]) / len(stats["ball_confidences"])
        if stats["ball_confidences"]
        else 0.0
    )

    browser_video_path = convert_to_browser_mp4(output_path)
    return browser_video_path, stats

st.set_page_config(page_title="SoccerVision", layout="wide")

st.title("⚽ SoccerVision")
st.markdown("### AI-powered soccer clip analyzer using YOLO and OpenCV")
st.write(
    "Upload a soccer video clip to automatically detect players and the ball. "
    "This app uses the YOLOv8 object detection model to analyze each frame."
)

uploaded_file = st.file_uploader("Upload a soccer clip", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.success("✅ Video uploaded successfully!")
    
    st.divider()
    st.subheader("📊 Processing Video...")
    
    # Step 1: Save uploaded video
    local_video_path = save_uploaded_video(uploaded_file)
    
    # Step 2: Extract metadata
    video_metadata = get_video_metadata(local_video_path)
    st.info(f"Video loaded: {video_metadata['frame_count']} frames @ {video_metadata['fps']:.1f} fps")
    
    # Step 3: Load YOLO model
    with st.spinner("🔄 Loading YOLO model..."):
        try:
            model = YOLO("yolov8n.pt")
        except Exception:
            model = None
            st.warning("Could not load YOLO model. Proceeding without detections.")
    
    # Step 4: Process video with detections
    with st.spinner("🎬 Processing frames and detecting objects..."):
        processed_video_path, detection_stats = process_video(local_video_path, model)
    
    st.success("✅ Processing complete!")
    st.divider()
    
    # Display results in two columns
    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("📹 Uploaded Video")
        st.video(str(local_video_path))
        
        st.subheader("📋 Video Metadata")
        metadata_display = {
            "Resolution": f"{int(video_metadata['width'])}x{int(video_metadata['height'])}",
            "Frame Count": int(video_metadata['frame_count']),
            "FPS": f"{video_metadata['fps']:.2f}",
        }
        for key, value in metadata_display.items():
            st.metric(key, value)

    with right_column:
        st.subheader("🤖 YOLO Detection Output")
        st.video(str(processed_video_path))

    st.divider()
    
    # Display detection statistics dashboard
    st.subheader("📊 Detection Summary")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.metric("👥 Total Players", detection_stats["player_count"])

    with stat_col2:
        st.metric("⚽ Total Balls", detection_stats["ball_count"])

    with stat_col3:
        avg_player_conf = detection_stats["player_avg_confidence"]
        st.metric("Player Confidence", f"{avg_player_conf:.3f}")

    with stat_col4:
        avg_ball_conf = detection_stats["ball_avg_confidence"]
        st.metric("Ball Confidence", f"{avg_ball_conf:.3f}")

    # Display detailed stats
    st.subheader("📈 Detailed Statistics")
    details = {
        "Total Players Detected": detection_stats["player_count"],
        "Total Balls Detected": detection_stats["ball_count"],
        "Average Player Confidence": f"{detection_stats['player_avg_confidence']:.4f}",
        "Average Ball Confidence": f"{detection_stats['ball_avg_confidence']:.4f}",
    }
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Detection Counts**")
        st.json({
            "Players": detection_stats["player_count"],
            "Balls": detection_stats["ball_count"],
        })
    
    with col2:
        st.write("**Confidence Scores**")
        st.json({
            "Avg Player Confidence": f"{detection_stats['player_avg_confidence']:.4f}",
            "Avg Ball Confidence": f"{detection_stats['ball_avg_confidence']:.4f}",
        })

    st.divider()
    st.caption(
        "💾 Files saved: "
        f"uploaded to `{local_video_path}` | "
        f"output to `{processed_video_path}`"
    )