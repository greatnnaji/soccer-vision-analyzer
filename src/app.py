from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO


BASE_OUTPUT_DIR = Path("outputs")
UPLOADED_VIDEO_DIR = BASE_OUTPUT_DIR / "uploaded_videos"
PROCESSED_VIDEO_DIR = BASE_OUTPUT_DIR / "processed_videos"


def save_uploaded_video(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Path:
    UPLOADED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    uploaded_path = UPLOADED_VIDEO_DIR / uploaded_file.name

    with uploaded_path.open("wb") as target_file:
        target_file.write(uploaded_file.getbuffer())

    return uploaded_path


def get_video_metadata(video_path: Path) -> dict[str, float | int]:
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


def process_video(video_path: Path, model: YOLO | None = None) -> Path:
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

    return output_path


st.set_page_config(page_title="SoccerVision", layout="wide")

st.title("SoccerVision")
st.write("AI-powered soccer clip analyzer using YOLO and OpenCV.")

uploaded_file = st.file_uploader("Upload a soccer clip", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.success("Video uploaded successfully.")

    local_video_path = save_uploaded_video(uploaded_file)
    video_metadata = get_video_metadata(local_video_path)

    # Load YOLO model (this may download weights the first time)
    with st.spinner("Loading YOLO model..."):
        try:
            model = YOLO("yolov8n.pt")
        except Exception:
            model = None

    processed_video_path = process_video(local_video_path, model)

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Uploaded Video")
        st.video(str(local_video_path))

    with right_column:
        st.subheader("Video Metadata")
        st.json(video_metadata)

        st.subheader("Processed Video")
        st.video(str(processed_video_path))

    st.caption(f"Saved upload to {local_video_path} and processed output to {processed_video_path}.")