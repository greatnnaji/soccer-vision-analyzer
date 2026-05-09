from pathlib import Path

import cv2
import streamlit as st


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


def process_video(video_path: Path) -> Path:
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

    frame_index = 0
    progress = st.progress(0, text="Processing video frames...")

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_index += 1
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
    processed_video_path = process_video(local_video_path)

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