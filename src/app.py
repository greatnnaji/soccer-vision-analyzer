import streamlit as st

st.set_page_config(page_title="SoccerVision", layout="wide")

st.title("SoccerVision")
st.write("AI-powered soccer clip analyzer using YOLO and OpenCV.")

uploaded_file = st.file_uploader("Upload a soccer clip", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.success("Video uploaded successfully.")
    st.video(uploaded_file)

    st.info("Next step: run YOLO detection on this uploaded video.")