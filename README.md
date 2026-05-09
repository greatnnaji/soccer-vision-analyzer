# SoccerVision

SoccerVision is an AI-powered soccer clip analyzer built with Python, OpenCV, YOLO, and Streamlit.

## Overview

SoccerVision takes a soccer video clip as input, runs object detection on the video, and generates an annotated output video showing detected players and the ball.

The goal of this project is to understand the basics of computer vision through a practical sports-related use case. Instead of only classifying images, this project works with video frames, object detection, and simple tracking-style analysis.

## Features

- Upload a soccer video clip
- Detect players and soccer balls using YOLO
- Draw bounding boxes on detected objects
- Generate an annotated output video
- Display basic video and detection statistics
- Provide a simple Streamlit interface for testing

## Tech Stack

- Python
- OpenCV
- YOLO
- Streamlit
- NumPy
- Pandas

## Project Structure

```text
soccer-vision-analyzer/
├── src/
│   └── app.py
├── sample_videos/
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore