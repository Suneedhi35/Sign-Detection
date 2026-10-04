"""
Word-Level ISL Gesture Recognition Module
==========================================
Video-based word recognition using MediaPipe landmarks + BiLSTM classifier.

Pipeline:
  1. download_data.py    — Download ISL dictionary videos from Google Drive
  2. extract_landmarks.py — Extract MediaPipe hand/pose landmarks from videos
  3. train.py            — Train BiLSTM classifier on landmark sequences
  4. predict.py          — Real-time inference for Streamlit app integration
"""

from .config import Config

__all__ = ["Config"]
