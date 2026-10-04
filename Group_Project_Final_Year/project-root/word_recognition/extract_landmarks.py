"""
Extract MediaPipe hand landmarks from ISL Dictionary videos.
=============================================================
For each video file in data/raw_videos/<folder>/:
  1. Sample SEQUENCE_LENGTH evenly-spaced frames
  2. Run MediaPipe Hands to extract 21 landmarks × 3 coords per hand
  3. Save as .npy array of shape (SEQUENCE_LENGTH, NUM_FEATURES)

Also generates data/word_classes.json mapping word names → class indices.

Usage:
    python -m word_recognition.extract_landmarks
    python -m word_recognition.extract_landmarks --letters A B
    python -m word_recognition.extract_landmarks --visualize
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: mediapipe is required.  Install with:  pip install mediapipe>=0.10.9")
    sys.exit(1)

from .config import Config


# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def _is_video(filename: str) -> bool:
    """Check if a file is a video by extension."""
    return os.path.splitext(filename.lower())[1] in VIDEO_EXTENSIONS


def _sample_frames(video_path: str, n_frames: int) -> list[np.ndarray]:
    """
    Sample n_frames evenly-spaced frames from a video file.
    Returns a list of BGR numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ⚠️  Cannot open: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    # Evenly spaced indices
    if total >= n_frames:
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
    else:
        # If video has fewer frames than needed, repeat the last frame
        indices = list(range(total)) + [total - 1] * (n_frames - total)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        elif frames:
            # Duplicate last successful frame if read fails
            frames.append(frames[-1].copy())

    cap.release()

    # Ensure exactly n_frames
    while len(frames) < n_frames:
        if frames:
            frames.append(frames[-1].copy())
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

    return frames[:n_frames]


def _extract_hand_landmarks(
    frame: np.ndarray,
    hands_detector,
) -> np.ndarray:
    """
    Extract hand landmarks from a single frame.
    Returns array of shape (NUM_FEATURES,) = (126,) for two hands.
    If only one hand is detected, the other is zero-padded.
    If no hands detected, returns all zeros.
    """
    features = np.zeros(Config.NUM_FEATURES, dtype=np.float32)

    # MediaPipe expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= 2:  # Max 2 hands
                break

            offset = hand_idx * Config.FEATURES_PER_HAND
            for lm_idx, lm in enumerate(hand_landmarks.landmark):
                base = offset + lm_idx * Config.COORDS_PER_LANDMARK
                features[base] = lm.x
                features[base + 1] = lm.y
                features[base + 2] = lm.z

    return features


def _get_word_name(filename: str) -> str:
    """
    Extract the word name from a video filename.
    E.g., 'About.mp4' → 'About', 'Good Morning.mp4' → 'Good Morning'
    """
    name = os.path.splitext(filename)[0]
    # Clean up common patterns
    name = name.strip()
    # Remove numbering prefixes like "001_" or "1. "
    import re
    name = re.sub(r"^\d+[\s._-]*", "", name)
    return name if name else filename


def extract_landmarks_for_video(
    video_path: str,
    hands_detector,
) -> np.ndarray | None:
    """
    Extract a landmark sequence from a single video.
    Returns array of shape (SEQUENCE_LENGTH, NUM_FEATURES) or None on failure.
    """
    frames = _sample_frames(video_path, Config.SEQUENCE_LENGTH)
    if not frames:
        return None

    sequence = np.zeros(
        (Config.SEQUENCE_LENGTH, Config.NUM_FEATURES), dtype=np.float32
    )

    for i, frame in enumerate(frames):
        sequence[i] = _extract_hand_landmarks(frame, hands_detector)

    # Check if we got any meaningful landmarks (not all zeros)
    if np.abs(sequence).sum() < 1e-6:
        return None

    return sequence


def extract_all(
    letters: list[str] | None = None,
    skip_existing: bool = True,
):
    """
    Extract landmarks from all videos in data/raw_videos/.
    Generates .npy files in data/landmarks/ and word_classes.json.
    """
    Config.ensure_dirs()

    if not os.path.exists(Config.RAW_VIDEO_DIR):
        print(f"❌ Raw video directory not found: {Config.RAW_VIDEO_DIR}")
        print("   Run download_data first:  python -m word_recognition.download_data")
        return

    # Collect all video files
    video_files = []  # (folder_name, filename, full_path)
    for folder in sorted(os.listdir(Config.RAW_VIDEO_DIR)):
        folder_path = os.path.join(Config.RAW_VIDEO_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        if letters:
            if folder.upper() not in [l.upper() for l in letters]:
                continue
        for fname in sorted(os.listdir(folder_path)):
            if _is_video(fname):
                video_files.append((folder, fname, os.path.join(folder_path, fname)))

    if not video_files:
        print("❌ No video files found in raw_videos/")
        return

    print("=" * 60)
    print("  MediaPipe Landmark Extraction")
    print("=" * 60)
    print(f"  Videos found: {len(video_files)}")
    print(f"  Sequence length: {Config.SEQUENCE_LENGTH} frames")
    print(f"  Features per frame: {Config.NUM_FEATURES}")
    print(f"  Output directory: {Config.LANDMARKS_DIR}")
    print("=" * 60)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE,
    )

    # Process each video
    word_to_class: dict[str, int] = {}
    class_counter = 0
    processed = 0
    skipped = 0
    failed = 0

    for folder, fname, video_path in video_files:
        word_name = _get_word_name(fname)
        # Use folder/word as unique key to handle same word name in different folders
        word_key = f"{folder}_{word_name}"

        # Output path
        out_dir = os.path.join(Config.LANDMARKS_DIR, folder)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{word_name}.npy")

        if skip_existing and os.path.exists(out_path):
            skipped += 1
            # Still register the class
            if word_key not in word_to_class:
                word_to_class[word_key] = class_counter
                class_counter += 1
            continue

        print(f"  [{processed + skipped + failed + 1}/{len(video_files)}] "
              f"{folder}/{fname} → {word_name}")

        sequence = extract_landmarks_for_video(video_path, hands)

        if sequence is not None:
            np.save(out_path, sequence)
            if word_key not in word_to_class:
                word_to_class[word_key] = class_counter
                class_counter += 1
            processed += 1
        else:
            print(f"    ⚠️  No landmarks extracted — skipping")
            failed += 1

    hands.close()

    # Build final class mapping: use clean word names as labels
    # We want each unique video to be a separate class if they represent
    # different words, or the same class if they represent the same word.
    # Re-build mapping based on word name (not folder-specific).
    final_mapping: dict[str, int] = {}
    idx = 0
    for folder in sorted(os.listdir(Config.LANDMARKS_DIR)):
        folder_path = os.path.join(Config.LANDMARKS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        for npy_file in sorted(os.listdir(folder_path)):
            if npy_file.endswith(".npy"):
                word = os.path.splitext(npy_file)[0]
                if word not in final_mapping:
                    final_mapping[word] = idx
                    idx += 1

    Config.save_class_mapping(final_mapping)

    print("\n" + "=" * 60)
    print(f"  ✅ Extraction complete!")
    print(f"     Processed: {processed}")
    print(f"     Skipped (existing): {skipped}")
    print(f"     Failed: {failed}")
    print(f"     Total word classes: {len(final_mapping)}")
    print(f"     Classes saved to: {Config.CLASSES_FILE}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks from ISL videos"
    )
    parser.add_argument(
        "--letters", nargs="*", default=None,
        help="Only process specific letter folders (e.g., A B C)"
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Re-extract even if .npy already exists"
    )
    args = parser.parse_args()

    extract_all(letters=args.letters, skip_existing=not args.force)


if __name__ == "__main__":
    main()
