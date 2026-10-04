"""
Real-time inference for ISL word recognition.
===============================================
Used by app.py to predict words from live webcam frames.

Maintains a sliding frame buffer, extracts MediaPipe landmarks
in real-time, and runs the trained LSTM classifier.

Usage (standalone test):
    python -m word_recognition.predict --test
"""

import os
import json
import numpy as np
import cv2
import torch

from .config import Config
from .model import ISLWordLSTM

# Lazy-loaded to avoid import-time MediaPipe init
_mp_hands = None
_hands_detector = None


def _init_mediapipe():
    """Lazily initialize MediaPipe Hands."""
    global _mp_hands, _hands_detector
    if _hands_detector is None:
        import mediapipe as mp
        _mp_hands = mp.solutions.hands
        _hands_detector = _mp_hands.Hands(
            static_image_mode=False,  # Video mode for tracking
            max_num_hands=2,
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE,
        )


def _extract_frame_landmarks(frame: np.ndarray) -> np.ndarray:
    """
    Extract hand landmarks from a single BGR frame.
    Returns array of shape (NUM_FEATURES,).
    """
    _init_mediapipe()
    features = np.zeros(Config.NUM_FEATURES, dtype=np.float32)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _hands_detector.process(rgb)

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= 2:
                break
            offset = hand_idx * Config.FEATURES_PER_HAND
            for lm_idx, lm in enumerate(hand_landmarks.landmark):
                base = offset + lm_idx * Config.COORDS_PER_LANDMARK
                features[base] = lm.x
                features[base + 1] = lm.y
                features[base + 2] = lm.z

    return features


class WordPredictor:
    """
    Real-time word predictor for the Streamlit app.

    Maintains a sliding window of frames, extracts landmarks,
    and predicts the word when the buffer is full.
    """

    def __init__(self, model_path: str | None = None, device: str | None = None):
        """
        Args:
            model_path: Path to trained word_lstm.pth
            device:     'cuda' or 'cpu' (auto-detected if None)
        """
        self.model_path = model_path or Config.WORD_MODEL_PATH
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: ISLWordLSTM | None = None
        self.word_classes: dict[str, int] = {}
        self.idx_to_word: dict[int, str] = {}
        self.is_loaded = False

        # Frame buffer for sliding window
        self.frame_buffer: list[np.ndarray] = []
        self.landmark_buffer: list[np.ndarray] = []

        # Load model if available
        self._try_load()

    def _try_load(self) -> bool:
        """Attempt to load the trained model and class mapping."""
        if not os.path.exists(self.model_path):
            print(f"  ⚠️  Word model not found: {self.model_path}")
            print(f"      Train it first:  python -m word_recognition.train")
            return False

        if not os.path.exists(Config.CLASSES_FILE):
            print(f"  ⚠️  Class mapping not found: {Config.CLASSES_FILE}")
            return False

        try:
            # Load class mapping
            with open(Config.CLASSES_FILE, "r") as f:
                self.word_classes = json.load(f)
            self.idx_to_word = {v: k for k, v in self.word_classes.items()}

            # Build and load model
            self.model = ISLWordLSTM(
                num_features=Config.NUM_FEATURES,
                hidden_size=Config.HIDDEN_SIZE,
                num_layers=Config.NUM_LSTM_LAYERS,
                num_classes=len(self.word_classes),
                dropout=Config.DROPOUT,
                bidirectional=Config.BIDIRECTIONAL,
            )
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self.model.to(self.device).eval()
            self.is_loaded = True
            print(f"  ✅ Word model loaded: {len(self.word_classes)} classes")
            return True

        except Exception as e:
            print(f"  ❌ Failed to load word model: {e}")
            self.is_loaded = False
            return False

    def reset_buffer(self):
        """Clear the frame/landmark buffers."""
        self.frame_buffer.clear()
        self.landmark_buffer.clear()

    def add_frame(self, frame: np.ndarray) -> dict | None:
        """
        Add a frame to the buffer and attempt prediction.

        Args:
            frame: BGR numpy array from webcam

        Returns:
            None if buffer not yet full, or a dict:
            {
                "word": str,
                "confidence": float (0-100),
                "top_5": [(word, confidence), ...],
                "buffer_progress": float (0-1),
            }
        """
        if not self.is_loaded:
            return None

        # Extract landmarks for this frame
        landmarks = _extract_frame_landmarks(frame)
        self.landmark_buffer.append(landmarks)

        # Keep buffer at max size (sliding window)
        if len(self.landmark_buffer) > Config.SEQUENCE_LENGTH:
            self.landmark_buffer.pop(0)

        # Return progress if buffer not full
        progress = len(self.landmark_buffer) / Config.SEQUENCE_LENGTH
        if len(self.landmark_buffer) < Config.SEQUENCE_LENGTH:
            return {
                "word": "",
                "confidence": 0.0,
                "top_5": [],
                "buffer_progress": progress,
            }

        # Buffer is full — run prediction
        return self._predict()

    def _predict(self) -> dict:
        """Run LSTM prediction on the current landmark buffer."""
        sequence = np.array(self.landmark_buffer, dtype=np.float32)
        tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)

            # Top-5 predictions
            top5_probs, top5_indices = torch.topk(probs, min(5, probs.size(1)), dim=1)
            top5 = [
                (self.idx_to_word.get(idx.item(), "?"), prob.item() * 100)
                for prob, idx in zip(top5_probs[0], top5_indices[0])
            ]

            best_word = top5[0][0]
            best_conf = top5[0][1]

        return {
            "word": best_word,
            "confidence": best_conf,
            "top_5": top5,
            "buffer_progress": 1.0,
        }

    def predict_from_frames(self, frames: list[np.ndarray]) -> dict | None:
        """
        One-shot prediction from a list of frames.
        Useful for testing or batch inference.
        """
        if not self.is_loaded:
            return None

        self.reset_buffer()
        result = None
        for frame in frames:
            result = self.add_frame(frame)
        return result

    @property
    def available_words(self) -> list[str]:
        """List of all words the model can recognise."""
        return sorted(self.word_classes.keys())

    @property
    def num_classes(self) -> int:
        return len(self.word_classes)


def _test():
    """Quick test: load model and check it works."""
    print("Testing WordPredictor…")
    predictor = WordPredictor()

    if not predictor.is_loaded:
        print("Model not available. Train first with:")
        print("  python -m word_recognition.train")
        return

    print(f"Model loaded with {predictor.num_classes} word classes.")
    print(f"Available words: {predictor.available_words[:20]}…")

    # Generate dummy frames
    dummy_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    for _ in range(Config.SEQUENCE_LENGTH)]
    result = predictor.predict_from_frames(dummy_frames)

    if result:
        print(f"Prediction: {result['word']} ({result['confidence']:.1f}%)")
        print(f"Top 5: {result['top_5']}")
    else:
        print("Prediction failed.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        _test()


if __name__ == "__main__":
    main()
