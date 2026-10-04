"""
Configuration for the Word-Level ISL Recognition Pipeline.
All paths, hyperparameters, and Drive folder IDs live here.
"""

import os
import json

# ──────────────────────────────────────────────────────────────
#  PATHS
# ──────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    """Central configuration object."""

    # Project root
    BASE_DIR = _BASE_DIR

    # Data directories
    DATA_DIR = os.path.join(_BASE_DIR, "data")
    RAW_VIDEO_DIR = os.path.join(DATA_DIR, "raw_videos")
    LANDMARKS_DIR = os.path.join(DATA_DIR, "landmarks")
    CLASSES_FILE = os.path.join(DATA_DIR, "word_classes.json")

    # Model output
    MODELS_DIR = os.path.join(
        _BASE_DIR,
        "AI-Enabled-Sign-Language-Communication-System-main",
        "models",
    )
    WORD_MODEL_PATH = os.path.join(MODELS_DIR, "word_lstm.pth")
    TRAINING_LOG_PATH = os.path.join(MODELS_DIR, "word_training_log.json")

    # CSV with Drive links
    CSV_PATH = os.path.join(_BASE_DIR, "ISL_Dictionary_words.csv")

    # ──────────────────────────────────────────────────────────
    #  MEDIAPIPE SETTINGS
    # ──────────────────────────────────────────────────────────
    # 21 landmarks per hand × 3 coords (x, y, z) = 63 per hand
    # Two hands = 126 features per frame
    LANDMARKS_PER_HAND = 21
    COORDS_PER_LANDMARK = 3
    FEATURES_PER_HAND = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 63
    NUM_FEATURES = FEATURES_PER_HAND * 2  # 126 (both hands)

    # ──────────────────────────────────────────────────────────
    #  VIDEO / SEQUENCE SETTINGS
    # ──────────────────────────────────────────────────────────
    SEQUENCE_LENGTH = 30  # Fixed number of frames sampled per video
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5

    # ──────────────────────────────────────────────────────────
    #  MODEL HYPERPARAMETERS
    # ──────────────────────────────────────────────────────────
    HIDDEN_SIZE = 256
    NUM_LSTM_LAYERS = 2
    DROPOUT = 0.3
    BIDIRECTIONAL = True

    # ──────────────────────────────────────────────────────────
    #  TRAINING HYPERPARAMETERS
    # ──────────────────────────────────────────────────────────
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EARLY_STOP_PATIENCE = 7
    LR_SCHEDULER_PATIENCE = 3
    LR_SCHEDULER_FACTOR = 0.5

    # Train / Val / Test split ratios
    TRAIN_RATIO = 0.80
    VAL_RATIO = 0.10
    TEST_RATIO = 0.10

    # ──────────────────────────────────────────────────────────
    #  GOOGLE DRIVE FOLDER IDS  (parsed from CSV at import)
    # ──────────────────────────────────────────────────────────
    DRIVE_FOLDERS: dict[str, str] = {}  # letter -> folder URL

    @classmethod
    def ensure_dirs(cls):
        """Create all required data directories."""
        for d in [cls.DATA_DIR, cls.RAW_VIDEO_DIR, cls.LANDMARKS_DIR]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def load_drive_links(cls):
        """Parse Drive folder URLs from the ISL dictionary CSV."""
        import pandas as pd
        if not os.path.exists(cls.CSV_PATH):
            print(f"[WARN] CSV not found at {cls.CSV_PATH}")
            return
        df = pd.read_csv(cls.CSV_PATH)
        df.columns = [c.strip() for c in df.columns]
        for _, row in df.iterrows():
            name = str(row["Folder Name"]).strip()
            link = str(row["Link"]).strip()
            if name and name != "All Dictionary Videos" and link:
                cls.DRIVE_FOLDERS[name] = link

    @classmethod
    def load_class_mapping(cls) -> dict[str, int]:
        """Load word → class index mapping from JSON."""
        if os.path.exists(cls.CLASSES_FILE):
            with open(cls.CLASSES_FILE, "r") as f:
                return json.load(f)
        return {}

    @classmethod
    def save_class_mapping(cls, mapping: dict[str, int]):
        """Save word → class index mapping to JSON."""
        cls.ensure_dirs()
        with open(cls.CLASSES_FILE, "w") as f:
            json.dump(mapping, f, indent=2)


# Auto-load Drive links on import
Config.load_drive_links()
