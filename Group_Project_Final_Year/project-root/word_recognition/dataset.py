"""
PyTorch Dataset for ISL word landmark sequences.
=================================================
Loads pre-extracted .npy landmark files and provides
(tensor, class_index) pairs for training.

Supports:
  - Train/val/test splitting
  - Data augmentation (time jitter, coordinate noise, temporal scaling)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

from .config import Config


class ISLWordDataset(Dataset):
    """
    Dataset of ISL word landmark sequences.

    Each sample is a numpy array of shape (SEQUENCE_LENGTH, NUM_FEATURES)
    stored as a .npy file under data/landmarks/<letter_folder>/<word>.npy.
    """

    def __init__(
        self,
        landmarks_dir: str | None = None,
        classes_file: str | None = None,
        augment: bool = False,
    ):
        """
        Args:
            landmarks_dir: Path to directory containing landmark .npy files.
            classes_file:  Path to word_classes.json.
            augment:       Whether to apply data augmentation.
        """
        self.landmarks_dir = landmarks_dir or Config.LANDMARKS_DIR
        self.classes_file = classes_file or Config.CLASSES_FILE
        self.augment = augment

        # Load class mapping
        if not os.path.exists(self.classes_file):
            raise FileNotFoundError(
                f"Class mapping not found: {self.classes_file}\n"
                "Run extract_landmarks first."
            )
        with open(self.classes_file, "r") as f:
            self.word_to_idx: dict[str, int] = json.load(f)

        self.idx_to_word: dict[int, str] = {v: k for k, v in self.word_to_idx.items()}
        self.num_classes = len(self.word_to_idx)

        # Collect all .npy files with their labels
        self.samples: list[tuple[str, int]] = []  # (npy_path, class_idx)
        self._scan_files()

        if not self.samples:
            raise RuntimeError(
                f"No .npy landmark files found in {self.landmarks_dir}"
            )

    def _scan_files(self):
        """Scan the landmarks directory for .npy files."""
        for folder in sorted(os.listdir(self.landmarks_dir)):
            folder_path = os.path.join(self.landmarks_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in sorted(os.listdir(folder_path)):
                if not fname.endswith(".npy"):
                    continue
                word_name = os.path.splitext(fname)[0]
                if word_name in self.word_to_idx:
                    npy_path = os.path.join(folder_path, fname)
                    class_idx = self.word_to_idx[word_name]
                    self.samples.append((npy_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        npy_path, class_idx = self.samples[idx]
        sequence = np.load(npy_path).astype(np.float32)

        # Ensure correct shape
        if sequence.shape[0] != Config.SEQUENCE_LENGTH:
            # Pad or truncate
            if sequence.shape[0] < Config.SEQUENCE_LENGTH:
                pad = np.zeros(
                    (Config.SEQUENCE_LENGTH - sequence.shape[0], Config.NUM_FEATURES),
                    dtype=np.float32,
                )
                sequence = np.vstack([sequence, pad])
            else:
                sequence = sequence[: Config.SEQUENCE_LENGTH]

        if sequence.shape[1] != Config.NUM_FEATURES:
            # Pad features if needed
            padded = np.zeros(
                (Config.SEQUENCE_LENGTH, Config.NUM_FEATURES), dtype=np.float32
            )
            feat_len = min(sequence.shape[1], Config.NUM_FEATURES)
            padded[:, :feat_len] = sequence[:, :feat_len]
            sequence = padded

        # Apply augmentation
        if self.augment:
            sequence = self._augment(sequence)

        tensor = torch.from_numpy(sequence)
        return tensor, class_idx

    def _augment(self, seq: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a landmark sequence."""
        seq = seq.copy()

        # 1. Coordinate noise (jitter)
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.01, seq.shape).astype(np.float32)
            seq += noise

        # 2. Temporal scaling (speed variation)
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.8, 1.2)
            n = seq.shape[0]
            new_n = int(n * scale)
            if new_n > 0:
                indices = np.linspace(0, n - 1, new_n, dtype=int)
                seq_scaled = seq[indices]
                # Pad/truncate back to original length
                if len(seq_scaled) < n:
                    pad = np.zeros((n - len(seq_scaled), seq.shape[1]), dtype=np.float32)
                    seq_scaled = np.vstack([seq_scaled, pad])
                seq = seq_scaled[:n]

        # 3. Random frame dropout
        if np.random.random() < 0.2:
            n_drop = np.random.randint(1, max(2, seq.shape[0] // 10))
            drop_indices = np.random.choice(seq.shape[0], n_drop, replace=False)
            seq[drop_indices] = 0.0

        # 4. Mirror (flip x-coordinates for left/right hand invariance)
        if np.random.random() < 0.3:
            for hand in range(2):
                offset = hand * Config.FEATURES_PER_HAND
                for lm in range(Config.LANDMARKS_PER_HAND):
                    x_idx = offset + lm * Config.COORDS_PER_LANDMARK
                    seq[:, x_idx] = 1.0 - seq[:, x_idx]  # Flip x

        return seq


def get_splits(
    dataset: ISLWordDataset,
    train_ratio: float = Config.TRAIN_RATIO,
    val_ratio: float = Config.VAL_RATIO,
    test_ratio: float = Config.TEST_RATIO,
    random_seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """
    Split dataset into train, validation, and test subsets.
    Uses stratified splitting to maintain class balance.
    """
    indices = list(range(len(dataset)))
    labels = [dataset.samples[i][1] for i in indices]

    # Check minimum samples per class for stratification
    from collections import Counter
    class_counts = Counter(labels)
    min_count = min(class_counts.values())

    if min_count < 3:
        # Not enough samples for stratified split — use random
        print(f"  ⚠️  Some classes have <3 samples. Using random (non-stratified) split.")
        train_idx, temp_idx = train_test_split(
            indices, test_size=(1.0 - train_ratio), random_state=random_seed
        )
        if len(temp_idx) > 1:
            relative_val = val_ratio / (val_ratio + test_ratio)
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=(1.0 - relative_val), random_state=random_seed
            )
        else:
            val_idx, test_idx = temp_idx, []
    else:
        train_idx, temp_idx, train_lbl, temp_lbl = train_test_split(
            indices, labels,
            test_size=(1.0 - train_ratio),
            stratify=labels,
            random_state=random_seed,
        )
        if len(temp_idx) > 1:
            relative_val = val_ratio / (val_ratio + test_ratio)
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=(1.0 - relative_val),
                stratify=temp_lbl,
                random_state=random_seed,
            )
        else:
            val_idx, test_idx = temp_idx, []

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )
