"""
Bidirectional LSTM with Attention for ISL Word Classification.
==============================================================
Input:  (batch, seq_len, num_features)  — landmark sequences
Output: (batch, num_classes)            — class logits

Architecture:
  Input → LayerNorm → BiLSTM (2 layers) → Attention Pooling → FC → Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class Attention(nn.Module):
    """
    Learnable attention mechanism over LSTM hidden states.
    Computes a weighted sum of all time-step outputs instead
    of using only the final hidden state.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size) — attention-weighted sum
        """
        # (batch, seq_len, 1)
        attn_weights = self.attention(lstm_output)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted sum: (batch, hidden_size)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context


class ISLWordLSTM(nn.Module):
    """
    Bidirectional LSTM classifier for ISL word recognition
    from MediaPipe hand landmark sequences.
    """

    def __init__(
        self,
        num_features: int = Config.NUM_FEATURES,
        hidden_size: int = Config.HIDDEN_SIZE,
        num_layers: int = Config.NUM_LSTM_LAYERS,
        num_classes: int = 10,  # Set dynamically from word_classes.json
        dropout: float = Config.DROPOUT,
        bidirectional: bool = Config.BIDIRECTIONAL,
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # Direction multiplier
        self.num_directions = 2 if bidirectional else 1
        self.effective_hidden = hidden_size * self.num_directions

        # Input normalization
        self.layer_norm = nn.LayerNorm(num_features)

        # Input projection (optional — helps if num_features is small)
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Attention
        self.attention = Attention(self.effective_hidden)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.effective_hidden, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, num_features)
        Returns:
            logits: (batch, num_classes)
        """
        # Normalize input
        x = self.layer_norm(x)

        # Project input features
        x = self.input_proj(x)

        # LSTM
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size * num_directions)

        # Attention pooling
        context = self.attention(lstm_out)
        # context: (batch, effective_hidden)

        # Classification
        logits = self.classifier(context)
        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(num_classes: int, device: str = "cpu") -> ISLWordLSTM:
    """Build and return the model, moved to the specified device."""
    model = ISLWordLSTM(
        num_features=Config.NUM_FEATURES,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LSTM_LAYERS,
        num_classes=num_classes,
        dropout=Config.DROPOUT,
        bidirectional=Config.BIDIRECTIONAL,
    )
    model = model.to(device)
    print(f"  Model parameters: {model.count_parameters():,}")
    return model
