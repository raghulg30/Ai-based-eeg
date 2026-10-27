"""CSV loading and dataset validation."""
from __future__ import annotations
import numpy as np
import pandas as pd

META_COLUMNS = {"label", "subject_id", "sampling_rate", "session_id", "segment_id"}


def load_segments(path: str, default_fs: float = 256.0):
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column (0=non-seizure, 1=seizure)")
    signal_cols = [c for c in df.columns if c not in META_COLUMNS]
    if len(signal_cols) < 16:
        raise ValueError("CSV must contain at least 16 numeric sample columns besides metadata")
    labels = pd.to_numeric(df["label"], errors="raise").astype(int).to_numpy()
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must be binary: 0=non-seizure, 1=seizure")
    signals = df[signal_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if "sampling_rate" in df.columns:
        rates = pd.to_numeric(df["sampling_rate"], errors="coerce").fillna(default_fs).to_numpy(float)
    else:
        rates = np.full(len(df), default_fs, dtype=float)
    if np.any(rates <= 0):
        raise ValueError("Sampling rates must be positive")
    subjects = df["subject_id"].astype(str).to_numpy() if "subject_id" in df.columns else None
    return signals, labels, rates, subjects, signal_cols
