"""Signal filtering and feature extraction for single-channel EEG segments."""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}


def preprocess_signal(signal: np.ndarray, fs: float, low: float = 0.5, high: float = 45.0,
                      notch: float | None = 50.0) -> np.ndarray:
    """Remove non-finite samples, demean, notch-filter (if valid), then band-pass."""
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 16:
        raise ValueError("Each EEG segment must contain at least 16 samples")
    if not np.isfinite(x).all():
        finite = np.isfinite(x)
        if finite.sum() < 16:
            raise ValueError("Not enough finite samples in EEG segment")
        x = np.interp(np.arange(x.size), np.flatnonzero(finite), x[finite])
    x = x - np.mean(x)
    if notch and 0 < notch < fs / 2:
        b, a = iirnotch(notch, Q=30, fs=fs)
        try:
            x = filtfilt(b, a, x)
        except ValueError:
            pass
    nyquist = fs / 2
    lo = max(low, 0.01)
    hi = min(high, nyquist * 0.95)
    if lo >= hi:
        raise ValueError(f"Invalid bandpass for sampling rate {fs}: low={lo}, high={hi}")
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    try:
        x = sosfiltfilt(sos, x)
    except ValueError:
        # Short segments may not satisfy filtfilt padding requirements.
        pass
    return x


def extract_features(signal: np.ndarray, fs: float) -> dict[str, float]:
    """Extract interpretable time and frequency domain features."""
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 16 or not np.isfinite(x).all():
        raise ValueError("Signal must contain at least 16 finite samples")
    eps = 1e-12
    dx = np.diff(x)
    ddx = np.diff(dx)
    var = float(np.var(x))
    dvar = float(np.var(dx)) if dx.size else 0.0
    ddvar = float(np.var(ddx)) if ddx.size else 0.0
    rms = float(np.sqrt(np.mean(x * x)))
    f, pxx = welch(x, fs=fs, nperseg=min(256, x.size))
    total_power = float(np.trapz(pxx, f)) + eps
    p_norm = pxx / (float(np.sum(pxx)) + eps)
    out: dict[str, float] = {
        "mean": float(np.mean(x)), "std": float(np.std(x)), "variance": var,
        "rms": rms, "peak_to_peak": float(np.ptp(x)),
        "line_length": float(np.mean(np.abs(dx))) if dx.size else 0.0,
        "zero_crossing_rate": float(np.mean(np.diff(np.signbit(x)) != 0)),
        "hjorth_activity": var,
        "hjorth_mobility": float(np.sqrt(dvar / (var + eps))),
        "hjorth_complexity": float(np.sqrt(ddvar / (dvar + eps)) / (np.sqrt(dvar / (var + eps)) + eps)),
        "spectral_entropy": float(-np.sum(p_norm * np.log2(p_norm + eps)) / np.log2(max(len(p_norm), 2))),
        "dominant_frequency": float(f[int(np.argmax(pxx))]),
        "total_power": total_power,
    }
    for name, (low, high) in BANDS.items():
        mask = (f >= low) & (f < min(high, fs / 2))
        out[f"{name}_power"] = float(np.trapz(pxx[mask], f[mask])) if mask.sum() >= 2 else 0.0
    return out
