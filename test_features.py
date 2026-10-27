import numpy as np
from eeg_seizure.features import preprocess_signal, extract_features


def test_preprocess_and_features_are_finite():
    fs = 256
    t = np.arange(512) / fs
    x = np.sin(2*np.pi*10*t) + 0.2*np.sin(2*np.pi*30*t)
    cleaned = preprocess_signal(x, fs)
    features = extract_features(cleaned, fs)
    assert cleaned.shape == x.shape
    assert features["std"] > 0
    assert all(np.isfinite(list(features.values())))


def test_short_signal_rejected():
    try:
        preprocess_signal(np.ones(5), 256)
    except ValueError:
        return
    raise AssertionError("Expected short signal to be rejected")
