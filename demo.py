"""Generate clearly synthetic EEG-like segments for a software demo only."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def make_demo(output: str, n_subjects: int = 12, segments_per_subject: int = 30,
              samples: int = 512, fs: float = 256.0, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / fs
    rows = []
    for s in range(n_subjects):
        subject = f"S{s+1:02d}"
        for i in range(segments_per_subject):
            label = int(rng.random() < 0.25)
            phase = rng.uniform(0, 2*np.pi)
            x = 0.6*np.sin(2*np.pi*rng.uniform(6, 12)*t + phase)
            x += 0.25*np.sin(2*np.pi*rng.uniform(15, 28)*t + phase/2)
            x += rng.normal(0, 0.35, samples)
            if label:
                # Synthetic rhythmic burst: only to exercise the ML pipeline.
                start = rng.integers(samples//5, samples//2)
                end = min(samples, start + rng.integers(samples//4, samples//2))
                x[start:end] += 1.4*np.sin(2*np.pi*rng.uniform(3, 6)*t[start:end])
            row = {"subject_id": subject, "label": label, "sampling_rate": fs}
            row.update({f"sample_{j:04d}": float(v) for j, v in enumerate(x)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} synthetic segments to {output}. Do not treat these as clinical data.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/demo_eeg.csv")
    p.add_argument("--subjects", type=int, default=12)
    p.add_argument("--segments-per-subject", type=int, default=30)
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--sampling-rate", type=float, default=256.0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    make_demo(a.output, a.subjects, a.segments_per_subject, a.samples, a.sampling_rate, a.seed)

if __name__ == "__main__":
    main()
