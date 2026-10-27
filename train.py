"""Train and evaluate an EEG segment classifier."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
    classification_report, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .data import load_segments
from .features import preprocess_signal, extract_features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="CSV: one EEG segment per row")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--model", choices=["svm", "random_forest"], default="random_forest")
    p.add_argument("--sampling-rate", type=float, default=256.0)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    outdir = Path(a.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    signals, labels, rates, subjects, signal_cols = load_segments(a.input, a.sampling_rate)
    feature_rows = []
    cleaned_signals = []
    for sig, fs in zip(signals, rates):
        clean = preprocess_signal(sig, float(fs))
        cleaned_signals.append(clean)
        feature_rows.append(extract_features(clean, float(fs)))
    X = pd.DataFrame(feature_rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(np.unique(labels)) < 2:
        raise ValueError("Input must contain both seizure and non-seizure examples")

    split_details = {"strategy": "stratified random segment split", "warning": None}
    if subjects is not None and len(np.unique(subjects)) >= 4:
        # Try group split and ensure both classes are present in train and test.
        splitter = GroupShuffleSplit(n_splits=30, test_size=a.test_size, random_state=a.seed)
        selected = None
        for tr, te in splitter.split(X, labels, groups=subjects):
            if len(np.unique(labels[tr])) == 2 and len(np.unique(labels[te])) == 2:
                selected = (tr, te); break
        if selected:
            train_idx, test_idx = selected
            split_details = {"strategy": "subject-independent GroupShuffleSplit", "warning": None,
                             "train_subjects": sorted(set(subjects[train_idx])),
                             "test_subjects": sorted(set(subjects[test_idx]))}
        else:
            train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=a.test_size,
                random_state=a.seed, stratify=labels)
            split_details = {"strategy": "stratified random segment split fallback", "warning":
                "Could not find a group split with both classes in train and test; results may have subject leakage."}
    else:
        train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=a.test_size,
            random_state=a.seed, stratify=labels)
        split_details["warning"] = "Subject IDs absent or too few unique subjects; results may have subject leakage."

    if a.model == "svm":
        estimator = Pipeline([("scaler", StandardScaler()), ("classifier", SVC(C=1.0, kernel="rbf", probability=True,
                                                                                 class_weight="balanced", random_state=a.seed))])
    else:
        estimator = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2,
                                           random_state=a.seed, n_jobs=-1)
    estimator.fit(X.iloc[train_idx], labels[train_idx])
    pred = estimator.predict(X.iloc[test_idx])
    probs = estimator.predict_proba(X.iloc[test_idx])[:, 1] if hasattr(estimator, "predict_proba") else None
    cm = confusion_matrix(labels[test_idx], pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    specificity = tn / (tn + fp) if tn + fp else 0.0
    test_duration_hours = len(test_idx) * (signals.shape[1] / float(np.median(rates))) / 3600.0
    metrics = {
        "model": a.model, "n_segments": int(len(labels)), "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "class_counts": {"non_seizure": int((labels == 0).sum()), "seizure": int((labels == 1).sum())},
        "accuracy": float(accuracy_score(labels[test_idx], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels[test_idx], pred)),
        "sensitivity_recall": float(recall_score(labels[test_idx], pred, zero_division=0)),
        "specificity": float(specificity), "precision": float(precision_score(labels[test_idx], pred, zero_division=0)),
        "f1_score": float(f1_score(labels[test_idx], pred, zero_division=0)),
        "false_positives": fp, "missed_seizure_segments": fn,
        "missed_seizure_rate": float(fn / (fn + tp)) if fn + tp else None,
        "false_alarms_per_hour_rough": float(fp / test_duration_hours) if test_duration_hours > 0 else None,
        "roc_auc": float(roc_auc_score(labels[test_idx], probs)) if probs is not None and len(np.unique(labels[test_idx])) == 2 else None,
        "split": split_details,
        "interpretation_note": "Synthetic-data metrics are software-demo results only. False alarms/hour is rough unless segments represent continuous monitoring.",
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (outdir / "classification_report.txt").write_text(classification_report(labels[test_idx], pred,
        labels=[0, 1], target_names=["non-seizure", "seizure"], zero_division=0))
    joblib.dump({"model": estimator, "feature_names": list(X.columns), "model_name": a.model}, outdir / "model.joblib")
    fig, ax = plt.subplots(figsize=(5, 4)); im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Non-seizure", "Seizure"]); ax.set_yticks([0, 1], ["Non-seizure", "Seizure"])
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.set_title("Confusion matrix")
    for (i, j), value in np.ndenumerate(cm): ax.text(j, i, str(value), ha="center", va="center")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(outdir / "confusion_matrix.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for label, name in [(0, "Non-seizure"), (1, "Seizure")]:
        ids = np.flatnonzero(labels == label)
        if len(ids): ax.plot(np.arange(min(signals.shape[1], 512)) / rates[ids[0]], cleaned_signals[ids[0]][:512], label=f"Example {name}")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude (input units)"); ax.set_title("Example preprocessed EEG-like segments"); ax.legend(); fig.tight_layout()
    fig.savefig(outdir / "example_segments.png", dpi=160); plt.close(fig)
    if a.model == "random_forest":
        pd.DataFrame({"feature": X.columns, "importance": estimator.feature_importances_}).sort_values("importance", ascending=False).to_csv(outdir / "feature_importance.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved reports and model to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
