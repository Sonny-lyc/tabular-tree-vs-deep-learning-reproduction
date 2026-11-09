"""Small, from-scratch reproduction of two experiments in Grinsztajn et al. (2022).

Paper: "Why do tree-based models still outperform deep learning on tabular data?"
NeurIPS 2022 Datasets and Benchmarks Track.

This script is a deliberately small demonstration, not the full 20,000 compute-hour
benchmark. It reproduces, on the `electricity` dataset from OpenML:

  1. Finding 2 -- robustness to uninformative features:
     (a) removing the least important features;
     (b) adding Gaussian noise features.
  2. Finding 3 -- sensitivity to random orthogonal rotations of the input space.

Run:  python reproduce.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import special_ortho_group
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, QuantileTransformer


SEED = 0
RNG = np.random.default_rng(SEED)


def make_models() -> dict[str, object]:
    return {
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 256),
            activation="relu",
            alpha=1e-4,
            batch_size=512,
            learning_rate_init=1e-3,
            max_iter=250,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.1,
            random_state=SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=SEED,
            n_jobs=-1,
        ),
        "HistGB": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.1,
            random_state=SEED,
        ),
    }


def load_electricity(n_train: int = 10_000, random_state: int = SEED):
    data = fetch_openml(data_id=151, as_frame=True, parser="auto")
    X = data.frame.drop(columns=["class"])
    y = LabelEncoder().fit_transform(data.frame["class"])

    # Keep the paper's medium-size training setting (~10k samples).
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        train_size=n_train,
        random_state=random_state,
        stratify=y,
    )
    return X_tr, X_te, y_tr, y_te


def gaussianize(X_tr, X_te):
    qt = QuantileTransformer(
        n_quantiles=1000,
        output_distribution="normal",
        random_state=SEED,
    )
    X_tr = qt.fit_transform(X_tr)
    X_te = qt.transform(X_te)
    return np.asarray(X_tr), np.asarray(X_te)


def evaluate(X_tr, y_tr, X_te, y_te, models):
    scores = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        scores[name] = accuracy_score(y_te, model.predict(X_te))
    return scores


def experiment_remove_features(out_dir: Path):
    X_tr, X_te, y_tr, y_te = load_electricity()
    X_tr_g, X_te_g = gaussianize(X_tr, X_te)

    # Feature importance for ranking, computed with a random forest.
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(X_tr_g, y_tr)
    order = np.argsort(rf.feature_importances_)[::-1]  # most -> least important

    fractions = [0.0, 0.2, 0.4, 0.6]
    rows = []
    for frac in fractions:
        n_remove = int(round(frac * X_tr_g.shape[1]))
        keep = order[: X_tr_g.shape[1] - n_remove] if n_remove else order
        scores = evaluate(
            X_tr_g[:, keep],
            y_tr,
            X_te_g[:, keep],
            y_te,
            make_models(),
        )
        for name, acc in scores.items():
            rows.append(
                {
                    "experiment": "remove_features",
                    "fraction_removed": frac,
                    "n_features_kept": len(keep),
                    "model": name,
                    "accuracy": acc,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "remove_features.csv", index=False)
    print(df.pivot(index="n_features_kept", columns="model", values="accuracy"))
    return df


def experiment_add_features(out_dir: Path):
    X_tr, X_te, y_tr, y_te = load_electricity()
    X_tr_g, X_te_g = gaussianize(X_tr, X_te)
    k_values = [0, 5, 10, 20, 50]
    rows = []
    for k in k_values:
        rng = np.random.default_rng(SEED)
        X_tr_aug = np.hstack([X_tr_g, rng.normal(size=(X_tr_g.shape[0], k))])
        X_te_aug = np.hstack([X_te_g, rng.normal(size=(X_te_g.shape[0], k))])
        scores = evaluate(X_tr_aug, y_tr, X_te_aug, y_te, make_models())
        for name, acc in scores.items():
            rows.append(
                {
                    "experiment": "add_features",
                    "n_added": k,
                    "model": name,
                    "accuracy": acc,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "add_features.csv", index=False)
    print(df.pivot(index="n_added", columns="model", values="accuracy"))
    return df


def experiment_rotations(out_dir: Path):
    X_tr, X_te, y_tr, y_te = load_electricity()
    X_tr_g, X_te_g = gaussianize(X_tr, X_te)
    d = X_tr_g.shape[1]
    rows = []
    for seed in range(5):
        # Original (no rotation).
        scores = evaluate(X_tr_g, y_tr, X_te_g, y_te, make_models())
        for name, acc in scores.items():
            rows.append(
                {
                    "condition": "original",
                    "seed": seed,
                    "model": name,
                    "accuracy": acc,
                }
            )
        # Random orthogonal rotation.
        Q = special_ortho_group.rvs(d, random_state=seed)
        X_tr_r = X_tr_g @ Q
        X_te_r = X_te_g @ Q
        scores = evaluate(X_tr_r, y_tr, X_te_r, y_te, make_models())
        for name, acc in scores.items():
            rows.append(
                {
                    "condition": "rotated",
                    "seed": seed,
                    "model": name,
                    "accuracy": acc,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "rotations.csv", index=False)
    summary = df.groupby(["condition", "model"])["accuracy"].agg(["mean", "std"])
    print(summary)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=== Experiment 1a: remove least important features ===")
    experiment_remove_features(args.out)
    print("\n=== Experiment 1b: add Gaussian noise features ===")
    experiment_add_features(args.out)
    print("\n=== Experiment 2: random rotations ===")
    experiment_rotations(args.out)


if __name__ == "__main__":
    main()
