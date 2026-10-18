"""End-to-end training workflow: load data -> clean -> scale -> train several
models -> compare them on a leaderboard.   Run:  python examples/benchmark.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minilearn import Pipeline
from minilearn.preprocessing import SimpleImputer, StandardScaler
from minilearn.model_selection import train_test_split, cross_val_score
from minilearn.metrics import accuracy_score, precision_score
from minilearn.linear_model import LogisticRegression, LinearSVC
from minilearn.neighbors import KNeighborsClassifier
from minilearn.tree import DecisionTreeClassifier
from minilearn.ensemble import RandomForestClassifier


def make_demo_data(n=400, seed=0):
    """Synthetic dataset: 4 features, 2 classes, ~5% missing values."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, 4)) + y[:, None] * np.array([1.5, -1.0, 0.5, 0.0])
    X[rng.random(X.shape) < 0.05] = np.nan
    return X, y


def main():
    X, y = make_demo_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    candidates = {
        "Logistic Regression": LogisticRegression(),
        "Linear SVM": LinearSVC(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=0),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=0),
    }

    results = []
    for name, model in candidates.items():
        # Every model gets the same cleaning steps in front of it.
        pipe = Pipeline([("impute", SimpleImputer("mean")),
                         ("scale", StandardScaler()),
                         ("model", model)])
        cv = cross_val_score(pipe, X_train, y_train, cv=5, random_state=0)
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)
        results.append((name, cv.mean(), accuracy_score(y_test, predictions),
                        precision_score(y_test, predictions)))

    results.sort(key=lambda row: row[2], reverse=True)   # best test accuracy first
    print(f"{'Model':<22}{'CV acc':>9}{'Test acc':>10}{'Precision':>11}")
    print("-" * 52)
    for name, cv_acc, acc, prec in results:
        print(f"{name:<22}{cv_acc:>9.3f}{acc:>10.3f}{prec:>11.3f}")


if __name__ == "__main__":
    main()
