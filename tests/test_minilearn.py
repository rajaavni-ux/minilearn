"""Simple self-checking tests. Run with:  python tests/test_minilearn.py

Each test uses a case you can verify by hand, so no outside library is needed.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minilearn import Pipeline, clone
from minilearn.preprocessing import SimpleImputer, StandardScaler, MinMaxScaler
from minilearn.model_selection import train_test_split, KFold, cross_val_score
from minilearn.metrics import (accuracy_score, precision_score,
                               mean_squared_error, root_mean_squared_error, r2_score)
from minilearn.linear_model import LinearRegression, LogisticRegression, LinearSVC
from minilearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from minilearn.tree import DecisionTreeClassifier
from minilearn.ensemble import RandomForestClassifier


def make_blobs(n_per_class=100, centers=((0, 0), (4, 4)), seed=0):
    """Well-separated clusters of points, one cluster per class."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(c, 1.0, size=(n_per_class, 2)) for c in centers])
    y = np.repeat(np.arange(len(centers)), n_per_class)
    return X, y


# ---------- metrics ----------
def test_metrics():
    assert accuracy_score([1, 0, 1, 1], [1, 0, 0, 1]) == 0.75
    # predicted positive 3 times, 2 were right -> 2/3
    assert abs(precision_score([1, 0, 1, 1, 0], [1, 1, 1, 0, 0]) - 2 / 3) < 1e-12
    assert abs(mean_squared_error([1, 2, 3], [1, 2, 5]) - 4 / 3) < 1e-12
    assert abs(root_mean_squared_error([1, 2, 3], [1, 2, 5]) - np.sqrt(4 / 3)) < 1e-12
    assert r2_score([1, 2, 3], [1, 2, 3]) == 1.0
    assert precision_score([0, 1, 2, 2], [0, 1, 2, 1], average="macro") > 0


# ---------- preprocessing ----------
def test_imputer():
    X = np.array([[1, np.nan], [3, 4], [5, 6]])
    out = SimpleImputer("mean").fit_transform(X)
    assert out[0, 1] == 5.0 and not np.isnan(out).any()
    assert SimpleImputer("constant", fill_value=-1).fit_transform(X)[0, 1] == -1
    assert SimpleImputer("median").fit_transform(X)[0, 1] == 5.0
    assert np.isnan(X[0, 1])                       # original untouched


def test_scalers():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    Z = StandardScaler().fit_transform(X)
    assert np.allclose(Z.mean(axis=0), 0) and np.allclose(Z.std(axis=0), 1)
    M = MinMaxScaler().fit_transform(X)
    assert M.min() == 0 and M.max() == 1
    scaler = StandardScaler().fit(X)
    assert np.allclose(scaler.inverse_transform(scaler.transform(X)), X)
    # constant column must not cause division by zero
    assert not np.isnan(StandardScaler().fit_transform(np.ones((4, 2)))).any()


# ---------- model selection ----------
def test_split_and_kfold():
    X, y = np.arange(40).reshape(20, 2), np.arange(20)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
    assert len(X_te) == 5 and len(X_tr) == 15
    assert set(y_tr) | set(y_te) == set(y) and not set(y_tr) & set(y_te)

    seen = []
    for train_idx, test_idx in KFold(n_splits=3, shuffle=True, random_state=0).split(X):
        assert not set(train_idx) & set(test_idx)
        seen.extend(test_idx)
    assert sorted(seen) == list(range(20))         # every row tested exactly once


# ---------- models ----------
def test_linear_regression():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 5
    model = LinearRegression().fit(X, y)
    assert np.allclose(model.coef_, [3, -2, 0], atol=1e-8)
    assert abs(model.intercept_ - 5) < 1e-8
    assert model.score(X, y) > 0.999999


def test_classifiers_binary():
    X, y = make_blobs()
    X = StandardScaler().fit_transform(X)
    models = [LogisticRegression(), LinearSVC(), KNeighborsClassifier(),
              DecisionTreeClassifier(max_depth=5, random_state=0),
              RandomForestClassifier(n_estimators=25, random_state=0)]
    for model in models:
        acc = model.fit(X, y).score(X, y)
        assert acc > 0.95, f"{type(model).__name__} accuracy too low: {acc}"


def test_classifiers_multiclass():
    X, y = make_blobs(centers=((0, 0), (6, 0), (3, 6)), seed=1)
    X = StandardScaler().fit_transform(X)
    models = [LogisticRegression(), LinearSVC(), KNeighborsClassifier(),
              DecisionTreeClassifier(max_depth=5), RandomForestClassifier(n_estimators=25, random_state=0)]
    for model in models:
        acc = model.fit(X, y).score(X, y)
        assert acc > 0.93, f"{type(model).__name__} multiclass accuracy too low: {acc}"


def test_string_labels_and_proba():
    X, y = make_blobs()
    labels = np.where(y == 1, "spam", "ham")
    model = LogisticRegression().fit(StandardScaler().fit_transform(X), labels)
    assert set(model.predict(StandardScaler().fit(X).transform(X))) <= {"ham", "spam"}
    assert np.allclose(model.predict_proba(StandardScaler().fit(X).transform(X)).sum(axis=1), 1)


def test_tree_exactness_and_importance():
    # y depends only on feature 0, so the tree should split on it, perfectly
    X = np.array([[1, 7], [2, 3], [3, 9], [10, 4], [11, 8], [12, 1]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    tree = DecisionTreeClassifier().fit(X, y)
    assert (tree.predict(X) == y).all()
    assert tree.tree_.feature == 0 and tree.feature_importances_[0] == 1.0


def test_knn_regressor():
    X = np.array([[0.0], [1.0], [2.0], [10.0]])
    y = np.array([0.0, 1.0, 2.0, 10.0])
    pred = KNeighborsRegressor(n_neighbors=2).fit(X, y).predict([[0.4]])
    assert abs(pred[0] - 0.5) < 1e-12              # neighbours are 0 and 1


# ---------- workflow ----------
def test_pipeline_and_cross_val():
    X, y = make_blobs(seed=3)
    X[::10, 0] = np.nan                            # sprinkle in missing values
    pipe = Pipeline([("impute", SimpleImputer("mean")),
                     ("scale", StandardScaler()),
                     ("model", LogisticRegression())])
    scores = cross_val_score(pipe, X, y, cv=5, random_state=0)
    assert len(scores) == 5 and scores.mean() > 0.9
    assert pipe is not clone(pipe) and not hasattr(clone(pipe).steps[0][1], "statistics_")


def test_errors_are_helpful():
    for bad_call in (lambda: LogisticRegression().predict([[1, 2]]),
                     lambda: LinearRegression().fit([[np.nan]], [1.0])):
        try:
            bad_call()
        except (RuntimeError, ValueError):
            continue
        raise AssertionError("expected a helpful error")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\nAll {len(tests)} tests passed.")
