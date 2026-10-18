# MiniLearn Core ML Engine

Pure Python + NumPy. No scikit-learn, no TensorFlow.

    python tests/test_minilearn.py     # run the checks
    python examples/benchmark.py       # end-to-end leaderboard demo

Every estimator follows the same pattern: `model.fit(X, y)` then `model.predict(X)`.
