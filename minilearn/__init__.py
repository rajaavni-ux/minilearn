"""MiniLearn: a from-scratch, NumPy-only machine learning library."""
from . import (preprocessing, model_selection, metrics,
               linear_model, neighbors, tree, ensemble)
from .base import BaseEstimator, clone
from .pipeline import Pipeline

__version__ = "0.1.0"
__all__ = ["BaseEstimator", "clone", "Pipeline", "preprocessing", "model_selection",
           "metrics", "linear_model", "neighbors", "tree", "ensemble"]
