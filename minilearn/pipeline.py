"""Chain preprocessing steps and a model into one object."""
from .base import BaseEstimator, clone


class Pipeline(BaseEstimator):
    """Run data through transformers in order, then into a final model.

    Pipeline([("impute", SimpleImputer()),
              ("scale", StandardScaler()),
              ("model", LogisticRegression())])

    The point: every step is fitted on training data only, and new data
    automatically goes through the same steps before predicting.
    """

    def __init__(self, steps):
        self.steps = steps

    @property
    def _estimator_type(self):
        return self.steps[-1][1]._estimator_type    # same kind as the final model

    def _clone(self):
        return Pipeline([(name, clone(step)) for name, step in self.steps])

    def fit(self, X, y=None):
        for _, step in self.steps[:-1]:
            X = step.fit_transform(X, y)            # each step learns, then passes data on
        self.steps[-1][1].fit(X, y)
        return self

    def _transform_all(self, X):
        for _, step in self.steps[:-1]:
            X = step.transform(X)                   # only transform, never re-fit
        return X

    def predict(self, X):
        return self.steps[-1][1].predict(self._transform_all(X))
