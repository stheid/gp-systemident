import gpytorch
import numpy as np
import torch
from sklearn.base import MultiOutputMixin, RegressorMixin, BaseEstimator


class BaseGPR(MultiOutputMixin, RegressorMixin, BaseEstimator):
    def __init__(self):
        self.model = None
        self.likelihood = None

    def predict(self, X):
        self.model.eval()
        self.likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(X)).mean.numpy()
        return pred

    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)

        err = y.numpy() - pred
        return np.sqrt(np.power(err, 2)).mean()
