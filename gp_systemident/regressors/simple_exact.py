import logging

import gpytorch
import numpy as np
import torch
from sklearn.base import RegressorMixin, BaseEstimator, MultiOutputMixin

logger = logging.getLogger(__name__)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, dims):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=dims
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=dims, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class SimpleExactGPR(MultiOutputMixin, RegressorMixin, BaseEstimator):
    def __init__(self):
        self.model = None
        self.likelihood = None

    def fit(self, X, y, training_iter=50):
        d = y.shape[1]

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=d)
        self.model = ExactGPModel(X, y, self.likelihood, d)

        self.likelihood.train()
        self.model.train()

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(X)
            # Calc loss and backprop gradients
            loss = -mll(output, y)
            loss.backward()
            logger.info('Iter %d/%d - Loss: %.3f', i + 1, training_iter, loss.item())
            optimizer.step()

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


class NStateSimpleExactGPR(SimpleExactGPR):
    def __init__(self, d_state, d_act):
        super().__init__()
        self.d_state = d_state
        self.d_act = d_act

    def predict(self, X):
        d = X.shape[1]

        state = X[:, :self.d_state]
        for i in range((d - self.d_state) // self.d_act):
            act = X[:, self.d_state + i * self.d_act: self.d_state + (i + 1) * self.d_act]
            input = torch.cat((state, act), dim=1)
            pred = super().predict(input)
            state += pred
        return (X[:, :self.d_state] - state).numpy()
