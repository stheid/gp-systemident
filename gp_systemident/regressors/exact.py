import logging

import gpytorch
import torch
from tqdm import trange

from gp_systemident.regressors.base import BaseGPR

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


class ExactMultioutputGPR(BaseGPR):
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

        with trange(training_iter) as t:
            for _ in t:
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = self.model(X)
                # Calc loss and backprop gradients
                loss = -mll(output, y)
                loss.backward()
                t.set_postfix(loss=loss.item())
                optimizer.step()


class NStateExactMultioutputGPR(ExactMultioutputGPR):
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
