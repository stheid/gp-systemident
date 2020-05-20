import gpytorch
import torch
from tqdm import trange, tqdm

from gp_systemident.regressors.base import BaseGPR


class VarMultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, dims):
        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([dims])
        )

        # We have to wrap the VariationalStrategy in a MultitaskVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.MultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ), num_tasks=dims
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([dims]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([dims])),
            batch_shape=torch.Size([dims])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalMultioutputGPR(BaseGPR):
    def fit(self, X, y, training_iter=20, points=16):
        d_in = X.shape[1]
        d_out = y.shape[1]

        # The shape of the inducing points should be (2 x m x 1) - so that we learn different inducing
        # points for each output
        inducing_points = torch.rand(d_out, points, d_in)
        self.model = VarMultitaskGPModel(inducing_points, d_out)

        # We're going to use a multitask likelihood with this model
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=d_out)

        self.likelihood.train()
        self.model.train()

        # We use SGD here, rather than Adam. Empirically, we find that SGD is better for variational regression
        optimizer = torch.optim.SGD([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.01)

        # Training loader
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y))  # noqa

        # Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=y.size(0))

        # We use more CG iterations here because the preconditioner introduced in the NeurIPS paper seems to be less
        # effective for VI.
        with trange(training_iter, desc="Epoch") as epochs:
            for _ in epochs:
                mavg_loss = None
                # Within each iteration, we will go over each minibatch of data
                with tqdm(train_loader, desc="Minibatch", leave=False) as minibatch_iter:
                    for x_batch, y_batch in minibatch_iter:
                        optimizer.zero_grad()
                        output = self.model(x_batch)
                        loss = -mll(output, y_batch)
                        mavg_loss = loss.item() if mavg_loss is None else (4 * mavg_loss + loss.item()) / 5
                        minibatch_iter.set_postfix(moving_avg_loss=mavg_loss)
                        loss.backward()
                        optimizer.step()
                epochs.set_postfix(loss=mavg_loss)
