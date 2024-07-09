import torch
from glmm import GLMM
import numpy as np
from typing import Tuple

# Set the precision of the print statements to 3 decimal places
torch.set_printoptions(precision=3)

class PoissonGLMM(GLMM):
    '''Poisson GLMM model'''
    def __init__(self, Y, X, K):
        super().__init__(Y, X, K)
        self.init_model_params()
        self.init_var_params()

    def var_params(self):
        '''Return the optimizable variational parameters.
    
        Returns:
            A generator object that yields the variational parameters to be optimized.
        '''
        yield self.eta
        yield self.log_lamda

    def init_model_params(self) -> None:
        self.mu = (torch.randn(self.d)).to(torch.float64).requires_grad_()
        self.L = (torch.randn(self.d, self.K)).to(torch.float64).requires_grad_()
        self.log_D_diag = (torch.randn(self.d)).to(torch.float64).requires_grad_()
        self.B = (torch.randn(self.d, self.Q)).to(torch.float64).requires_grad_()
        self.I_k = torch.eye(self.K).to(torch.float64)

    def expCondLogProb(self) -> torch.Tensor:
        """
        Calculates the expected conditional log probability of the model given the observed data.

        Returns:
            A tensor representing the expected conditional log probability of the model.
        """
        return torch.sum(self.Y * self.mn - torch.mul(torch.exp(self.mn - 100000) , torch.exp(self.Sn_ii/2)))
    
    def init_var_params(self):
        '''Initializes the variational parameters
        Example:
            >>> model = GLMM()
            >>> model.init_var_params()
        '''
        # eta and lamda are N x d tensors
        self.eta = torch.randn(self.N, self.d).to(torch.float64).requires_grad_()
        # lamda is a N x d tensor which is positive from uniform distribution
        self.log_lamda = torch.randn(self.N, self.d).to(torch.float64).requires_grad_()
    
if __name__ == '__main__':
    torch.manual_seed(100)
    np.random.seed(100)

    N = 10000
    d = 5
    K = 3
    Q = 3
    X = torch.randn(N, Q)/10
    B = torch.zeros(d, Q)
    mu = torch.randn(d)/10
    L = torch.randn(d, K)/10
    D  = torch.abs(torch.randn(d))/10
    sigma = L @ L.T + torch.diag(D)

    print("True Parameters")
    print("Mu: ", mu)
    print("Sigma: ", sigma)
    print("B: ", B)
    print("-------------------------")

    mu_broadcasted = mu + (B @ X.T).T

    # Create a multivariate normal distribution
    normal_sampler = torch.distributions.MultivariateNormal(loc=mu_broadcasted, covariance_matrix=sigma)

    # Sample all z_i at once (shape: N x d)
    z_samples = normal_sampler.sample()

    # Transform to get mu_yi using the exponential function (shape: N x d)
    mu_yi = torch.exp(z_samples)

    # Sample from the Poisson distribution
    Y = torch.poisson(mu_yi)

    # Ensure Y is of type torch.float64
    Y = Y.to(torch.float64)

    print("Y: \n", Y)

    model = PoissonGLMM(Y.numpy(), X.numpy(), K=K)
    model.train(max_iter=500, lr= 0.001, optimizer = 'Adam') # Optimze eta and lamda directly

    # Print MSE for each parameter
    print("Mean Squared Error for sigma: ", torch.mean((model.sigma - sigma)**2).item())
    print("Mean Squared Error for mu: ", torch.mean((model.mu - mu)**2).item())
    print("Mean Squared Error for B: ", torch.mean((model.B - B)**2).item())

    # Print the time taken for training
    print("Time taken: ", model.time_taken)
    print("ELBO: ", model.elbo.item())
    print("-------------------------")
    print("Model Parameters")
    print(model.mu)
    print(model.sigma)
    print(model.B)
    print("-------------------------")
    print("Eta and Lambda")
    print("Eta: ", model.eta)
    print("Lambda: ", torch.exp(model.log_lamda))
    print("-------------------------")