import torch
from glmm import GLMM
import numpy as np
from typing import Tuple
from torch.distributions.multivariate_normal import MultivariateNormal

class BernoulliGLMM(GLMM):
    '''Bernoulli GLMM model'''
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
        return torch.sum((2*self.Y - 1) * torch.distributions.Normal(0,1).cdf((0.626*self.mn)/(torch.sqrt(1+((torch.pi)/5.35)*self.Sn_ii))) ) # this is a d dimensional vector
    
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
    
    N = 100
    d = 3
    K = 2
    Q = 1
    X = torch.randn(N, Q)
    B = torch.randn(d, Q)
    mu = torch.randn(d)
    L = torch.randn(d, K)
    D  = torch.exp(torch.randn(d))
    sigma = L @ L.T + torch.diag(D)

    mu = mu.to(torch.float64)
    sigma = sigma.to(torch.float64)
    B = B.to(torch.float64)
    X = X.to(torch.float64)

    # Broadcast mu to match the batch size N
    mu_broadcasted = mu + (B @ X.T).T

    # Create a multivariate normal distribution
    normal_sampler = MultivariateNormal(loc=mu_broadcasted, covariance_matrix=sigma)

    # Sample all z_i at once (shape: N x d)
    z_samples = normal_sampler.sample()

    # Transform to get mu_yi using the sigmoid function (shape: N x d)
    mu_yi = torch.sigmoid(z_samples)

    # Sample from the Bernoulli distribution
    Y = torch.bernoulli(mu_yi)

    # Ensure Y is of type torch.float64
    Y = Y.to(torch.float64)

    print(Y)
    # Fit the model
    model = BernoulliGLMM(Y=Y.numpy(), X=X.numpy(), K=2) # K is the number of latent factors
    model.init_model_params()
    model.init_var_params()
    model.train(max_iter=500, lr= 0.001, optimizer = 'Adam')

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