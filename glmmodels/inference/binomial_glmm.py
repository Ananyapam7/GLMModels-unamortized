import torch
from glmm import GLMM
import numpy as np
from typing import Tuple
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.binomial import Binomial
from scipy.special import roots_hermite 

class BinomialGLMM(GLMM):
    '''Binomial GLMM model'''
    def __init__(self, Y, X, K):
        super().__init__(Y, X, K)
        self.init_model_params()
        self.init_var_params()
        self.init_gauss_hermite()

    def init_gauss_hermite(self):
        self.num_roots = 10
        self.roots, self.weights = roots_hermite(self.num_roots)
        self.roots = torch.tensor(self.roots, dtype=torch.float64)
        self.roots = self.roots.view(-1, 1, 1)
        self.weights = torch.tensor(self.weights, dtype=torch.float64)
        self.weights = self.weights.view(-1, 1, 1)

    def init_model_params(self) -> None:
        self.mu = (torch.randn(self.d)).to(torch.float64).requires_grad_()
        self.L = (torch.randn(self.d, self.K)).to(torch.float64).requires_grad_()
        self.log_D_diag = (torch.randn(self.d)).to(torch.float64).requires_grad_()
        self.B = (torch.randn(self.d, self.Q)).to(torch.float64).requires_grad_()
        self.I_k = torch.eye(self.K).to(torch.float64)
        # Number of draws
        self.M = torch.full((self.N, self.d), 10, dtype=torch.float64).requires_grad_()

    def model_params(self):
        yield self.mu
        yield self.L
        yield self.log_D_diag
        yield self.M
        if self.weights:
            yield self.B

    def expCondLogProb(self) -> torch.Tensor:
        """
        Calculates the expected conditional log probability of the model given the observed data.

        Returns:
            A tensor representing the expected conditional log probability of the model.
        """
        self.scaled_roots = - (torch.sqrt(torch.tensor(2.0) * self.Sn_ii) * self.roots + self.mn)
        self.scaled_weights = self.weights / torch.sqrt(torch.tensor(torch.pi))
        return torch.sum(torch.log(torch.sigmoid(self.scaled_roots)) * self.scaled_weights)
    
    def computeEtaLamda(self) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Computes the eta and lambda parameters used in the ELBO calculation.

        If a neural network model is provided, the method takes in Y and a transformed 
        version of Y as inputs, concatenates them, and feeds them into the neural network. 
        The output of the neural network is split into two tensors of length N, with the 
        first tensor being added to a transformed version of Y to obtain the eta parameter, 
        and the second tensor being exponentiated and added to a transformed version of Y 
        to obtain the lambda parameter.

        If no neural network model is provided, the method computes eta and lambda using 
        the phi1, phi2, phi3, and phi4 parameters.

        Returns:
        - eta: A tensor of length N containing the eta parameters.
        - lamda: A tensor of length N containing the lambda parameters.
        '''
        if self.nn_model is not None:
            y_log = torch.log(0.5 + self.Y)
            nn_input = torch.cat((self.Y, torch.log(1 + self.Y)))
            out1, out2 = torch.split(self.nn_model(nn_input.float()), self.N, dim=0)
            eta = out1 + y_log
            lamda = torch.exp(out2 + y_log)
        else:
            eta = self.phi1 + torch.log(torch.exp(self.phi2) + self.Y)
            lamda = torch.exp(self.phi3 * torch.log(torch.exp(self.phi4) + self.Y))
        return eta, lamda
    
    def init_var_params(self):
        '''Initializes the variational parameters
        - phi1: A d-dimensional tensor representing the first variational parameter.
        - phi2: A d-dimensional tensor representing the second variational parameter.
        - phi3: A d-dimensional tensor representing the third variational parameter.
        - phi4: A d-dimensional tensor representing the fourth variational parameter.
        - I_k: A K x K-dimensional identity self.sigma_hat.

        Example:
            >>> model = GLMM()
            >>> model.init_var_params()
        '''
        self.phi1 = torch.ones(self.d).to(torch.float64).requires_grad_()
        self.phi2 = (-3 * torch.ones(self.d)).to(torch.float64).requires_grad_()
        self.phi3 = torch.ones(self.d).to(torch.float64).requires_grad_()
        self.phi4 = (-3 * torch.ones(self.d)).to(torch.float64).requires_grad_()
    
if __name__ == '__main__':
    
    N = 1000
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

    # setting the number of draws to 10 for each element
    M = torch.full((N, d), 10, dtype=torch.float64)

    Y = Binomial(total_count=M, probs=mu_yi).sample()

    # Ensure Y is of type torch.float64
    Y = Y.to(torch.float64)

    print(Y)
    # # Fit the model
    model = BinomialGLMM(Y=Y.numpy(), X=X.numpy(), K=2) # K is the number of latent factors
    model.train(max_iter=500, lr= 0.001, optimizer = 'Adam', weights=True, nn_model=[10, 10])