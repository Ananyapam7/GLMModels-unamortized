import torch
from glmm import GLMM
import numpy as np
from typing import Tuple

from torch.distributions import MultivariateNormal, Gamma

class GammaGLMM(GLMM):
    '''Gamma GLMM model'''
    def __init__(self, Y, X, K):
        super().__init__(Y, X, K)
        self.init_model_params()
        self.init_var_params()

    def model_params(self):
        '''Returns the model parameters'''
        yield self.mu
        yield self.L
        yield self.log_D_diag
        yield self.alpha
        if self.weights:
            yield self.B
    
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
    
    def expCondLogProb(self) -> torch.Tensor:
        '''Computes the expected conditional log probability'''
        return torch.sum(self.alpha * self.mn - self.alpha * self.Y * torch.exp(self.mn + self.Sn_ii/2))
    
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
            eta  = self.phi1 + self.phi2 * self.Y
            lamda = self.phi3 + self.phi4 * self.Y
        return eta, lamda

    def init_model_params(self) -> None:
        self.mu = (torch.randn(self.d)).to(torch.float64).requires_grad_()
        self.L = (torch.randn(self.d, self.K)).to(torch.float64).requires_grad_()
        self.log_D_diag = (torch.randn(self.d)).to(torch.float64).requires_grad_()
        self.B = (torch.randn(self.d, self.Q)).to(torch.float64).requires_grad_()
        self.I_k = torch.eye(self.K).to(torch.float64)
        # Shape parameter
        self.alpha = torch.ones(N, d).to(torch.float64).requires_grad_()

if __name__ == '__main__':
    
    torch.manual_seed(100)
    np.random.seed(100)

    N = 10000
    d = 5
    K = 3
    Q = 3
    X = torch.randn(N, Q)
    B = torch.zeros(d, Q)
    mu = torch.randn(d)
    L = torch.randn(d, K)
    D  = torch.exp(torch.randn(d))
    sigma = L @ L.T + torch.diag(D)

    mu_broadcasted = mu + (B @ X.T).T

    normal_sampler = MultivariateNormal(loc=mu_broadcasted, covariance_matrix=sigma)

    z_samples = normal_sampler.sample()

    mu_yi = torch.exp(z_samples)

    # Define alpha for Gamma distribution
    # alpha is a N x d tensor
    # create a tensor of shape N x d with all elements equal to 1
    alpha = torch.ones(N, d)

    gamma_dist = Gamma(concentration=alpha, rate=mu_yi)

    Y = gamma_dist.sample()

    Y = Y.to(torch.float64)

    # print(Y)

    model = GammaGLMM(Y.numpy(), X.numpy(), K=K)
    model.train(max_iter=500, lr= 0.001, optimizer = 'Adam', weights=False, nn_model=[10, 10])