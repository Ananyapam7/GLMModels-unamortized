import sys
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import seaborn as sns

# np.set_printoptions(threshold=sys.maxsize)
# torch.set_printoptions(linewidth=sys.maxsize, profile="full", sci_mode=False)
torch.set_printoptions(precision=3)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

eps = 1e-6

# Debugging
torch.autograd.set_detect_anomaly(True)

class GLMM:
    '''An abstract class that the GLMM models inherit from'''
    def __init__(self, Y, X, K):
        '''Initialize the model
        N: Number of samples
        d: Number of response variables
        Q: Number of covariates
        Y: Response matrix, Nxd
        X: Covariate matrix, NxQ
        K: Number of latent factors

        nn_model: Neural network model
        optimizer: Optimizer

        Parameters
        ----------
        Y : numpy.ndarray
            Response matrix, Nxd
        X : numpy.ndarray
            Covariate matrix, NxQ
        K : int
            Number of latent factors
        Returns
        -------
        None
        '''
        # Data parameters
        self.Y = (torch.from_numpy(Y)).to(torch.float64) # Response vector, Y is Nxd matrix
        self.X = (torch.from_numpy(X)).to(torch.float64) # Data matrix, X is NxQ matrix
        assert self.X.shape[0] == self.Y.shape[0], 'X and Y must have the same number of rows'
        self.N = self.Y.shape[0]
        self.d = self.Y.shape[1]
        self.Q = self.X.shape[1]
        self.K = K # Number of Latent factors

        self.nn_model = None # Neural network model
        self.optimizer = None # Optimizer
        self.converged = True # Convergence flag
        self.error = None # Error type

    def model_params(self):
        '''Return the optimizable model parameters.
    
        Returns:
            A generator object that yields the model parameters to be optimized.
        
        Example:
            >>> model = GLMM()
            >>> params = list(model.model_params())
            >>> len(params)
            4
        '''
        # Yield the model parameters to be optimized
        yield self.mu
        yield self.L
        yield self.log_D_diag
        yield self.B
    
    def compute_elbo_naive(self) -> torch.Tensor:
        '''Compute the evidence lower bound (ELBO) for the GLMM using a naive approach.

        Returns:
            elbo (torch.Tensor): The computed ELBO value.
        
        Raises:
            AssertionError: If Y + exp(phi2) or Y + exp(phi4) is not positive, or if det(sigma) is not positive.

        This function computes the ELBO for the GLMM using a simple approach that assumes independence between the latent variables.
        The ELBO is given by the sum of the log-likelihood, the entropy term and the cross-entropy term.

        The entropy term is calculated from the determinant of the inverse covariance matrix and the trace of the inverse covariance matrix times the precision matrix.

        The cross-entropy term is calculated from the determinant of the covariance matrix, the trace of the precision matrix times the posterior covariance matrix, and the quadratic form of the difference between the mean and the posterior mean of the latent variables.

        The log-likelihood term is computed using the expected conditional log-probability of the observed data given the latent variables.
        '''
        assert torch.all(self.Y + torch.exp(self.phi2)) > 0, "Y + exp(phi2) must be positive"
        assert torch.all(self.Y + torch.exp(self.phi4)) > 0, "Y + exp(phi4) must be positive"
        # self.new_log_D_diag = eps + self.log_D_diag
        self.lamda = torch.exp(self.log_lamda)

        # Compute the sigma matrix
        Lt = torch.t(self.L)
        self.sigma = self.L @ Lt + torch.diag(torch.exp(self.log_D_diag))

        # Compute the inverse of sigma and Sn
        sigma_inv = torch.inverse(self.sigma)
        Sn_inv = torch.diag_embed(self.lamda) + sigma_inv
        assert torch.all(torch.det(Sn_inv)) > 0, "det(Sn_inv) must be positive"
        Sn = torch.inverse(Sn_inv)

        # Compute the mean and covariance
        muplusBX = self.mu + torch.einsum('ij, kj -> ki', self.B , self.X)
        mean_intermediate = self.lamda * self.eta + torch.einsum('ij, kj -> ki', sigma_inv, muplusBX)
        self.mn = torch.einsum('kij, kj -> ki', Sn, mean_intermediate)
        mu_minus_mn = muplusBX - self.mn

        # Compute the entropy and cross entropy terms
        det_sigma = torch.det(self.sigma)
        assert det_sigma > 0, "det(sigma) must be positive"
        entropy = 0.5 * torch.sum(torch.log(torch.det(Sn)))
        trace_sigma_inv_Sn = torch.einsum('ij, ljk -> lik', sigma_inv , Sn).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        crossEntropy = 0.5 * torch.sum((torch.log(det_sigma) + trace_sigma_inv_Sn + torch.einsum('ki , ij, kj -> k', mu_minus_mn , sigma_inv , mu_minus_mn)) )
        
        # Compute the ELBO
        self.Sn_ii = torch.diagonal(Sn, dim1 = -2, dim2 = -1)
        self.rho = self.expCondLogProb()
        self.elbo = self.rho + entropy - crossEntropy

        return self.elbo

    def compute_elbo_efficient(self) -> torch.Tensor:
        """
        Computes the evidence lower bound (ELBO) using the final ELBO.

        Returns:
        - elbo (torch.Tensor): A scalar tensor containing the computed ELBO.

        This function first computes the matrices eta and lambda used in the derivation of the ELBO, as well as the matrix
        sigma. It then proceeds to compute the determinant of sigma and several traces needed to compute the final ELBO.
        Finally, it computes mu_minus_mn, which is used to compute the final term in the ELBO.

        This function assumes that the variables self.mu, self.B, self.X, self.L, self.log_D_diag, self.I_k, self.eta,
        self.lamda, self.sigma, and self.mn have already been initialized or computed.

        Raises:
        - AssertionError: If det(sigma) or det(I_k - C_Ut_RU) are not positive, or if inv_R contain any non-positive
        elements.
        """
        self.lamda = torch.exp(self.log_lamda)
        self.new_log_D_diag = eps + self.log_D_diag
        Lt = torch.t(self.L)
        self.sigma = self.L @ Lt + torch.diag(torch.exp(self.new_log_D_diag))
        # self.sigma = self.L @ Lt + torch.diag(torch.exp(self.log_D_diag))

        ######################################det(sigma)#############################################
        det_D = torch.prod(torch.exp(self.new_log_D_diag))
        inv_D = torch.reciprocal(torch.exp(self.new_log_D_diag)) # inv D as a vector

        # det_D = torch.prod(torch.exp(self.log_D_diag))
        # inv_D = torch.reciprocal(torch.exp(self.log_D_diag)) # inv D as a vector

        Lt_inv_D_L = Lt @ torch.einsum('i, ij -> ij', inv_D, self.L)
        # Lt_inv_D_L2 = Lt @ (self.L * inv_D.unsqueeze(-1))  # Replaced einsum
        # assert torch.all(Lt_inv_D_L == Lt_inv_D_L2)
        inv_C = self.I_k + Lt_inv_D_L
        det_inv_C = torch.det(inv_C)
        det_sigma = det_D * det_inv_C # det(sigma)
        assert det_sigma > 0, "det(sigma) must be positive"
        ##########################################det(S_n)############################################
        inv_R = self.lamda + inv_D # R_inv = lamda + inv(D) as a vector
        assert torch.all(inv_R) > 0, "R_inv must be positive"
        log_inv_R = torch.log(inv_R)
        R = torch.reciprocal(inv_R)
        U = torch.einsum('i, ij -> ij', inv_D, self.L) # Multiplication of a vector (as a diagonal matrix) by a matrix
        # U2 = self.L * inv_D.unsqueeze(-1)  # Replaced einsum
        # assert torch.all(U == U2)
        RU = torch.einsum('ki, ij -> kij', R, U) # Multiplication of a list of vectors (as diagonal matrices) by a matrix
        # RU2 = R.unsqueeze(-1) * U.unsqueeze(0)  # Replaced einsum
        # assert torch.all(RU == RU2)
        Ut = torch.t(U)
        UtR = torch.einsum('ij, kj -> kij', Ut, R) # Multiplication of a matrix by a list of vectors (as diagonal matrices)
        # UtR_alternative = Ut.unsqueeze(0) * R.unsqueeze(1)
        # assert torch.all(UtR == UtR_alternative)
        Ut_RU = torch.einsum('ij, kjl -> kil', Ut, RU) # Multiplication of a matrix by a list of matrices
        # Ut_RU_alternative = torch.bmm(Ut.unsqueeze(0).expand(RU.size(0), -1, -1), RU)
        # assert torch.all(Ut_RU == Ut_RU_alternative)
        C = torch.inverse(inv_C)
        C_Ut_RU = torch.einsum('ij, kjl -> kil', C, Ut_RU) # Multiplication of a matrix by a list of matrices
        # C_Ut_RU_alternative = torch.bmm(C.unsqueeze(0).expand(Ut_RU.size(0), -1, -1), Ut_RU)
        # assert torch.all(C_Ut_RU == C_Ut_RU_alternative)
        det_Ik_minus_C_Ut_RU = torch.det(self.I_k - C_Ut_RU)
        assert torch.all(det_Ik_minus_C_Ut_RU) > 0, "det(I_k - C_Ut_RU) must be positive"

        #############################trace(sigma_inv_S_n)#############################################
        #trace_1
        trace_1 = torch.einsum('bi -> b', (R * inv_D))
        # trace_1_alternative = (R * inv_D).sum(dim=1)
        # assert torch.all(trace_1_alternative==trace_1)

        #trace_2
        Q = inv_D - torch.einsum('j, ij, j -> ij', inv_D, R, inv_D)
        Lt_Q_L = torch.einsum('ij, kj, jl -> kil', Lt, Q, self.L)
        P = torch.inverse(self.I_k + Lt_Q_L)
        Rinv_D_R = torch.einsum('ij, j, ij -> ij', R, inv_D, R)
        # Rinv_D_R_alternative = R * inv_D.view(1, -1) * R
        # assert torch.all(Rinv_D_R == Rinv_D_R_alternative)
        Rinv_D_RU = torch.einsum('ki, ij -> kij', Rinv_D_R, U)
        Ut_Rinv_D_RU = torch.einsum('ij, kjl -> kil', Ut, Rinv_D_RU)
        # Ut_Rinv_D_RU_alternative = torch.bmm(Ut.unsqueeze(0).expand(Rinv_D_RU.size(0), *Ut.size()), Rinv_D_RU)
        # assert torch.all(Ut_Rinv_D_RU == Ut_Rinv_D_RU_alternative)
        trace_2 = torch.einsum('bii -> b' , torch.einsum('ijk, ikl -> ijl', P , Ut_Rinv_D_RU))

        # trace_2_alternative = torch.bmm(P, Ut_Rinv_D_RU).diagonal(dim1=1, dim2=2).sum(dim=1)
        # assert torch.all(trace_2_alternative==trace_2)

        #trace_3
        trace_3 = torch.einsum('bii -> b' , C_Ut_RU)

        # trace_3_alternative = C_Ut_RU.diagonal(dim1=1, dim2=2).sum(dim=1)
        # assert torch.all(trace_3_alternative==trace_3)

        #trace_4
        trace_4 = torch.einsum('bii -> b' , torch.einsum('ijk, ikl, ilm -> ijm', C_Ut_RU, P, Ut_RU))

        # trace_4_alternative = torch.bmm(torch.bmm(C_Ut_RU, P), Ut_RU).diagonal(dim1=1, dim2=2).sum(dim=1)
        # assert torch.all(trace_4_alternative==trace_4)

        trace_sigma_inv_Sn = trace_1 + trace_2 - trace_3 - trace_4
        
        ##################################mu_minus_mn_sigma_inv_mu_minus_mn#############################
        muplusBX = self.mu + torch.einsum('ij, kj -> ki', self.B , self.X)
        # muplusBX_alternative = self.mu + torch.mm(self.X, self.B.t())
        # assert torch.all(muplusBX_alternative==muplusBX)

        Ut_mu = torch.einsum('ij, kj -> ki', Ut , muplusBX)
        C_Ut_mu = torch.einsum('ij, kj -> ki', C , Ut_mu)
        # C_Ut_mu_alternative = torch.mm(Ut_mu, C.t())
        # assert torch.all(C_Ut_mu_alternative==C_Ut_mu)
        U_C_Ut_mu = torch.einsum('ij, kj -> ki', U , C_Ut_mu)
        # U_C_Ut_mu_alternative = torch.mm(C_Ut_mu, U.t())
        # assert torch.all(U_C_Ut_mu_alternative==U_C_Ut_mu)
        sigma_inv_mu = torch.einsum('i, ki -> ki', inv_D , muplusBX) - U_C_Ut_mu
        nu = self.lamda * self.eta + sigma_inv_mu
        R_nu = R*nu
        Ut_R_nu = torch.einsum('ij, kj -> ki', Ut, R_nu)
        P_Ut_R_nu = torch.einsum('ijk, ik -> ij', P, Ut_R_nu)
        # P_Ut_R_nu_alternative = torch.bmm(P, Ut_R_nu.unsqueeze(2)).squeeze(2)
        # assert torch.all(P_Ut_R_nu_alternative==P_Ut_R_nu)
        U_P_Ut_R_nu = torch.einsum('ij, kj -> ki', U, P_Ut_R_nu)
        # U_P_Ut_R_nu_alternative = torch.mm(P_Ut_R_nu, U.t())
        # assert torch.all(U_P_Ut_R_nu_alternative==U_P_Ut_R_nu)
        R_U_P_Ut_R_nu = R * U_P_Ut_R_nu
        self.mn = R_nu + R_U_P_Ut_R_nu
        mu_minus_mn = muplusBX - self.mn
        U_C_Ut_mu_minus_mn = torch.einsum('ij, kj -> ki' , U , torch.einsum('ij, kj -> ki' , C , torch.einsum('ij, kj -> ki', Ut, mu_minus_mn)))
        inv_D_mu_minus_mn = inv_D * mu_minus_mn
        sigma_inv_mu_minus_mn = inv_D_mu_minus_mn - U_C_Ut_mu_minus_mn
        mu_minus_mn_sigma_inv_mu_minus_mn = torch.einsum('ij, ij -> i', mu_minus_mn, sigma_inv_mu_minus_mn)
        
        ############################################Sn_ii###############################################
        ## Extracting the diagonal elements of Sn
        assert torch.all(torch.det(inv_C - Ut_RU)) > 0, "det(inv_C - Ut_RU) must be positive"
        Inv_inv_C_minus_Ut_R_U = torch.inverse(inv_C - Ut_RU)
        Inv_inv_C_minus_Ut_R_U_Ut_R = torch.einsum('kij, kjl -> kil', Inv_inv_C_minus_Ut_R_U, UtR)
        self.Sn_ii = R + torch.einsum('kij, kji -> ki', RU, Inv_inv_C_minus_Ut_R_U_Ut_R)
        # print(self.Sn_ii)
        assert torch.all(self.Sn_ii > 0), "Sn must be positive definite"

        ############################################rho#################################################
        self.rho = self.expCondLogProb()

        ############################################ELBO################################################
        entropy = 0.5 * torch.sum(-torch.sum(log_inv_R, 1) - torch.log(det_Ik_minus_C_Ut_RU)) # Non trivial optimization
        crossEntropy = 0.5 * torch.sum(torch.log(det_sigma) + (trace_sigma_inv_Sn + mu_minus_mn_sigma_inv_mu_minus_mn))
        self.elbo = self.rho + entropy - crossEntropy
        # print(self.elbo)
        return self.elbo

    def compute_elbo(self) -> torch.Tensor:
        '''Compute the Evidence Lower Bound (ELBO) using either the naive or the efficient method depending on the dimensionality of z.

        If the dimensionality d of the latent variables is smaller than a threshold THRES, the naive method is used,
        which computes the ELBO by naively inverting the covariance matrix of the variational distribution q(z).

        If d is greater than or equal to THRES, the efficient method is used, which computes the ELBO by using the
        Woodbury matrix identity to invert the covariance matrix of the variational distribution q(z).

        Returns:
        - elbo: A scalar Tensor containing the ELBO.
        '''
        return self.compute_elbo_efficient()

        # THRES = 7 # Threshold for the decision of using the efficient or naive way
        # if self.d < THRES:
        #     return self.compute_elbo_naive()
        # else:
        #     return self.compute_elbo_efficient()
            # return self.compute_elbo_efficient_prior()

    def create_network(self, layer_sizes):
        layers = []
        prev_size = layer_sizes[0]
        
        # Loop through each layer size in the list and create a linear layer with a Tanh activation
        for size in layer_sizes[1:]:
            layers.append(torch.nn.Linear(prev_size, size))
            layers.append(torch.nn.Tanh())
            prev_size = size
        
        # Construct the network with the layers and return it
        return torch.nn.Sequential(*layers)

    def set_optimizer(self):
        optimizers = {
            'LBFGS': torch.optim.LBFGS,
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'SGD': torch.optim.SGD,
            'RMSprop': torch.optim.RMSprop
        }
        if self.optimizer in optimizers:
            if self.optimizer == 'LBFGS':
                self.optim = optimizers[self.optimizer](self.optimizable_params, history_size=1000, max_iter=self.max_iter)
                self.run_optimizationLBFGS()
            else:
                self.optim = optimizers[self.optimizer](self.optimizable_params, lr=self.lr)
                self.run_optimization()
        else:
            raise ValueError('Optimizer not supported')

    def train(self, optimizer='LBFGS', max_iter=1000, lr = 0.01) -> None:
        '''Train the model using a specified optimizer and hyperparameters.
        Args:
            optimizer (str): Name of the optimizer to use for training. Supported optimizers include 'LBFGS', 'Adam', 'AdamW', 'SGD', and 'RMSprop'.
            max_iter (int): Maximum number of iterations to run the optimizer for.
            lr (float): Learning rate for Adam, AdamW, SGD, and RMSprop optimizers.
        Returns:
            None

        Raises:
            ValueError: If an unsupported optimizer is specified.

        Notes:
            This function sets up the optimizer and its hyperparameters, and runs the optimization loop. If `nn_model` is not None, a neural network with the specified dimensions is created and added to the set of parameters to be optimized. 
        '''
        self.max_iter = max_iter
        self.lr = lr
        self.optimizer = optimizer
        self.optimizable_params = list(self.model_params()) + list(self.var_params())
        
        self.start_time = time.time()
        self.set_optimizer()
        self.end_time = time.time()
        self.time_taken = self.end_time - self.start_time
        
    def run_optimizationLBFGS(self):
        '''Run the optimization for the L-BFGS algorithm'''
        def closure():
            '''Closure function for the optimization'''
            self.optim.zero_grad()
            try:
                loss = -self.compute_elbo()
                loss.backward()
            except AssertionError as e:
                self.error = e
                self.converged = False
                return True
            return loss

        while True:
            self.optim.step(closure)
            if closure():
                break

    def run_optimization(self):
        '''Run the iterative optimization for the optimizers Adam, AdamW, SGD and RMSProp'''
        for self.iter in tqdm(range(self.max_iter)):
            self.optim.zero_grad()
            try:
                loss = -self.compute_elbo()
                loss.backward()
            except AssertionError as e:
                self.error = e
                self.converged = False
                break
            self.optim.step()