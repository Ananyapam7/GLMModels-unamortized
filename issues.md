## Issues and Bugs

- Need to test on real datasets [Done] (Mean is the same, but variance is different, need to figure out why)
- Test on fish dataset and compare results of our model with Chiquet's model [Done] (Results are quite similar)
- Sometimes there are wierd assertion errors
- Need to add more explicit assertions
- Sometimes the exact same code works and sometimes it doesn't [Resolved] (Added a seed to the code)
- Need to profile the code
- Need to add CI/CD using Travis, Azure, etc.
- Need to rename files to be more explicit
- Need to add more models for expCondLogProb [Resolved] (Added Bernoulli, Binomial, Gamma, Gumbel)
- Need to check the code for the expected conditional log probability and add using Gauss-Hermite quadrature
- Need a way to handle infs, exp overflows, etc. [Resolved] (Using float64)
- Do computations in the log space (See the log sum exp trick)
- Need a way to handle missing data
- Need to add more simulated tests
- Need to add more specific initialization methods
- Need to add more specific priors for learning B
- Need to optimize using the appropriate ELBO for optimization [Resolved] (Added a THRESHOLD variable to choose between the naive and expected elbo)
- Need to update the requirements.txt file and remove the unnecessary packages from the setup.py file and import statements
- Need a way to automatically choose K, the number of latent factors
- Need a metric to automatically determine the fit of the model without knowing the true values
- Need a way to call train using Adam, SGD, or other optimizers and LBFGS [Resolved] (Added a method to call train using Adam and LBF)
- Need to add more count data models - Gumbel, etc. [Resolved]
- Need to check the code for the expected conditional log probability and add using Gauss-Hermite quadrature
- Modularize the code for fixed effects and non-fixed effects [Resolved] (No longer needed)
- Need to correct the code for naive elbo [Resolved]
- Entropy is negative? [Resolved](Differential Entropy can be negative)
- Need to correct the code for expected conditional log probability[Resolved]
- Need to add more documentation
- Need to modify the code to handle the case when the number of latent factors is 1
- Need to modify the code to handle the case when the number of latent factors is 0
- Need to rename stuff using PEP8 guidelines
- Do PCA and take average of the last components as the initialize D to I * avg(last components) [Failed] (Did not work)
- Parametrize D with eps + exp(D) to ensure that D is positive
- Add a prior for all the entries log D with variance 5
- Implement the model in stan [Later]

## Assertions

1. Assertion in initialization of the model - 
        assert torch.all(diag_sigma > 0), 'diag_sigma must be positive' 
2. Assertion in train - 
        - assert torch.all(self.Y + torch.exp(self.phi2)) > 0, "Y + exp(phi2) must be positive"
        - assert torch.all(self.Y + torch.exp(self.phi4)) > 0, "Y + exp(phi4) must be positive"  
        - assert assert det_sigma > 0, "det(sigma) must be positive"
        - assert sigma must be positive definite, and hence det(sigma) > 0
        - assert Sn must be positive definite

