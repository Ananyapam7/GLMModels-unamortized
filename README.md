# glmmodels

> glmmodels contains a collection of generalized linear mixed models (GLMMs) for modelling and performing factor analysis of count data in Python. 
> This package implements efficient algorithms to fit such models accompanied with tools for visualization and diagnostic.

## Dependencies

The code requires the following packages:

* `torch` >= 1.0.0
* `numpy` >= 1.14.0
* `pandas` >= 0.22.0

## Installation

The code requires Python 3.6 or later. To install the required packages, run

```bash
pip install -r requirements.txt
pip install glmmodels
```

## Usage

The package contains the following models: 

* `PoissonGLMM` - Poisson GLMM with a log link function
* `BernoulliGLMM` - Bernoulli GLMM with a logit link function
* `BinomialGLMM` - Binomial GLMM with a logit link function
* `GammaGLMM` - Gamma GLMM with a log link function
* `GumbelGLMM` - Gumbel GLMM with a log link function

The following code snippet shows how to use the package to fit a Poisson GLMM on the `barents` dataset.

```python
import numpy as np
import pandas as pd
from glmmodels import PoissonGLMM

# Load the barents dataset
barents = pd.read_csv('data/barents.csv')
barents = barents.dropna()
barentsY = barents.iloc[: , :30] # Response (counts)
barentsX = barents.iloc[: , 31:34] # Covariates (fixed effects)

# Fit the model
model = PoissonGLMM(barentsY, barentsX, K=2, verbose=True) # K latent factors
model.train(max_iter=1000) # Train the model using the default optimizer (L-BFGS)

# Print the model parameters
print(model.mu)
print(model.L)
print(model.D)
print(model.sigma)
print(model.B)
```

The optimization is done by default using the L-BFGS algortihm. To use a different optimizer, use the `optimizer` argument in the `train` method. For example, to use stochastic gradient descent (SGD) with a learning rate of 0.01, use

```python
model.train(optimizer='SGD', lr=0.01) # To use SGD with a learning rate of 0.01
```

To use a neural network for learning the pan sample variational parameters, use the `nn` argument in the `train` method by passing a list of hidden layer sizes. For example, to use a neural network with 2 hidden layers of size 10, use

```python
model.train(nn=[10, 10]) # To use a neural network with 2 hidden layers of size 10
```