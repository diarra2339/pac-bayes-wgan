"""
Parts of this code was taken from the implementation of Prez-Ortix et al.
See their full repository here: https://github.com/mperezortiz/PBB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from utils import get_device


class Gaussian(nn.Module):
    """Implementation of a Gaussian random variable.
    This class is taken from the implementation of Perex-Ortix et al.
    See: https://github.com/mperezortiz/PBB
    """

    def __init__(self, mu, rho, fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)

    @property
    def sigma(self):
        # Computation of standard deviation:
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. Specifically, we use sigma = log(exp(rho)+1)
        return torch.log(1 + torch.exp(self.rho))
        # return torch.exp(self.rho)  # rho should be replaced by log_var. Let's see if it works

    def sample(self):
        # Return a sample from the Gaussian distribution
        epsilon = torch.randn(self.sigma.size()).to(get_device())
        return self.mu + self.sigma * epsilon

    def compute_kl(self, other):
        # Compute KL divergence between two Gaussians (self and other)
        b1 = torch.pow(self.sigma, 2)
        b0 = torch.pow(other.sigma, 2)

        term1 = torch.log(torch.div(b0, b1))
        term2 = torch.div(torch.pow(self.mu - other.mu, 2), b0)
        term3 = torch.div(b1, b0)
        kl_div = (torch.mul(term1 + term2 + term3 - 1, 0.5)).sum()
        return kl_div


class ProbLinLayer(nn.Module):
    def __init__(self, in_features, out_features, rho_prior, rho_init, bias=True, dist='gaussian',
                 init_type='nn_default', prior_params='zeros', init_layer=None, prior_layer=None):
        super(ProbLinLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dist = dist if dist == 'laplace' else 'gaussian'
        self.kl_div = None

        self._set_prior_params(prior_params, rho_prior, prior_layer)
        self._set_init_params(init_type, rho_init, init_layer)
        if not bias:
            self.prior_bias, self.bias = None, None

    def freeze_mean(self):
        self.weight.mu.requires_grad = False
        self.bias.mu.requires_grad = False

    def forward(self, batch, sample=False):
        # during training we sample from the model distribution
        # sample = True can also be set during testing if we want to sample
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample() if self.bias is not None else None
        else:  # otherwise we use the posterior means for weight and bias
            weight = self.weight.mu
            bias = self.bias.mu if self.bias is not None else None
        if self.training:
            # sum of the KL computed for weights and biases
            weight_kl = self.weight.compute_kl(self.prior_weight)
            bias_kl = self.bias.compute_kl(self.prior_bias) if self.bias is not None else 0
            self.kl_div = weight_kl + bias_kl

        return F.linear(batch, weight, bias)

    def _set_prior_params(self, prior_params, rho_prior, prior_layer):
        # parameter needed for nn_default initialization
        k = 1 / np.sqrt(self.in_features)

        # Define the means of the parameters of the prior
        if prior_params == 'zeros':
            prior_weight_mu = torch.zeros(self.out_features, self.in_features)
            prior_bias_mu = torch.zeros(self.out_features)
        elif prior_params == 'nn_default':
            prior_weight_mu = torch.FloatTensor(self.out_features, self.in_features).uniform_(-k, k)
            prior_bias_mu = torch.FloatTensor(self.out_features).uniform_(-k, k)
        elif prior_params == 'normal':
            prior_weight_mu = torch.randn(self.out_features, self.in_features)
            prior_bias_mu = torch.zeros(self.out_features)
        elif prior_params == 'layer':
            if prior_layer is not None:
                prior_weight_mu = copy.deepcopy(prior_layer.weight.data)
                prior_bias_mu = copy.deepcopy(prior_layer.bias.data) if prior_layer.bias is not None else None
            else:
                raise RuntimeError('Please provide a layer for the prior distribution!')
        else:
            raise RuntimeError('The parameters prior_params must be in [zeros, nn_default, normal, layer]')

        # Define the standard deviation of the parameters of the prior
        prior_weight_rho = torch.ones(self.out_features, self.in_features) * rho_prior
        prior_bias_rho = torch.ones(self.out_features) * rho_prior

        # set the weight and bias of the prior
        self.prior_weight = Gaussian(mu=prior_weight_mu, rho=prior_weight_rho, fixed=True)
        self.prior_bias = Gaussian(mu=prior_bias_mu, rho=prior_bias_rho, fixed=True)

    def _set_init_params(self, init_type, rho_init, init_layer):
        # parameter needed for nn_default initialization
        k = 1 / np.sqrt(self.in_features)

        # Initialize the parameters of the layer (soon-to-be posterior)
        if init_type == 'zeros':
            weight_mu = torch.zeros(self.out_features, self.in_features)
            bias_mu = torch.zeros(self.out_features)
        elif init_type == 'nn_default':
            weight_mu = torch.FloatTensor(self.out_features, self.in_features).uniform_(-k, k)
            bias_mu = torch.FloatTensor(self.out_features).uniform_(-k, k)
        elif init_type == 'normal':
            weight_mu = torch.randn(self.out_features, self.in_features)
            bias_mu = torch.zeros(self.out_features)
        elif init_type == 'layer':
            if init_layer is not None:
                weight_mu = copy.deepcopy(init_layer.weight.data)
                bias_mu = copy.deepcopy(init_layer.bias.data) if init_layer.bias is not None else None
            else:
                raise RuntimeError('Please provide a layer for the initial distribution!')
        else:
            raise RuntimeError('The parameters init_type must be in [zeros, nn_default, normal, layer]')

        # Define the standard deviation of the parameters
        weight_rho = torch.ones(self.out_features, self.in_features) * rho_init
        bias_rho = torch.ones(self.out_features) * rho_init

        # set the weight and bias of the (soon-to-be) posterior
        self.weight = Gaussian(mu=weight_mu, rho=weight_rho, fixed=False)
        self.bias = Gaussian(mu=bias_mu, rho=bias_rho, fixed=False)
