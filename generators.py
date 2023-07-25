import torch
import torch.nn as nn

from lnets.models.layers.dense.bjorck_linear import BjorckLinear
from lnets.models.activations.group_sort import GroupSort

from prob_nn import ProbLinLayer
from utils import default_config


class DetGen4mlpLin(nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_dim=2):
        super(DetGen4mlpLin, self).__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        # build the layers
        self.layer_1 = nn.Linear(in_features=latent_dim, out_features=hidden_dim)
        self.activation_1 = nn.ReLU(True)
        self.layer_2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.activation_2 = nn.ReLU(True)
        self.layer_3 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.activation_3 = nn.ReLU(True)
        self.layer_4 = nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, latent_batch):
        out = self.activation_1(self.layer_1(latent_batch))
        out = self.activation_2(self.layer_2(out))
        out = self.activation_3(self.layer_3(out))
        out = self.layer_4(out)  # ignore the last activation
        return out


class ProbGen4mlpLin(nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_dim=2, rho_init=-5, rho_prior=-5):
        super(ProbGen4mlpLin, self).__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.rho_prior = rho_prior
        self.rho_init = rho_init

        # build the layers
        self.layer_1 = ProbLinLayer(in_features=latent_dim, out_features=hidden_dim, rho_prior=rho_prior, rho_init=rho_init)
        self.activation_1 = nn.ReLU(True)
        self.layer_2 = ProbLinLayer(in_features=hidden_dim, out_features=hidden_dim, rho_prior=rho_prior, rho_init=rho_init)
        self.activation_2 = nn.ReLU(True)
        self.layer_3 = ProbLinLayer(in_features=hidden_dim, out_features=hidden_dim, rho_prior=rho_prior, rho_init=rho_init)
        self.activation_3 = nn.ReLU(True)
        self.layer_4 = ProbLinLayer(in_features=hidden_dim, out_features=out_dim, rho_prior=rho_prior, rho_init=rho_init)

    def forward(self, latent_batch, sample=True):
        out = self.activation_1(self.layer_1(latent_batch, sample=sample))
        out = self.activation_2(self.layer_2(out, sample=sample))
        out = self.activation_3(self.layer_3(out, sample=sample))
        out = self.layer_4(out, sample=sample)

        if self.training:
            self._compute_kl()
        return out

    def _compute_kl(self):
        self.kl_div = self.layer_1.kl_div + self.layer_2.kl_div + self.layer_3.kl_div + self.layer_4.kl_div

    def initialize(self, generator: DetGen4mlpLin, rho_prior, rho_init):
        assert self.hidden_dim == generator.hidden_dim
        self.layer_1 = ProbLinLayer(in_features=self.latent_dim, out_features=self.hidden_dim, rho_prior=rho_prior, rho_init=rho_init,
                                    bias=True, dist='gaussian', init_type='layer', prior_params='layer',
                                    init_layer=generator.layer_1, prior_layer=generator.layer_1)
        self.layer_2 = ProbLinLayer(in_features=self.hidden_dim, out_features=self.hidden_dim, rho_prior=rho_prior, rho_init=rho_init,
                                    bias=True, dist='gaussian', init_type='layer', prior_params='layer',
                                    init_layer=generator.layer_2, prior_layer=generator.layer_2)
        self.layer_3 = ProbLinLayer(in_features=self.hidden_dim, out_features=self.hidden_dim, rho_prior=rho_prior, rho_init=rho_init,
                                    bias=True, dist='gaussian', init_type='layer', prior_params='layer',
                                    init_layer=generator.layer_3, prior_layer=generator.layer_3)
        self.layer_4 = ProbLinLayer(in_features=self.hidden_dim, out_features=self.out_dim, rho_prior=rho_prior, rho_init=rho_init,
                                    bias=True, dist='gaussian', init_type='layer', prior_params='layer',
                                    init_layer=generator.layer_4, prior_layer=generator.layer_4)


class Critic4mlpBjorckGS(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Critic4mlpBjorckGS, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # build the layers
        self.layer_1 = BjorckLinear(in_features=in_dim, out_features=hidden_dim, config=default_config())
        self.activation_1 = GroupSort(num_units=2)
        self.layer_2 = BjorckLinear(in_features=hidden_dim, out_features=hidden_dim, config=default_config())
        self.activation_2 = GroupSort(num_units=2)
        self.layer_3 = BjorckLinear(in_features=hidden_dim, out_features=hidden_dim, config=default_config())
        self.activation_3 = GroupSort(num_units=2)
        self.layer_4 = BjorckLinear(in_features=hidden_dim, out_features=1, config=default_config())

    def forward(self, batch):
        out = self.activation_1(self.layer_1(batch))
        out = self.activation_2(self.layer_2(out))
        out = self.activation_3(self.layer_3(out))
        out = self.layer_4(out)
        return out

    def initialize(self, critic):
        assert self.hidden_dim == critic.hidden_dim
        self.layer_1.weight.data = critic.layer_1.weight.data.clone()
        self.layer_1.bias.data = critic.layer_1.bias.data.clone()
        self.layer_2.weight.data = critic.layer_2.weight.data.clone()
        self.layer_2.bias.data = critic.layer_2.bias.data.clone()
        self.layer_3.weight.data = critic.layer_3.weight.data.clone()
        self.layer_3.bias.data = critic.layer_3.bias.data.clone()
        self.layer_4.weight.data = critic.layer_4.weight.data.clone()
        self.layer_4.bias.data = critic.layer_4.bias.data.clone()


class Critic6mlpBjorckGS(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Critic6mlpBjorckGS, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # build the layers
        self.layer_1 = BjorckLinear(in_features=in_dim, out_features=hidden_dim, config=default_config())
        self.activation_1 = GroupSort(num_units=2)
        self.layer_2 = BjorckLinear(in_features=hidden_dim, out_features=hidden_dim, config=default_config())
        self.activation_2 = GroupSort(num_units=2)
        self.layer_3 = BjorckLinear(in_features=hidden_dim, out_features=hidden_dim, config=default_config())
        self.activation_3 = GroupSort(num_units=2)
        self.layer_4 = BjorckLinear(in_features=hidden_dim, out_features=hidden_dim, config=default_config())
        self.activation_4 = GroupSort(num_units=2)
        self.layer_5 = BjorckLinear(in_features=hidden_dim, out_features=hidden_dim, config=default_config())
        self.activation_5 = GroupSort(num_units=2)
        self.layer_6 = BjorckLinear(in_features=hidden_dim, out_features=1, config=default_config())

    def forward(self, batch):
        out = self.activation_1(self.layer_1(batch))
        out = self.activation_2(self.layer_2(out))
        out = self.activation_3(self.layer_3(out))
        out = self.activation_4(self.layer_4(out))
        out = self.activation_5(self.layer_5(out))
        out = self.layer_6(out)
        return out

    def initialize(self, critic):
        assert self.hidden_dim == critic.hidden_dim
        self.layer_1.weight.data = critic.layer_1.weight.data.clone()
        self.layer_1.bias.data = critic.layer_1.bias.data.clone()
        self.layer_2.weight.data = critic.layer_2.weight.data.clone()
        self.layer_2.bias.data = critic.layer_2.bias.data.clone()
        self.layer_3.weight.data = critic.layer_3.weight.data.clone()
        self.layer_3.bias.data = critic.layer_3.bias.data.clone()
        self.layer_4.weight.data = critic.layer_4.weight.data.clone()
        self.layer_4.bias.data = critic.layer_4.bias.data.clone()
        self.layer_5.weight.data = critic.layer_5.weight.data.clone()
        self.layer_5.bias.data = critic.layer_5.bias.data.clone()
        self.layer_6.weight.data = critic.layer_6.weight.data.clone()
        self.layer_6.bias.data = critic.layer_6.bias.data.clone()




