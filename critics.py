import torch
import torch.nn as nn


from utils import default_config
from lnets.models.layers.dense.bjorck_linear import BjorckLinear
from lnets.models.activations.group_sort import GroupSort


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