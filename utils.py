import torch
import numpy as np
import json
from omegaconf import OmegaConf


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def default_config():
    with open('main_conf.json', "r", encoding="utf-8") as f:
        json_data = json.load(f)
    main_conf = OmegaConf.create(json_data)
    main_conf.cuda = torch.cuda.is_available()
    return main_conf


class Average:
    def __init__(self):
        self.entries = []

    def add(self, entry):
        # number is either a float, or a 1x1 Tensor
        assert type(entry) in [float, torch.Tensor]

        number = float(entry.detach()) if torch.is_tensor(entry) else entry

        self.entries.append(number)

    def compute(self):
        return np.mean(self.entries)


