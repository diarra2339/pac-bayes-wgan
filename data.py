import torch
import torch.distributions as D
from torch.utils.data import Dataset, Subset, TensorDataset, DataLoader

import numpy as np


class GMM:
    def __init__(self, coeffs, means, variances):
        # means and variances have the same form: list of lists of floats or list of Tensors
        if type(means) is list and type(means[0]) is list:
            means = [torch.Tensor(x) for x in means]
            variances = [torch.Tensor(x) for x in variances]
            means = torch.stack(means)
            variances = torch.stack(variances)
            self.dimension = len(means[0])
        elif type(means) is list and type(means[0]) is torch.Tensor:
            means = torch.stack(means)
            variances = torch.stack(variances)
        elif type(means) is torch.Tensor:
            pass  # use means and variances as is
        else:
            raise ValueError('The variables means and variances do not have the right form :-(')

        mix = D.Categorical(torch.Tensor(coeffs))
        comp = D.Independent(D.Normal(means, torch.sqrt(variances)), 1)
        self.means = means
        self.variances = variances
        self.gmm = D.MixtureSameFamily(mix, comp)

    def sample(self, num_samples):
        return self.gmm.sample([num_samples])


class GridGMMTrunc:
    def __init__(self, comps=5, side=8, max_side=8.2, sigma=0.05, train_size=100000, val_size=50000, test_size=50000, batch_size=64, workers=7):
        self.comps = comps  # the total number of components is comps * comps
        self.side = side  # the side of the square containing the centers of the components
        self.max_side = max_side
        self.sigma = sigma

        margin = (self.max_side - self.side) / 2
        self.min_, self.max_ = -self.side / 2 - margin, self.side / 2 + margin

        self.set_loaders(train_size=train_size, val_size=val_size, test_size=test_size, batch_size=batch_size, workers=workers)

    def truncate(self, data):
        # data = data[(self.min_ <= data[:, 0]) * (data[:, 0] <= self.max_) * (self.min_ <= data[:, 1]) * (data[:, 0] <= self.max_)]
        data = data[torch.vstack([(self.min_ <= data[:, 0]), (data[:, 0] <= self.max_),
                                  (self.min_ <= data[:, 1]), (data[:, 1] <= self.max_)]).all(axis=0)]
        return data

    def set_loaders(self, train_size, val_size, test_size, batch_size, workers):
        centers_x = [-self.side / 2 + i * self.side / (self.comps - 1) for i in range(self.comps)]
        centers_y = [-self.side / 2 + j * self.side / (self.comps - 1) for j in range(self.comps)]

        means = [[centers_x[i], centers_y[j]] for i in range(self.comps) for j in range(self.comps)]
        variances = [[self.sigma**2, self.sigma**2] for _ in range(self.comps**2)]
        coeffs = [1 for _ in range(self.comps**2)]
        gmm = GMM(coeffs=coeffs, means=means, variances=variances)
        data = gmm.sample(train_size + val_size + test_size + int(0.5 * train_size))
        data = self.truncate(data)
        assert data.shape[0] >= train_size + val_size + test_size

        self.train_data = TensorDataset(data[:train_size], torch.zeros(train_size, 1))
        self.val_data = TensorDataset(data[train_size: train_size + val_size], torch.zeros(val_size, 1))
        self.test_data = TensorDataset(data[train_size + val_size: train_size + val_size + test_size], torch.zeros(test_size, 1))
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=True, num_workers=workers)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=True, num_workers=workers)


class RingGMMTrunc:
    def __init__(self, n_comps=8, radius=3, max_radius=3.2, sigma=0.05, train_size=60000, val_size=30000, test_size=30000, batch_size=256, workers=5):
        self.max_radius = max_radius if max_radius is not None else radius + 1
        self.n_comps = n_comps
        self.radius = radius
        self.max_radius = max_radius
        self.sigma = sigma

        self.set_loaders(train_size=train_size, val_size=val_size, test_size=test_size, batch_size=batch_size, workers=workers)

    def set_loaders(self, train_size, val_size, test_size, batch_size, workers):
        centers_x = [self.radius * np.cos((2 * np.pi * i) / self.n_comps) for i in range(self.n_comps)]
        centers_y = [self.radius * np.sin((2 * np.pi * i) / self.n_comps) for i in range(self.n_comps)]

        means = [[centers_x[i], centers_y[i]] for i in range(self.n_comps)]
        variances = [[self.sigma**2, self.sigma**2] for _ in range(self.n_comps)]
        coeffs = [1 for _ in range(self.n_comps)]
        gmm = GMM(coeffs=coeffs, means=means, variances=variances)
        data = gmm.sample(train_size + val_size + test_size + int(0.5*train_size))
        data = self.truncate(data)
        assert data.shape[0] >= train_size + val_size + test_size

        self.train_data = TensorDataset(data[:train_size], torch.zeros(train_size, 1))
        self.val_data = TensorDataset(data[train_size: train_size + val_size], torch.zeros(val_size))
        self.test_data = TensorDataset(data[train_size + val_size: train_size + val_size + test_size], torch.zeros(test_size))
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=True, num_workers=workers)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=True, num_workers=workers)

    def truncate(self, data):
        data = data[data[:, 0]**2 + data[:, 1]**2 <= self.max_radius**2]
        return data


