import numpy as np
import torch
from tqdm import tqdm

from utils import get_device


def get_real_samples(num_samples, data_loader):
    batch_size = data_loader.batch_size
    tensors, iters = [], 0
    for x, _ in data_loader:
        tensors.append(x)
        iters += 1
        if iters == num_samples // batch_size + 1:
            break
    real_samples = torch.cat(tensors, dim=0)[:num_samples]
    return real_samples


def get_fake_samples(model, num_samples, sample):
    fake_samples = model.generate_samples(num_samples=num_samples, sample=sample)
    return fake_samples


def get_critic_mean(model, samples):
    samples = samples.to(get_device())
    with torch.no_grad():
        logits = model.critic(samples)
    mean = logits.mean()
    return mean


def compute_bound(emp_risk, kl_div, kl_coeff, diameter, n, delta):
    lamda = 1 / kl_coeff
    diam_term = lamda * diameter ** 2 / (4 * n)
    kl_term = kl_div / lamda
    delta_term = np.log(1 / delta) / lamda
    bound = emp_risk + diam_term + kl_term + delta_term
    return bound


def compute_risks(model, train_loader, test_loader, num_train_samples, num_test_samples, sample=False):
    train_samples = get_real_samples(num_samples=num_train_samples, data_loader=train_loader)
    test_samples = get_real_samples(num_samples=num_test_samples, data_loader=test_loader)
    fake_samples_train = get_fake_samples(model=model, num_samples=num_train_samples, sample=sample)
    fake_samples_test = get_fake_samples(model=model, num_samples=num_test_samples, sample=sample)

    train_mean = get_critic_mean(model=model, samples=train_samples)
    test_mean = get_critic_mean(model=model, samples=test_samples)
    fake_mean_train = get_critic_mean(model=model, samples=fake_samples_train)
    fake_mean_test = get_critic_mean(model=model, samples=fake_samples_test)

    emp_risk = -(fake_mean_train - train_mean)  # get the negative critic loss
    test_risk = -(fake_mean_test - test_mean)  # get the negative critic loss

    return emp_risk, test_risk


def compare_bound(model, train_loader, test_loader, num_test_samples, kl_coeff, diameter, n, delta, sample, num_generators=1):
    sample = True if num_generators > 1 else sample
    assert num_generators > 0

    emp_list, test_list, bound_list = [], [], []
    for _ in tqdm(range(num_generators)):
        train_samples = get_real_samples(num_samples=n, data_loader=train_loader)
        test_samples = get_real_samples(num_samples=num_test_samples, data_loader=test_loader)
        fake_samples_train = get_fake_samples(model=model, num_samples=n, sample=sample)
        fake_samples_test = get_fake_samples(model=model, num_samples=num_test_samples, sample=sample)

        train_mean = get_critic_mean(model=model, samples=train_samples)
        test_mean = get_critic_mean(model=model, samples=test_samples)
        fake_mean_train = get_critic_mean(model=model, samples=fake_samples_train)
        fake_mean_test = get_critic_mean(model=model, samples=fake_samples_test)

        emp_risk = -(fake_mean_train - train_mean)  # get the negative critic loss
        test_risk = -(fake_mean_test - test_mean)  # get the negative critic loss

        bound = compute_bound(emp_risk=emp_risk, kl_div=model.generator.kl_div,
                              kl_coeff=kl_coeff, diameter=diameter, n=n, delta=delta)
        emp_list.append(emp_risk.cpu().detach())
        test_list.append(test_risk.cpu().detach())
        bound_list.append(bound.cpu().detach())
    emp_risk, test_risk, bound = np.mean(emp_list), np.mean(test_list), np.mean(bound_list)

    return emp_risk, test_risk, bound

