# Author: Kyle Kastner
# License: BSD 3-clause
import numpy as np
from theano import tensor
from ..utils import concatenate


def binary_crossentropy_nll(predicted_values, true_values):
    """ Returns likelihood compared to binary true_values """
    return (-true_values * tensor.log(predicted_values) - (
        1 - true_values) * tensor.log(1 - predicted_values)).sum(axis=-1)


def binary_entropy(values):
    return (-values * tensor.log(values)).sum(axis=-1)


def categorical_crossentropy_nll(predicted_values, true_values):
    """ Returns likelihood compared to one hot category labels """
    indices = tensor.argmax(true_values, axis=-1)
    rows = tensor.arange(true_values.shape[0])
    if predicted_values.ndim < 3:
        return -tensor.log(predicted_values)[rows, indices]
    elif predicted_values.ndim == 3:
        d0 = true_values.shape[0]
        d1 = true_values.shape[1]
        pred = predicted_values.reshape((d0 * d1, -1))
        ind = indices.reshape((d0 * d1,))
        s = tensor.arange(pred.shape[0])
        correct = -tensor.log(pred)[s, ind]
        return correct.reshape((d0, d1,))
    else:
        raise AttributeError("Tensor dim not supported")


def abs_error_nll(predicted_values, true_values):
    return tensor.abs_(predicted_values - true_values).sum(axis=-1)


def squared_error_nll(predicted_values, true_values):
    return tensor.sqr(predicted_values - true_values).sum(axis=-1)


def gaussian_error_nll(mu_values, sigma_values, true_values):
    """ sigma should come from a softplus layer """
    nll = 0.5 * (mu_values - true_values) ** 2 / sigma_values ** 2 + tensor.log(
        2 * np.pi * sigma_values ** 2)
    return nll


def log_gaussian_error_nll(mu_values, log_sigma_values, true_values):
    """ log_sigma should come from a linear layer """
    nll = 0.5 * (mu_values - true_values) ** 2 / tensor.exp(
        log_sigma_values) ** 2 + tensor.log(2 * np.pi) + 2 * log_sigma_values
    return nll


def masked_cost(cost, mask):
    return cost * mask


def gaussian_kl(list_of_mu_inputs, list_of_sigma_inputs, name):
    conc_mu = concatenate(list_of_mu_inputs, name)
    conc_sigma = concatenate(list_of_sigma_inputs, name)
    kl = 0.5 * tensor.sum(-2 * tensor.log(conc_sigma) + conc_mu ** 2
                          + conc_sigma ** 2 - 1, axis=1)
    return kl


def gaussian_log_kl(list_of_mu_inputs, list_of_log_sigma_inputs, name):
    """ log_sigma_inputs should come from linear layer"""
    conc_mu = concatenate(list_of_mu_inputs, name)
    conc_log_sigma = 0.5 * concatenate(list_of_log_sigma_inputs, name)
    kl = 0.5 * tensor.sum(-2 * conc_log_sigma + conc_mu ** 2
                          + tensor.exp(conc_log_sigma) ** 2 - 1, axis=1)
    return kl
