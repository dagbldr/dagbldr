# Author: Kyle Kastner
# License: BSD 3-clause
import numpy as np
from theano import tensor
import theano
from ..utils import concatenate


def binary_crossentropy(predicted_values, true_values):
    """
    Bernoulli negative log likelihood of predicted compared to binary
    true_values

    Parameters
    ----------
    predicted_values : tensor, shape 2D or 3D
        The predicted values out of some layer, normally a sigmoid_layer

    true_values : tensor, shape 2D or 3D
        The ground truth values. Mush have same shape as predicted_values

    Returns
    -------
    binary_crossentropy : tensor, shape predicted_values.shape[1:]
        The cost per sample, or per sample per step if 3D

    """
    return (-true_values * tensor.log(predicted_values) - (
        1 - true_values) * tensor.log(1 - predicted_values)).sum(axis=-1)


def binary_entropy(values):
    """
    Bernoulli entropy of values

    Parameters
    ----------
    values : tensor, shape 2D or 3D
        The values to calculate entropy over


    Returns
    -------
    binary_entropy : tensor, shape predicted_values.shape[1:]
        The cost per sample, or per sample per step if 3D

    """
    return (-values * tensor.log(values)).sum(axis=-1)


def categorical_crossentropy(predicted_values, true_values, eps=0.):
    """
    Multinomial negative log likelihood of predicted compared to one hot
    true_values

    Parameters
    ----------
    predicted_values : tensor, shape 2D or 3D
        The predicted class probabilities out of some layer,
        normally the output of softmax_layer

    true_values : tensor, shape 2D or 3D
        One hot ground truth values. Must be the same shape as
        predicted_values. One hot representations can be achieved using
        dagbldr.utils.convert_to_one_hot

    eps : float, default 0
        Epsilon to be added during log calculation to avoid NaN values.

    Returns
    -------
    categorical_crossentropy : tensor, shape predicted_values.shape[1:]
        The cost per sample, or per sample per step if 3D

    """
    indices = tensor.argmax(true_values, axis=-1)
    rows = tensor.arange(true_values.shape[0])
    if eps > 0:
        p = tensor.cast(predicted_values, theano.config.floatX) + eps
        p /= tensor.sum(p, axis=predicted_values.ndim - 1, keepdims=True)
    else:
        p = tensor.cast(predicted_values, theano.config.floatX)
    if predicted_values.ndim < 3:
        return -tensor.log(p)[rows, indices]
    elif predicted_values.ndim == 3:
        d0 = true_values.shape[0]
        d1 = true_values.shape[1]
        pred = p.reshape((d0 * d1, -1))
        ind = indices.reshape((d0 * d1,))
        s = tensor.arange(pred.shape[0])
        correct = -tensor.log(pred)[s, ind]
        return correct.reshape((d0, d1,))
    else:
        raise AttributeError("Tensor dim not supported")


def abs_error(predicted_values, true_values):
    """
    Gaussian negative log likelihood compared to true_values. Estimates the
    conditional median.

    Parameters
    ----------
    predicted_values : tensor, shape 2D or 3D
        The predicted values out of some layer.

    true_values : tensor, shape 2D or 3D
        Ground truth values. Must be the same shape as
        predicted_values.

    Returns
    -------
    abs_error : tensor, shape predicted_values.shape[1:]
        The cost per sample, or per sample per step if 3D

    """
    return tensor.abs_(predicted_values - true_values).sum(axis=-1)


def squared_error(predicted_values, true_values):
    """
    Gaussian negative log likelihood compared to true_values. Estimates the
    conditional mean.

    Parameters
    ----------
    predicted_values : tensor, shape 2D or 3D
        The predicted values out of some layer.

    true_values : tensor, shape 2D or 3D
        Ground truth values. Must be the same shape as
        predicted_values.

    Returns
    -------
    squared_error : tensor, shape predicted_values.shape[1:]
        The cost per sample, or per sample per step if 3D

    """

    return tensor.sqr(predicted_values - true_values).sum(axis=-1)


def gaussian_error(mu_values, sigma_values, true_values):
    """
    Gaussian negative log likelihood compared to true_values.

    Parameters
    ----------
    mu_values : tensor, shape 2D or 3D
        The predicted values out of some layer, normally a linear layer

    sigma_values : tensor, shape 2D or 3D
        The predicted values out of some layer, normally a softplus layer

    true_values : tensor, shape 2D or 3D
        Ground truth values. Must be the same shape as
        predicted_values.

    Returns
    -------
    nll : tensor, shape predicted_values.shape[1:]
        The cost per sample, or per sample per step if 3D

    """
    nll = 0.5 * (mu_values - true_values) ** 2 / sigma_values ** 2 + tensor.log(
        2 * np.pi * sigma_values ** 2)
    return nll


def log_gaussian_error(mu_values, log_sigma_values, true_values):
    """
    Gaussian negative log likelihood compared to true_values.

    Parameters
    ----------
    mu_values : tensor, shape 2D or 3D
        The predicted values out of some layer, normally a linear layer

    log_sigma_values : tensor, shape 2D or 3D
        The predicted values out of some layer, normally a linear layer

    true_values : tensor, shape 2D or 3D
        Ground truth values. Must be the same shape as
        predicted_values.

    Returns
    -------
    nll : tensor, shape predicted_values.shape[1:]
        The cost per sample, or per sample per step if 3D

    """
    nll = 0.5 * (mu_values - true_values) ** 2 / tensor.exp(
        log_sigma_values) ** 2 + tensor.log(2 * np.pi) + 2 * log_sigma_values
    return nll


def masked_cost(cost, mask):
    """
    Mask the values from a given cost.

    Parameters
    ----------
    cost : tensor, shape 2D or 3D
        The original cost out of some cost function

    mask : tensor, shape 2D or 3D
        The mask to use to cancel part of the cost

    Returns
    -------
    masked_cost : tensor, shape cost.shape
        The masked cost

    """
    return cost * mask


def gaussian_kl(list_of_mu_inputs, list_of_sigma_inputs, graph, name):
    """
    Kullback-Liebler divergence between a single multi-dimensional
    gaussian and an N(0, 1) prior.

    Parameters
    ----------
    list_of_mu_inputs : list of tensors, each shape 2D or 3D
        The inputs to treat as mu values, normally coming from
        linear_layer

    list_of_sigma_inputs : list of tensors, each shape 2D or 3D
        The inputs to treat as sigma values, normally coming from
        softplus_layer

    Returns
    -------
    kl : tensor, shape .shape
        Kullback-Liebler divergence

    """
    conc_mu = concatenate(list_of_mu_inputs, graph, name)
    conc_sigma = concatenate(list_of_sigma_inputs, graph, name)
    kl = 0.5 * tensor.sum(-2 * tensor.log(conc_sigma) + conc_mu ** 2
                          + conc_sigma ** 2 - 1, axis=-1)
    return kl


def gaussian_log_kl(list_of_mu_inputs, list_of_log_sigma_inputs, graph, name):
    """
    Kullback-Liebler divergence between a single multi-dimensional
    gaussian and an N(0, 1) prior.

    Parameters
    ----------
    list_of_mu_inputs : list of tensors, each shape 2D or 3D
        The inputs to treat as mu values, normally coming from
        linear_layer

    list_of_log_sigma_inputs : list of tensors, each shape 2D or 3D
        The inputs to treat as sigma values, normally coming from
        linear_layer

    Returns
    -------
    kl : tensor, shape .shape
        Kullback-Liebler divergence

    """
    conc_mu = concatenate(list_of_mu_inputs, graph, name)
    conc_log_sigma = 0.5 * concatenate(list_of_log_sigma_inputs, graph, name)
    kl = 0.5 * tensor.sum(-2 * conc_log_sigma + conc_mu ** 2
                          + tensor.exp(2 * conc_log_sigma) - 1, axis=-1)
    return kl
