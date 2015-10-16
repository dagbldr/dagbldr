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


def _epslog(x):
    return tensor.cast(tensor.log(tensor.clip(x, 1E-12, 1E12)),
                       theano.config.floatX)


def _log_add(a, b):
    max_ = tensor.maximum(a, b)
    return (max_ + tensor.log1p(tensor.exp(a + b - 2 * max_)))


def _log_dot_matrix(x, z):
    inf = 1E12
    log_dot = tensor.dot(x, z)
    zeros_to_minus_inf = (z.max(axis=0) - 1) * inf
    return log_dot + zeros_to_minus_inf


def _log_dot_tensor(x, z):
    inf = 1E12
    log_dot = (x.dimshuffle(1, 'x', 0) * z).sum(axis=0).T
    zeros_to_minus_inf = (z.max(axis=0) - 1) * inf
    return log_dot + zeros_to_minus_inf.T


def _recurrence_relation(y, y_mask):
    # with blank symbol of -1 this falls back to the recurrence that fails
    # with repeating symbols!
    blank_symbol = -1
    n_y = y.shape[0]
    blanks = tensor.zeros((2, y.shape[1])) + blank_symbol
    ybb = tensor.concatenate((y, blanks), axis=0).T
    sec_diag = (tensor.neq(ybb[:, :-2], ybb[:, 2:]) *
                tensor.eq(ybb[:, 1:-1], blank_symbol) *
                y_mask.T)

    # r1: LxL
    # r2: LxL
    # r3: LxLxB
    r2 = tensor.eye(n_y, k=1)
    r3 = (tensor.eye(n_y, k=2).dimshuffle(0, 1, 'x') *
          sec_diag.dimshuffle(1, 'x', 0))
    return r2, r3


def _class_batch_to_labeling_batch(y, y_hat, y_hat_mask):
    # Why dimshuffle...
    y_hat = y_hat.dimshuffle(0, 2, 1)
    y_hat = y_hat * y_hat_mask.dimshuffle(0, 'x', 1)
    batch_size = y_hat.shape[2]
    res = y_hat[:, y.astype('int32'), tensor.arange(batch_size)]
    return res


def _log_path_probs(y, y_mask, y_hat, y_hat_mask):
    pred_y = _class_batch_to_labeling_batch(y, y_hat, y_hat_mask)
    r2, r3 = _recurrence_relation(y, y_mask)

    def step(log_p_curr, log_p_prev):
        p1 = log_p_prev
        p2 = _log_dot_matrix(p1, r2)
        p3 = _log_dot_tensor(p1, r3)
        p123 = _log_add(p3, _log_add(p1, p2))

        return (log_p_curr.T +
                p123 +
                _epslog(y_mask.T))

    log_probabilities, _ = theano.scan(
        step,
        sequences=[_epslog(pred_y)],
        outputs_info=[_epslog(tensor.eye(y.shape[0])[0] *
                              tensor.ones(y.T.shape))])
    return log_probabilities


def _ctc_label_seq(y, y_mask):
    blank_symbol = -1
    # for y
    y_extended = y.T.dimshuffle(0, 1, 'x')
    blanks = tensor.zeros_like(y_extended) + blank_symbol
    concat = tensor.concatenate([y_extended, blanks], axis=2)
    res = concat.reshape((concat.shape[0],
                          concat.shape[1] * concat.shape[2])).T
    beginning_blanks = tensor.zeros((1, res.shape[1])) + blank_symbol
    blanked_y = tensor.concatenate([beginning_blanks, res], axis=0)

    y_mask_extended = y_mask.T.dimshuffle(0, 1, 'x')
    concat = tensor.concatenate([y_mask_extended,
                                 y_mask_extended], axis=2)
    res = concat.reshape((concat.shape[0],
                          concat.shape[1] * concat.shape[2])).T
    beginning_blanks = tensor.ones((1, res.shape[1]),
                                   dtype=theano.config.floatX)
    blanked_y_mask = tensor.concatenate([beginning_blanks, res], axis=0)
    return blanked_y, blanked_y_mask


def log_ctc_cost(true_labels, true_mask, predicted_labels, predicted_mask):
    """
    Log CTC cost between a minibatch of true labels and softmax predicted labels

    Parameters
    ----------
    true_labels : imatrix
       True labels as index values e.g. [[1, 2], [4, 7]]

    true_mask : imatrix, shape true_labels.shape
       Mask over labels

    predicted_labels : tensor
        3D probability tensor such as the softmax output of an RNN layer

    predicted_mask : matrix, shape (predicted_labels.shape[0],
                                    predicted_labels.shape[1])
        Mask over predictions

    Returns
    -------
    negative_log_label_prob : tensor, shape predicted_labels.shape
        negative log label probabilities per sequence

    References
    ----------
    [Graves2008] `Supervised Sequence Labelling With Recurrent Neural Networks`

    Notes
    -----
    Code interpreted from Shawn Tan, Rakesh Var, and Mohammad Pezeshki

    https://github.com/shawntan/theano-ctc/
    https://github.com/rakeshvar/rnn_ctc
    https://github.com/mohammadpz/CTC-Connectionist-Temporal-Classification

    """
    # TODO: Cleanup logic and avoid transpose
    y = true_labels.T
    y_mask = true_mask.T
    y, y_mask = _ctc_label_seq(y, y_mask)
    y_hat = predicted_labels
    y_hat_mask = predicted_mask
    y_hat_mask_len = tensor.sum(y_hat_mask, axis=0, dtype='int32')
    y_mask_len = tensor.sum(y_mask, axis=0, dtype='int32')
    log_probs = _log_path_probs(y, y_mask, y_hat, y_hat_mask)
    batch_size = log_probs.shape[1]
    labels_prob = _log_add(
        log_probs[y_hat_mask_len - 1, tensor.arange(batch_size),
                  y_mask_len - 1],
        log_probs[y_hat_mask_len - 1, tensor.arange(batch_size),
                  y_mask_len - 2])
    return -labels_prob
