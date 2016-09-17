# Author: Kyle Kastner
# License: BSD 3-clause
import numpy as np
from scipy import linalg
from scipy.misc import factorial
import theano
from theano import tensor
from theano.tensor.signal.downsample import max_pool_2d
from theano.sandbox.rng_mrg import MRG_RandomStreams
from ..utils import concatenate, as_shared
from ..core import get_name, set_shared, get_shared
from ..core import get_logger, get_type

logger = get_logger()
_type = get_type()


'''
def embedding(list_of_index_inputs, max_index, proj_dim, graph, name,
              random_state=None, strict=True, init_func=np_uniform):
    check_type = any([index_input.dtype != "int32"
                      for index_input in list_of_index_inputs])
    check_dim = any([index_input.ndim != 1
                     for index_input in list_of_index_inputs])
    if check_type or check_dim:
        raise ValueError("index_input must be an ivector!")
    embedding_W_name = name + "_embedding_W"
    list_of_names = [embedding_W_name]
    if not names_in_graph(list_of_names, graph):
        assert random_state is not None
        np_embedding_W = init_func((max_index, proj_dim), random_state)
        add_arrays_to_graph([np_embedding_W], list_of_names, graph,
                            strict=strict)
    else:
        if strict:
            raise AttributeError(
                "Name %s already found in graph with strict mode!" % name)
    embedding_W, = fetch_from_graph(list_of_names, graph)
    embeddings = [embedding_W[index_input]
                  for index_input in list_of_index_inputs]
    # could sum instead?
    output = concatenate(embeddings, graph, name, axis=embedding_W.ndim - 1)
    n_lists = len(list_of_index_inputs)
    return output.reshape((-1, n_lists, proj_dim))


def log_gaussian_mixture(list_of_inputs, graph, name, proj_dim=None,
                         batch_normalize=False, mode_switch=None,
                         random_state=None, n_components=5, strict=True,
                         init_func=np_tanh_fan_uniform):
    assert n_components >= 1
    assert proj_dim is not None

    def _reshape(l):
        if l.ndim == 2:
            dim0, dim1 = l.shape
            t = l.reshape((dim0, proj_dim, n_components))
        elif l.ndim == 3:
            dim0, dim1, dim2 = l.shape
            t = l.reshape((dim0, dim1, proj_dim, n_components))
        else:
            raise ValueError("input ndim not supported for gaussian "
                             "mixture layer")
        return t
    mus = projection(
        list_of_inputs=list_of_inputs, graph=graph,
        name=name + "_mus", proj_dim=n_components * proj_dim,
        batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=linear_activation)
    log_sigmas = projection(
        list_of_inputs=list_of_inputs, graph=graph,
        name=name + "_log_sigmas", proj_dim=n_components * proj_dim,
        batch_normalize=batch_normalize,
        mode_switch=mode_switch,
        random_state=random_state, strict=strict,
        init_func=init_func, act_func=linear_activation)
    coeffs = softmax(
        list_of_inputs=list_of_inputs, graph=graph, name=name + "_coeffs",
        proj_dim=n_components, random_state=random_state, strict=strict,
        init_func=init_func)
    mus = _reshape(mus)
    log_sigmas = _reshape(log_sigmas)
    return coeffs, mus, log_sigmas


def bernoulli_and_correlated_log_gaussian_mixture(
    list_of_inputs, graph, name, proj_dim=2, batch_normalize=False,
    mode_switch=None, random_state=None, n_components=5, strict=True,
    init_func=np_tanh_fan_uniform):
    assert n_components >= 1
    assert proj_dim is not None

    def _reshape(l, d=n_components):
        if d == 1:
            t = l.dimshuffle(
                *(list(range(corr.ndim - 1)) + ['x'] + [corr.ndim - 1]))
            return t
        if l.ndim == 2:
            dim0, dim1 = l.shape
            t = l.reshape((dim0, proj_dim, d))
        elif l.ndim == 3:
            dim0, dim1, dim2 = l.shape
            t = l.reshape((dim0, dim1, proj_dim, d))
        else:
            raise ValueError("input ndim not supported for gaussian "
                             "mixture layer")
        return t
    assert proj_dim == 2
    mus = projection(
        list_of_inputs=list_of_inputs, graph=graph,
        name=name + "_mus", proj_dim=n_components * proj_dim,
        batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=linear_activation)
    log_sigmas = projection(
        list_of_inputs=list_of_inputs, graph=graph,
        name=name + "_log_sigmas", proj_dim=n_components * proj_dim,
        batch_normalize=batch_normalize,
        mode_switch=mode_switch,
        random_state=random_state, strict=strict,
        init_func=init_func, act_func=linear_activation)
    coeffs = softmax(
        list_of_inputs=list_of_inputs, graph=graph, name=name + "_coeffs",
        proj_dim=n_components, random_state=random_state, strict=strict,
        init_func=init_func)
    calc_corr = factorial(proj_dim ** 2 // 2 - 1)
    corr = projection(
        list_of_inputs=list_of_inputs, graph=graph,
        name=name + "_corr", proj_dim=n_components * calc_corr,
        batch_normalize=batch_normalize,
        mode_switch=mode_switch,
        random_state=random_state, strict=strict,
        init_func=init_func, act_func=tensor.tanh)
    binary = projection(
        list_of_inputs=list_of_inputs, graph=graph,
        name=name + "_binary", proj_dim=1,
        batch_normalize=batch_normalize,
        mode_switch=mode_switch,
        random_state=random_state, strict=strict,
        init_func=init_func, act_func=tensor.nnet.sigmoid)
    mus = _reshape(mus)
    log_sigmas = _reshape(log_sigmas)
    corr = _reshape(corr, calc_corr)
    return binary, coeffs, mus, log_sigmas, corr


def bernoulli_and_correlated_gaussian_mixture(
    list_of_inputs, graph, name, proj_dim=2, batch_normalize=False,
    mode_switch=None, random_state=None, n_components=5, strict=True,
    init_func=np_tanh_fan_uniform):
    assert n_components >= 1
    assert proj_dim is not None

    def _reshape(l, d=n_components):
        if d == 1:
            t = l.dimshuffle(
                *(list(range(corr.ndim - 1)) + ['x'] + [corr.ndim - 1]))
            return t
        if l.ndim == 2:
            dim0, dim1 = l.shape
            t = l.reshape((dim0, proj_dim, d))
        elif l.ndim == 3:
            dim0, dim1, dim2 = l.shape
            t = l.reshape((dim0, dim1, proj_dim, d))
        else:
            raise ValueError("input ndim not supported for gaussian "
                             "mixture layer")
        return t
    assert proj_dim == 2
    mus = projection(
        list_of_inputs=list_of_inputs, graph=graph,
        name=name + "_mus", proj_dim=n_components * proj_dim,
        batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=linear_activation)
    sigmas = projection(
        list_of_inputs=list_of_inputs, graph=graph,
        name=name + "_sigmas", proj_dim=n_components * proj_dim,
        batch_normalize=batch_normalize,
        mode_switch=mode_switch,
        random_state=random_state, strict=strict,
        init_func=init_func, act_func=softplus)
    coeffs = softmax(
        list_of_inputs=list_of_inputs, graph=graph, name=name + "_coeffs",
        proj_dim=n_components, random_state=random_state, strict=strict,
        init_func=init_func)
    calc_corr = factorial(proj_dim ** 2 // 2 - 1)
    corr = projection(
        list_of_inputs=list_of_inputs, graph=graph,
        name=name + "_corr", proj_dim=n_components * calc_corr,
        batch_normalize=batch_normalize,
        mode_switch=mode_switch,
        random_state=random_state, strict=strict,
        init_func=init_func, act_func=tensor.tanh)
    binary = projection(
        list_of_inputs=list_of_inputs, graph=graph,
        name=name + "_binary", proj_dim=1,
        batch_normalize=batch_normalize,
        mode_switch=mode_switch,
        random_state=random_state, strict=strict,
        init_func=init_func, act_func=tensor.nnet.sigmoid)
    mus = _reshape(mus)
    sigmas = _reshape(sigmas)
    corr = _reshape(corr, calc_corr)
    return binary, coeffs, mus, sigmas, corr


def softmax_sample(list_of_multinomial_inputs, graph, name,
                         random_state=None):
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        print("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    theano_rng = MRG_RandomStreams(seed=theano_seed)
    conc_multinomial = concatenate(list_of_multinomial_inputs, graph,
                                   name,
                                   axis=list_of_multinomial_inputs[0].ndim - 1)
    conc_multinomial /= len(list_of_multinomial_inputs)
    samp = theano_rng.multinomial(pvals=conc_multinomial,
                                  dtype="int32")
    # We know shape of conc_multinomial == shape of random sample
    shape = calc_expected_dims(graph, conc_multinomial)
    list_of_random = [samp, ]
    list_of_names = [name + "_random", ]
    list_of_shapes = [shape, ]
    add_random_to_graph(list_of_random, list_of_shapes, list_of_names, graph)
    return samp


def gaussian_sample(list_of_mu_inputs, list_of_sigma_inputs,
                          graph, name, random_state=None):
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        print("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    theano_rng = MRG_RandomStreams(seed=theano_seed)
    conc_mu = concatenate(list_of_mu_inputs, graph, name,
                          axis=list_of_mu_inputs[0].ndim - 1)
    conc_sigma = concatenate(list_of_sigma_inputs, graph, name,
                             axis=list_of_sigma_inputs[0].ndim - 1)
    e = theano_rng.normal(size=(conc_sigma.shape[0],
                                conc_sigma.shape[1]),
                          dtype=conc_sigma.dtype)
    # We know shape of mu == shape of sigma == shape of random sample
    shape = calc_expected_dims(graph, conc_mu)
    list_of_random = [e, ]
    list_of_names = [name + "_random", ]
    list_of_shapes = [shape, ]
    add_random_to_graph(list_of_random, list_of_shapes, list_of_names, graph)
    samp = conc_mu + conc_sigma * e
    return samp


def gaussian_log_sample(list_of_mu_inputs, list_of_log_sigma_inputs,
                              graph, name, random_state=None):
    """ log_sigma_inputs should be from a linear """
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        print("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    theano_rng = MRG_RandomStreams(seed=theano_seed)
    conc_mu = concatenate(list_of_mu_inputs, graph, name,
                          axis=list_of_mu_inputs[0].ndim - 1)
    conc_log_sigma = concatenate(list_of_log_sigma_inputs, graph, name,
                                 axis=list_of_log_sigma_inputs[0].ndim - 1)
    e = theano_rng.normal(size=(conc_log_sigma.shape[0],
                                conc_log_sigma.shape[1]),
                          dtype=conc_log_sigma.dtype)
    # We know shape of mu == shape of log sigma == shape of random sample
    shape = calc_expected_dims(graph, conc_mu)
    list_of_random = [e, ]
    list_of_names = [name + "_random", ]
    list_of_shapes = [shape, ]
    add_random_to_graph(list_of_random, list_of_shapes, list_of_names, graph)
    samp = conc_mu + tensor.exp(0.5 * conc_log_sigma) * e
    return samp
'''
