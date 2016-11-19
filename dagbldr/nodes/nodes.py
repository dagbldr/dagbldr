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


def np_zeros(shape):
    """
    Builds a numpy variable filled with zeros

    Parameters
    ----------
    shape, tuple of ints
        shape of zeros to initialize

    Returns
    -------
    initialized_zeros, array-like
        Array-like of zeros the same size as shape parameter
    """
    return np.zeros(shape).astype(_type)


def np_ones(shape):
    """
    Builds a numpy variable filled with ones

    Parameters
    ----------
    shape, tuple of ints
        shape of ones to initialize

    Returns
    -------
    initialized_ones, array-like
        Array-like of ones the same size as shape parameter
    """
    return np.ones(shape).astype(_type)


def np_unit_uniform(shape, random_state):
    return np_uniform(shape, random_state, scale=1.)


def np_uniform(shape, random_state, scale=0.08):
    """
    Builds a numpy variable filled with uniform random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 0.08)
        scale to apply to uniform random values from (-1, 1)
        default of 0.08 results in uniform random values in (-0.08, 0.08)

    Returns
    -------
    initialized_uniform, array-like
        Array-like of uniform random values the same size as shape parameter
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        shp = shape
    # Make sure bounds aren't the same
    return random_state.uniform(low=-scale, high=scale, size=shp).astype(_type)


def np_normal(shape, random_state, scale=0.01):
    """
    Builds a numpy variable filled with normal random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 0.01)
        default of 0.01 results in normal random values with variance 0.01

    Returns
    -------
    initialized_normal, array-like
        Array-like of normal random values the same size as shape parameter
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        shp = shape
    return (scale * random_state.randn(*shp)).astype(_type)


def np_tanh_fan_uniform(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 1.)
        default of 1. results in normal uniform random values
        with sqrt(6 / (fan in + fan out)) scale

    Returns
    -------
    initialized_fan, array-like
        Array-like of random values the same size as shape parameter

    References
    ----------
    Understanding the difficulty of training deep feedforward neural networks
        X. Glorot, Y. Bengio
    """
    if type(shape[0]) is tuple:
        kern_sum = np.prod(shape[0]) + np.prod(shape[1])
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        kern_sum = np.sum(shape)
        shp = shape
    # The . after the 6 is critical! shape has dtype int...
    bound = scale * np.sqrt(6. / kern_sum)
    return random_state.uniform(low=-bound, high=bound, size=shp).astype(_type)


def np_tanh_fan_normal(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 1.)
        default of 1. results in normal random values
        with sqrt(2 / (fan in + fan out)) scale

    Returns
    -------
    initialized_fan, array-like
        Array-like of random values the same size as shape parameter

    References
    ----------
    Understanding the difficulty of training deep feedforward neural networks
        X. Glorot, Y. Bengio
    """
    # The . after the 2 is critical! shape has dtype int...
    if type(shape[0]) is tuple:
        kern_sum = np.prod(shape[0]) + np.prod(shape[1])
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        kern_sum = np.sum(shape)
        shp = shape
    var = scale * np.sqrt(2. / kern_sum)
    return var * random_state.randn(*shp).astype(_type)


def np_sigmoid_fan_uniform(shape, random_state, scale=4.):
    """
    Builds a numpy variable filled with random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 4.)
        default of 4. results in uniform random values
        with 4 * sqrt(6 / (fan in + fan out)) scale

    Returns
    -------
    initialized_fan, array-like
        Array-like of random values the same size as shape parameter

    References
    ----------
    Understanding the difficulty of training deep feedforward neural networks
        X. Glorot, Y. Bengio
    """
    return scale * np_tanh_fan_uniform(shape, random_state)


def np_sigmoid_fan_normal(shape, random_state, scale=4.):
    """
    Builds a numpy variable filled with random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 4.)
        default of 4. results in normal random values
        with 4 * sqrt(2 / (fan in + fan out)) scale

    Returns
    -------
    initialized_fan, array-like
        Array-like of random values the same size as shape parameter

    References
    ----------
    Understanding the difficulty of training deep feedforward neural networks
        X. Glorot, Y. Bengio
    """
    return scale * np_tanh_fan_normal(shape, random_state)


def np_variance_scaled_uniform(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 1.)
        default of 1. results in uniform random values
        with 1 * sqrt(1 / (n_dims)) scale

    Returns
    -------
    initialized_scaled, array-like
        Array-like of random values the same size as shape parameter

    References
    ----------
    Efficient Backprop
        Y. LeCun, L. Bottou, G. Orr, K. Muller
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        kern_sum = np.prod(shape[0])
    else:
        shp = shape
        kern_sum = shape[0]
    #  Make sure bounds aren't the same
    bound = scale * np.sqrt(3. / kern_sum)  # sqrt(3) for std of uniform
    return random_state.uniform(low=-bound, high=bound, size=shp).astype(_type)


def np_variance_scaled_randn(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 1.)
        default of 1. results in normal random values
        with 1 * sqrt(1 / (n_dims)) scale

    Returns
    -------
    initialized_scaled, array-like
        Array-like of random values the same size as shape parameter

    References
    ----------
    Efficient Backprop
        Y. LeCun, L. Bottou, G. Orr, K. Muller
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        kern_sum = np.prod(shape[0])
    else:
        shp = shape
        kern_sum = shape[0]
    # Make sure bounds aren't the same
    std = scale * np.sqrt(1. / kern_sum)
    return std * random_state.randn(*shp).astype(_type)


def np_deep_scaled_uniform(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 1.)
        default of 1. results in uniform random values
        with 1 * sqrt(6 / (n_dims)) scale

    Returns
    -------
    initialized_deep, array-like
        Array-like of random values the same size as shape parameter

    References
    ----------
    Diving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet
        K. He, X. Zhang, S. Ren, J. Sun
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        kern_sum = np.prod(shape[0])
    else:
        shp = shape
        kern_sum = shape[0]
    #  Make sure bounds aren't the same
    bound = scale * np.sqrt(6. / kern_sum)  # sqrt(3) for std of uniform
    return random_state.uniform(low=-bound, high=bound, size=shp).astype(_type)


def np_deep_scaled_normal(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 1.)
        default of 1. results in normal random values
        with 1 * sqrt(2 / (n_dims)) scale

    Returns
    -------
    initialized_deep, array-like
        Array-like of random values the same size as shape parameter

    References
    ----------
    Diving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet
        K. He, X. Zhang, S. Ren, J. Sun
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        kern_sum = np.prod(shape[0])
    else:
        shp = shape
        kern_sum = shape[0]
    # Make sure bounds aren't the same
    std = scale * np.sqrt(2. / kern_sum)  # sqrt(3) for std of uniform
    return std * random_state.randn(*shp).astype(_type)


def np_ortho(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with orthonormal random values

    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))

    random_state, numpy.random.RandomState() object

    scale, float (default 1.)
        default of 1. results in orthonormal random values sacled by 1.

    Returns
    -------
    initialized_ortho, array-like
        Array-like of random values the same size as shape parameter

    References
    ----------
    Exact solutions to the nonlinear dynamics of learning in deep linear
    neural networks
        A. Saxe, J. McClelland, S. Ganguli
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        flat_shp = (shp[0], np.prd(shp[1:]))
    else:
        shp = shape
        flat_shp = shape
    g = random_state.randn(*flat_shp)
    U, S, VT = linalg.svd(g, full_matrices=False)
    res = U if U.shape == flat_shp else VT  # pick one with the correct shape
    res = res.reshape(shp)
    return (scale * res).astype(_type)


def np_identity(shape, random_state, scale=0.98):
    """
    Identity initialization for square matrices

    Parameters
    ----------
    shape, tuple of ints
        shape of resulting array - shape[0] and shape[1] must match

    random_state, numpy.random.RandomState() object

    scale, float (default 0.98)
        default of .98 results in .98 * eye initialization

    Returns
    -------
    initialized_identity, array-like
        identity initialized square matrix same size as shape

    References
    ----------
    A Simple Way To Initialize Recurrent Networks of Rectified Linear Units
        Q. Le, N. Jaitly, G. Hinton
    """
    assert shape[0] == shape[1]
    res = np.eye(shape[0])
    return (scale * res).astype(_type)


def softplus_activation(X, eps=1E-4):
    return tensor.nnet.softplus(X) + eps


def relu_activation(X):
    return X * (X > 0)


def linear_activation(X):
    return X


def softmax_activation(X):
    # should work for both 2D and 3D
    e_X = tensor.exp(X - X.max(axis=-1, keepdims=True))
    out = e_X / e_X.sum(axis=-1, keepdims=True)
    return out


def _dropout(X, random_state, on_off_switch, p=0.):
    if p > 0:
        theano_seed = random_state.randint(-2147462579, 2147462579)
        # Super edge case...
        if theano_seed == 0:
            print("WARNING: prior layer got 0 seed. Reseeding...")
            theano_seed = random_state.randint(-2**32, 2**32)
        theano_rng = MRG_RandomStreams(seed=theano_seed)
        retain_prob = 1 - p
        if X.ndim == 2:
            X *= theano_rng.binomial(
                X.shape, p=retain_prob,
                dtype=_type) ** on_off_switch
            X /= retain_prob
        elif X.ndim == 3:
            # Dropout for recurrent - don't drop over time!
            X *= theano_rng.binomial((
                X.shape[1], X.shape[2]), p=retain_prob,
                dtype=_type) ** on_off_switch
            X /= retain_prob
        else:
            raise ValueError("Unsupported tensor with ndim %s" % str(X.ndim))
    return X


def dropout(list_of_inputs, graph, name, on_off_switch, dropout_prob=0.5,
            random_state=None):
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        print("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    conc_input = concatenate(list_of_inputs,
                             axis=list_of_inputs[0].ndim - 1)
    dropped = _dropout(conc_input, random_state, on_off_switch, p=dropout_prob)
    return dropped


def fixed_projection(list_of_inputs, list_of_input_dims, transform,
                     name=None, pre=None, post=None, strict=True):
    assert len(list_of_input_dims) == len(list_of_inputs)
    conc_input_dim = sum(list_of_input_dims)
    conc_input = concatenate(list_of_inputs,
                             axis=list_of_inputs[0].ndim - 1)
    """
    W_name = name + '_W'
    pre_name = name + '_pre'
    post_name = name + '_post'
    W_name = get_name()
    pre_name = get_name()
    post_name = get_name()
    """

    np_W = transform.astype(_type)
    W = as_shared(np_W)

    if pre is None:
        np_pre = np.zeros((conc_input_dim,)).astype(_type)
    else:
        np_pre = pre

    t_pre = as_shared(np_pre)

    if post is None:
        np_post = np.zeros_like(np_W[0]).astype(_type)
    else:
        np_post = post

    t_post = as_shared(np_post)
    logger.info((conc_input_dim, np_W[0].shape))
    return tensor.dot(conc_input + t_pre, W) + t_post


def _batch_normalization(input_variable, name, mode_switch,
                         alpha=0.5, strict=True):
    """Based on batch normalization by Jan Schluter for Lasagne"""
    raise ValueError("NYI")
    G_name = name + '_G'
    B_name = name + '_B'
    list_of_names = [G_name, B_name]
    if not names_in_graph(list_of_names, graph):
        input_dim = calc_expected_dims(graph, input_variable)[-1]
        np_G = np_ones((input_dim,))
        np_B = np_zeros((input_dim,))
        add_arrays_to_graph([np_G, np_B], list_of_names, graph,
                            strict=strict)
    else:
        if strict:
            raise AttributeError(
                "Name %s already found in graph with strict mode!" % name)
    G, B = fetch_from_graph(list_of_names, graph)
    eps = 1E-20
    batch_mean = input_variable.mean(axis=0, keepdims=True)
    batch_std = input_variable.std(axis=0, keepdims=True)
    running_mean_shape = calc_expected_dims(graph, batch_mean)
    running_std_shape = calc_expected_dims(graph, batch_std)
    running_mean = theano.clone(batch_mean, share_inputs=True)
    running_std = theano.clone(batch_std, share_inputs=True)
    running_mean, running_std = add_random_to_graph(
        [running_mean, running_std],
        [running_mean_shape, running_std_shape],
        [name + '_running_mean', name + '_running_std'], graph)
    running_mean.default_update = ((1 - alpha) * running_mean
                                   + alpha * batch_mean)
    running_std.default_update = ((1 - alpha) * running_std
                                  + alpha * batch_std)
    running_mean = tensor.addbroadcast(running_mean, 0)
    running_std = tensor.addbroadcast(running_std, 0)
    batch_mean += 0 * running_mean
    batch_std += 0 * running_std
    # include running_{mean, std} in computation graph for updates...
    fixed = (input_variable - running_mean) / (running_std + eps)
    batch = (input_variable - batch_mean) / (batch_std + eps)
    normed = (1 - mode_switch) * batch + mode_switch * fixed
    out = G * normed + B
    return out


def projection(list_of_inputs, list_of_input_dims, proj_dim, name=None,
               batch_normalize=False, mode_switch=None,
               random_state=None, strict=True,
               init_weights=None, init_biases=None,
               init_func=np_tanh_fan_uniform, act_func=linear_activation):
    assert len(list_of_input_dims) == len(list_of_inputs)
    conc_input_dim = sum(list_of_input_dims)

    if name is None:
        W_name = get_name()
        b_name = get_name()
    else:
        W_name = name + "_W"
        b_name = name + "_b"

    try:
        W = get_shared(W_name)
        if strict:
            raise AttributeError(
                "Name %s already found in parameters, strict mode!" % name)
    except NameError:
        assert random_state is not None
        if init_weights is None:
            np_W = init_func((conc_input_dim, proj_dim), random_state)
        else:
            np_W = init_weights
        W = as_shared(np_W)
        set_shared(W_name, W)

    try:
        b = get_shared(b_name)
        if strict:
            raise AttributeError(
                "Name %s already found in parameters, strict mode!" % name)
    except NameError:
        if init_biases is None:
            np_b = np_zeros((proj_dim,))
        else:
            np_b = init_biases
        b = as_shared(np_b)
        set_shared(b_name, b)

    conc_input = concatenate(list_of_inputs,
                             axis=list_of_inputs[0].ndim - 1)
    output = tensor.dot(conc_input, W) + b

    if batch_normalize:
        assert mode_switch is not None
        output = _batch_normalization(output, name, mode_switch)

    if act_func is not None:
        final = act_func(output)
    else:
        final = output
    return final


def linear(list_of_inputs, list_of_input_dims, proj_dim, name=None,
           batch_normalize=False, mode_switch=None,
           random_state=None, strict=True, init_func=np_tanh_fan_uniform):
    if name is None:
        name = get_name()
    else:
        name = name + "_linear"
    return projection(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        proj_dim=proj_dim, name=name, batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=linear_activation)


def softmax(list_of_inputs, list_of_input_dims, proj_dim, name=None,
            batch_normalize=False, mode_switch=None,
            random_state=None, strict=True, init_func=np_tanh_fan_uniform):
    if name is None:
        name = get_name()
    else:
        name = name + "_softmax"
    return projection(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        proj_dim=proj_dim, name=name, batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=softmax_activation)


def sigmoid(list_of_inputs, list_of_input_dims, proj_dim, name=None,
            batch_normalize=False, mode_switch=None,
            random_state=None, strict=True, init_func=np_sigmoid_fan_uniform):
    if name is None:
        name = get_name()
    else:
        name = name + "_sigmoid"
    return projection(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        proj_dim=proj_dim, name=name, batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=tensor.nnet.sigmoid)


def tanh(list_of_inputs, list_of_input_dims, proj_dim, name=None,
         batch_normalize=False, mode_switch=None,
         random_state=None, strict=True, init_func=np_tanh_fan_uniform):
    if name is None:
        name = get_name()
    else:
        name = name + "_tanh"
    return projection(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        proj_dim=proj_dim, name=name, batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=tensor.tanh)


def relu(list_of_inputs, list_of_input_dims, proj_dim, name=None,
         batch_normalize=False, mode_switch=None,
         random_state=None, strict=True, init_func=np_tanh_fan_uniform):
    if name is None:
        name = get_name()
    else:
        name = name + "_relu"
    return projection(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        proj_dim=proj_dim, name=name, batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=relu_activation)


def softplus(list_of_inputs, list_of_input_dims, proj_dim, name=None,
             batch_normalize=False, mode_switch=None,
             random_state=None, strict=True, init_func=np_tanh_fan_uniform):
    if name is None:
        name = get_name()
    else:
        name = name + "_softplus"
    return projection(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        proj_dim=proj_dim, name=name, batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=softplus_activation)


def exponential(list_of_inputs, list_of_input_dims, proj_dim, name=None,
                batch_normalize=False, mode_switch=None,
                random_state=None, strict=True, init_func=np_tanh_fan_uniform):
    if name is None:
        name = get_name()
    else:
        name = name + "_exponential"
    return projection(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        proj_dim=proj_dim, name=name, batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=tensor.exp)


def maxout(list_of_inputs, list_of_input_dims, proj_dim, name=None,
           maxout_rank=2,
           batch_normalize=False, mode_switch=None,
           random_state=None, strict=True,
           init_func=np_tanh_fan_uniform):
    if name is None:
        name = get_name()
    else:
        name = name + "_maxout"
    assert maxout_rank >= 1
    assert proj_dim is not None
    M = projection(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        proj_dim=maxout_rank * proj_dim, name=name,
        batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=linear_activation)
    if M.ndim == 2:
        dim0, dim1 = M.shape
        t = M.reshape((dim0, proj_dim, maxout_rank))
    elif M.ndim == 3:
        dim0, dim1, dim2 = M.shape
        t = M.reshape((dim0, dim1, proj_dim, maxout_rank))
    else:
        raise ValueError("input ndim not supported for maxout layer")
    maxout_out = tensor.max(t, axis=t.ndim - 1)
    return maxout_out


def softmax_zeros(list_of_inputs, list_of_input_dims, proj_dim, name=None,
                  batch_normalize=False, mode_switch=None,
                  random_state=None, strict=True,
                  init_func=None):
    def softmax_init(shape, random_state):
        return np_zeros(shape)
    if init_func is not None:
        raise ValueError("Init func options not available for softmax_zeros")
    else:
        init_func = softmax_init
    if name is None:
        name = get_name()
    else:
        name = name + "_softmax_zeros"
    if random_state is None:
        # just make one up
        random_state = np.random.RandomState(0)
    return projection(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        proj_dim=proj_dim, name=name, batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=softmax_activation)


def log_gaussian_mixture(list_of_inputs, list_of_input_dims, proj_dim,
                         name=None, n_components=5,
                         batch_normalize=False, mode_switch=None,
                         random_state=None, strict=True,
                         init_func=np_tanh_fan_uniform):
    assert n_components >= 1
    if name is None:
        name = get_name()
    else:
        name = name + "_log_gaussian_mixture"

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
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        proj_dim=n_components * proj_dim, name=name + "_mus",
        batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=linear_activation)
    log_sigmas = projection(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        proj_dim=n_components * proj_dim, name=name + "_log_sigmas",
        batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=linear_activation)
    coeffs = projection(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        proj_dim=n_components, name=name + "_coeffs",
        batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=softmax_activation)
    mus = _reshape(mus)
    log_sigmas = _reshape(log_sigmas)
    return coeffs, mus, log_sigmas


def bernoulli_and_correlated_log_gaussian_mixture(
    list_of_inputs, list_of_input_dims, proj_dim, name=None, n_components=5,
    batch_normalize=False, mode_switch=None, random_state=None, strict=True,
    init_func=np_tanh_fan_uniform):
    assert n_components >= 1
    if name is None:
        name = get_name()
    else:
        name = name + "_bernoulli_and_correlated_log_gaussian_mixture"

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
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        proj_dim=n_components * proj_dim, name=name + "_mus",
        batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=linear_activation)
    log_sigmas = projection(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        proj_dim=n_components * proj_dim, name=name + "_log_sigmas",
        batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=linear_activation)
    coeffs = projection(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        proj_dim=n_components, name=name + "_coeffs",
        batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=softmax_activation)
    mus = _reshape(mus)
    log_sigmas = _reshape(log_sigmas)

    calc_corr = factorial(proj_dim ** 2 // 2 - 1)
    corr = projection(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        proj_dim=n_components * calc_corr, name=name + "_corr",
        batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=tensor.tanh)
    binary = projection(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        proj_dim=1, name=name + "_binary",
        batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=np_sigmoid_fan_uniform,
        act_func=tensor.nnet.sigmoid)
    mus = _reshape(mus)
    log_sigmas = _reshape(log_sigmas)
    corr = _reshape(corr, calc_corr)
    return binary, coeffs, mus, log_sigmas, corr


def conv2d(list_of_inputs, list_of_input_dims, num_feature_maps,
           kernel_size=(3, 3),
           dilation=1, border_mode="same", name=None, batch_normalize=False,
           mode_switch=None,
           random_state=None, strict=True, init_func=np_tanh_fan_uniform,
           act_func=relu_activation):
    if dilation != 1:
        raise ValueError("Dilation not yet supported")
    if name is None:
        name = get_name()
    else:
        name = name + "_conv2d"
    if batch_normalize is not False:
        raise ValueError("BN not yet implemented")

    assert len(list_of_input_dims) == len(list_of_inputs)
    conc_input = concatenate(list_of_inputs, axis=1)
    # assumes bc01 format
    input_channels = sum([inp[0] for inp in list_of_input_dims])
    input_width = sum([inp[1] for inp in list_of_input_dims])
    input_height = sum([inp[2] for inp in list_of_input_dims])

    W_name = name + "_W"
    try:
        W = get_shared(W_name)
        if strict:
            raise AttributeError(
                "Name %s already found in parameters, strict mode!" % name)
    except NameError:
        np_W = init_func(((input_channels, input_width, input_height),
                         (num_feature_maps, kernel_size[0], kernel_size[1])),
                         random_state=random_state).astype(_type)
        W = as_shared(np_W)
        set_shared(W_name, W)

    b_name = name + "_b"
    try:
        b = get_shared(b_name)
        if strict:
            raise AttributeError(
                "Name %s already found in parameters, strict mode!" % name)
    except NameError:
        np_b = np_zeros((num_feature_maps,)).astype(_type)
        np_b = np_b.reshape((num_feature_maps,))
        b = as_shared(np_b)
        set_shared(b_name, b)
    b = b.dimshuffle('x', 0, 'x', 'x')

    s = int(np.floor(W.get_value().shape[-1] / 2.))
    z = tensor.nnet.conv2d(
        conc_input, W, border_mode='full')[:, :, s:-s, s:-s] + b
    # TODO: nvidia conv
    # z = dnn_conv(X, w, border_mode=int(np.floor(w.get_value().shape[-1]/2.)))
    return act_func(z)


def pool2d(list_of_inputs, pool_size=(2, 2), pool_func="max", name=None):
    if name is None:
        name = get_name()
    else:
        name = name + "_pool2d"
    conc_input = concatenate(list_of_inputs, axis=1)
    if pool_func == "max":
        act_func = max_pool_2d
    else:
        raise ValueError("pool_func %s not supported" % pool_func)
    return tensor.cast(act_func(conc_input, pool_size, ignore_border=True),
                       dtype=_type)


def embed(list_of_index_inputs, max_index, proj_dim, name=None,
          random_state=None, init_func=np_unit_uniform):
    check_type = any([index_input.dtype != "int32"
                      for index_input in list_of_index_inputs])
    lii = list_of_index_inputs
    if check_type:
        lii = [tensor.cast(li, "int32") for li in lii]
    if name is None:
        name = get_name()
    embed_W_name = name + "_embed_W"
    try:
        embed_W = get_shared(embed_W_name)
        if strict:
            raise AttributeError(
                "Name %s already found in parameters, strict mode!" % name)
    except NameError:
        np_embed_W = init_func((max_index, proj_dim), random_state)
        embed_W = as_shared(np_embed_W)
        set_shared(embed_W_name, embed_W)

    embeddings = [embed_W[index_input]
                  for index_input in lii]
    output = concatenate(embeddings, axis=embed_W.ndim - 1)
    n_lists = len(list_of_index_inputs)
    if n_lists != 1:
        raise ValueError("Unsupported number of list_of_index_inputs, currently only supports 1 element")
    o = output.reshape((-1, index_input.shape[1], proj_dim * n_lists))
    return o
