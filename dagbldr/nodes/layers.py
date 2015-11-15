# Author: Kyle Kastner
# License: BSD 3-clause
import numpy as np
from scipy import linalg
import theano
from theano import tensor
from theano.tensor.signal.downsample import max_pool_2d
from theano.sandbox.rng_mrg import MRG_RandomStreams
from ..utils import concatenate
from ..utils import calc_expected_dims, names_in_graph, add_arrays_to_graph
from ..utils import add_fixed_to_graph
from ..utils import fetch_from_graph, add_random_to_graph


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
    return np.zeros(shape).astype(theano.config.floatX)


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
    return np.ones(shape).astype(theano.config.floatX)


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
    return random_state.uniform(low=-scale, high=scale, size=shp).astype(
        theano.config.floatX)


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
    return (scale * random_state.randn(*shp)).astype(theano.config.floatX)


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
    return random_state.uniform(low=-bound, high=bound,
                                size=shp).astype(theano.config.floatX)


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
    return var * random_state.randn(*shp).astype(theano.config.floatX)


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
    return random_state.uniform(low=-bound, high=bound, size=shp).astype(
        theano.config.floatX)


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
    return std * random_state.randn(*shp).astype(theano.config.floatX)


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
    return random_state.uniform(low=-bound, high=bound, size=shp).astype(
        theano.config.floatX)


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
    return std * random_state.randn(*shp).astype(theano.config.floatX)


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
    return (scale * res).astype(theano.config.floatX)


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
    return (scale * res).astype(theano.config.floatX)


def softplus(X):
    return tensor.nnet.softplus(X) + 1E-4


def relu(X):
    return X * (X > 0)


def linear(X):
    return X


def softmax(X):
    # should work for both 2D and 3D
    e_X = tensor.exp(X - X.max(axis=-1, keepdims=True))
    out = e_X / e_X.sum(axis=-1, keepdims=True)
    return out


def dropout(X, random_state, on_off_switch, p=0.):
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
                dtype=theano.config.floatX) ** on_off_switch
            X /= retain_prob
        elif X.ndim == 3:
            # Dropout for recurrent - don't drop over time!
            X *= theano_rng.binomial((
                X.shape[1], X.shape[2]), p=retain_prob,
                dtype=theano.config.floatX) ** on_off_switch
            X /= retain_prob
        else:
            raise ValueError("Unsupported tensor with ndim %s" % str(X.ndim))
    return X


def dropout_layer(list_of_inputs, graph, name, on_off_switch, dropout_prob=0.5,
                  random_state=None):
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        print("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    conc_input = concatenate(list_of_inputs, graph, name,
                             axis=list_of_inputs[0].ndim - 1)
    dropped = dropout(conc_input, random_state, on_off_switch, p=dropout_prob)
    return dropped


def fixed_projection_layer(list_of_inputs, transform, graph, name,
                           pre=None, post=None, strict=True):
    conc_input = concatenate(list_of_inputs, graph, name,
                             axis=list_of_inputs[0].ndim - 1)
    W_name = name + '_W'
    pre_name = name + '_pre'
    post_name = name + '_post'
    list_of_names = [W_name, pre_name, post_name]
    if not names_in_graph(list_of_names, graph):
        conc_input_dim = int(sum([calc_expected_dims(graph, inp)[-1]
                                  for inp in list_of_inputs]))
        np_W = transform.astype(theano.config.floatX)

        if pre is None:
            np_pre = np.zeros((conc_input_dim,)).astype(
                theano.config.floatX)
        else:
            np_pre = pre

        if post is None:
            np_post = np.zeros_like(np_W[0]).astype(
                theano.config.floatX)
        else:
            np_post = post

        list_of_shapes = [np_W.shape, np_pre.shape, np_post.shape]
        W, t_pre, t_post = add_fixed_to_graph([np_W, np_pre, np_post],
                                              list_of_shapes,
                                              list_of_names, graph,
                                              strict=strict)
    else:
        if strict:
            raise AttributeError(
                "Name %s already found in graph with strict mode!" % name)
        else:
            raise AttributeError(
                "Repeated node use not yet supported")
    return tensor.dot(conc_input + t_pre, W) + t_post


def embedding_layer(list_of_index_inputs, max_index, proj_dim, graph, name,
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


def _batch_normalization(input_variable, graph, name, mode_switch,
                         alpha=0.5, strict=True):
    """Based on batch normalization by Jan Schluter for Lasagne"""
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
    normed_shape = calc_expected_dims(graph, input_variable)
    # necessary to get decent shape inference
    out, = add_random_to_graph([out], [normed_shape], [name + "_bn_out"], graph)
    return out


def projection_layer(list_of_inputs, graph, name, proj_dim=None,
                     batch_normalize=False, mode_switch=None,
                     random_state=None, strict=True,
                     init_func=np_tanh_fan_uniform, func=linear):
    W_name = name + '_W'
    b_name = name + '_b'
    list_of_names = [W_name, b_name]
    if not names_in_graph(list_of_names, graph):
        assert proj_dim is not None
        assert random_state is not None
        conc_input_dim = int(sum([calc_expected_dims(graph, inp)[-1]
                                  for inp in list_of_inputs]))
        np_W = init_func((conc_input_dim, proj_dim), random_state)
        np_b = np_zeros((proj_dim,))
        add_arrays_to_graph([np_W, np_b], list_of_names, graph,
                            strict=strict)
    else:
        if strict:
            raise AttributeError(
                "Name %s already found in graph with strict mode!" % name)
    W, b = fetch_from_graph(list_of_names, graph)
    conc_input = concatenate(list_of_inputs, graph, name,
                             axis=list_of_inputs[0].ndim - 1)
    output = tensor.dot(conc_input, W) + b
    if batch_normalize:
        assert mode_switch is not None
        output = _batch_normalization(output, graph, name, mode_switch)
    if func is not None:
        final = func(output)
    else:
        final = output
    return final


def linear_layer(list_of_inputs, graph, name, proj_dim=None,
                 batch_normalize=False, mode_switch=None,
                 random_state=None, strict=True, init_func=np_tanh_fan_uniform):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, func=linear)


def sigmoid_layer(list_of_inputs, graph, name, proj_dim=None,
                  batch_normalize=False, mode_switch=None,
                  random_state=None, strict=True,
                  init_func=np_sigmoid_fan_uniform):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, func=tensor.nnet.sigmoid)


def tanh_layer(list_of_inputs, graph, name, proj_dim=None,
               batch_normalize=False, mode_switch=None,
               random_state=None, strict=True, init_func=np_tanh_fan_uniform):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, batch_normalize=batch_normalize,
        mode_switch=mode_switch,
        random_state=random_state, strict=strict, init_func=init_func,
        func=tensor.tanh)


def softplus_layer(list_of_inputs, graph, name, proj_dim=None,
                   batch_normalize=False, mode_switch=None,
                   random_state=None, strict=True,
                   init_func=np_tanh_fan_uniform):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, func=softplus)


def exp_layer(list_of_inputs, graph, name, proj_dim=None,
              batch_normalize=False, mode_switch=None,
              random_state=None, strict=True, init_func=np_tanh_fan_uniform):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, func=tensor.exp)


def relu_layer(list_of_inputs, graph, name, proj_dim=None,
               batch_normalize=False, mode_switch=None,
               random_state=None, strict=True, init_func=np_tanh_fan_uniform):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, func=relu)


def maxout_layer(list_of_inputs, graph, name, proj_dim=None,
                 batch_normalize=False, mode_switch=None,
                 random_state=None, maxout_rank=2, strict=True,
                 init_func=np_tanh_fan_uniform):
    assert maxout_rank >= 1
    assert proj_dim is not None
    M = projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=maxout_rank * proj_dim, batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, func=linear)
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


def conv2d_layer(list_of_inputs, graph, name, num_feature_maps=None,
                 kernel_size=(3, 3), batch_normalize=False, mode_switch=None,
                 random_state=None, strict=True, init_func=np_tanh_fan_uniform,
                 func=relu):
    W_name = name + '_W'
    b_name = name + '_b'
    list_of_names = [W_name, b_name]
    if batch_normalize is not False:
        raise ValueError("Batch Normalization not yet implemented")
    if not names_in_graph(list_of_names, graph):
        assert num_feature_maps is not None
        assert random_state is not None
        assert kernel_size is not None
        # assumes bc01 format
        input_channels = int(sum([calc_expected_dims(graph, inp)[1]
                                  for inp in list_of_inputs]))
        input_width = [calc_expected_dims(graph, inp)[2]
                       for inp in list_of_inputs][0]
        input_height = [calc_expected_dims(graph, inp)[3]
                        for inp in list_of_inputs][0]
        np_W = init_func(((input_channels, input_width, input_height),
                          (num_feature_maps, kernel_size[0], kernel_size[1])),
                         random_state=random_state)
        np_b = np_zeros((num_feature_maps,))
        np_b = np_b.reshape((num_feature_maps,))
        add_arrays_to_graph([np_W, np_b], list_of_names, graph, strict=strict)
    else:
        if strict:
            raise AttributeError(
                "Name %s already found in graph with strict mode!" % name)
    conc_input = concatenate(list_of_inputs, graph, name, axis=1)
    W, b = fetch_from_graph(list_of_names, graph)
    b = b.dimshuffle('x', 0, 'x', 'x')
    s = int(np.floor(W.get_value().shape[-1] / 2.))
    z = tensor.nnet.conv2d(
        conc_input, W, border_mode='full')[:, :, s:-s, s:-s] + b
    # TODO: nvidia conv
    # z = dnn_conv(X, w, border_mode=int(np.floor(w.get_value().shape[-1]/2.)))
    return func(z)


def pool2d_layer(list_of_inputs, graph, name, pool_size=(2, 2),
                 pool_func="max"):
    conc_input = concatenate(list_of_inputs, graph, name, axis=1)
    if pool_func == "max":
        func = max_pool_2d
    else:
        raise ValueError("pool_func %s not supported" % pool_func)
    return func(conc_input, pool_size)


def softmax_zeros_layer(list_of_inputs, graph, name, proj_dim=None,
                        strict=True):
    def softmax_init(shape, random_state):
        return np_zeros(shape)
    return projection_layer(list_of_inputs=list_of_inputs, graph=graph,
                            name=name, proj_dim=proj_dim,
                            batch_normalize=False, mode_switch=None,
                            # random state is unused but necessary for proj
                            random_state=np.random.RandomState(0),
                            strict=strict, init_func=softmax_init,
                            func=softmax)


def softmax_layer(list_of_inputs, graph, name, proj_dim=None,
                  random_state=None, strict=True,
                  init_func=np_tanh_fan_uniform):
    return projection_layer(list_of_inputs=list_of_inputs, graph=graph,
                            name=name, proj_dim=proj_dim,
                            random_state=random_state,
                            strict=strict, init_func=init_func,
                            func=softmax)


def softmax_sample_layer(list_of_multinomial_inputs, graph, name,
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


def gaussian_sample_layer(list_of_mu_inputs, list_of_sigma_inputs,
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


def gaussian_log_sample_layer(list_of_mu_inputs, list_of_log_sigma_inputs,
                              graph, name, random_state=None):
    """ log_sigma_inputs should be from a linear_layer """
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


def tanh_recurrent_layer(list_of_inputs, mask, hidden_dim, graph, name,
                         random_state, strict=True):
    ndim = [len(calc_expected_dims(graph, inp)) for inp in list_of_inputs]
    check = [n for n in ndim if n != 3]
    if len(check) > 0:
        raise ValueError("Input with ndim != 3 detected!")

    # shape[0] is fake, but shape[1] and shape[2] are fine
    conc_input = concatenate(list_of_inputs, graph, name + "_input",
                             axis=list_of_inputs[0].ndim - 1)
    shape = calc_expected_dims(graph, conc_input)
    h0 = np_zeros((shape[1], hidden_dim))
    list_of_names = [name + '_h0']
    add_arrays_to_graph([h0], list_of_names, graph)
    h0_sym, = fetch_from_graph(list_of_names, graph)

    W_name = name + '_tanh_rec_step_W'
    b_name = name + '_tanh_rec_step_b'
    U_name = name + '_tanh_rec_step_U'
    list_of_names = [W_name, b_name, U_name]
    if not names_in_graph(list_of_names, graph):
        assert random_state is not None
        conc_input_dim = int(sum([calc_expected_dims(graph, inp)[-1]
                                  for inp in list_of_inputs]))
        shape = (conc_input_dim, hidden_dim)
        np_W = np_uniform(shape, random_state)
        np_b = np_zeros((shape[-1],))
        np_U = np_ortho((shape[-1], shape[-1]), random_state)
        add_arrays_to_graph([np_W, np_b, np_U], list_of_names, graph,
                            strict=strict)
    else:
        if strict:
            raise AttributeError(
                "Name %s already found in graph with strict mode!" % name)

    W, b, U = fetch_from_graph(list_of_names, graph)
    projected_input = tensor.dot(conc_input, W) + b

    def step(x_t, m_t, h_tm1, U):
        h_ti = tensor.tanh(x_t + tensor.dot(h_tm1, U))
        h_t = m_t[:, None] * h_ti + (1 - m_t)[:, None] * h_tm1
        return h_t

    h, updates = theano.scan(step, name=name + '_tanh_recurrent_scan',
                             sequences=[projected_input, mask],
                             outputs_info=[h0_sym],
                             non_sequences=[U])
    return h


def gru_recurrent_layer(list_of_inputs, mask, hidden_dim, graph, name,
                        random_state, strict=True):
    ndim = [len(calc_expected_dims(graph, inp)) for inp in list_of_inputs]
    check = [n for n in ndim if n != 3]
    if len(check) > 0:
        raise ValueError("Input with ndim != 3 detected!")

    # shape[0] is fake, but shape[1] and shape[2] are fine
    conc_input = concatenate(list_of_inputs, graph, name + "_input",
                             axis=list_of_inputs[0].ndim - 1)
    shape = calc_expected_dims(graph, conc_input)
    h0 = np_zeros((shape[1], hidden_dim))
    list_of_names = [name + '_h0']
    add_arrays_to_graph([h0], list_of_names, graph)
    h0_sym, = fetch_from_graph(list_of_names, graph)

    W_name = name + '_gru_rec_step_W'
    b_name = name + '_gru_rec_step_b'
    Urz_name = name + '_gru_rec_step_Urz'
    U_name = name + '_gru_rec_step_U'
    list_of_names = [W_name, b_name, Urz_name, U_name]
    if not names_in_graph(list_of_names, graph):
        assert random_state is not None
        conc_input_dim = int(sum([calc_expected_dims(graph, inp)[-1]
                                  for inp in list_of_inputs]))
        shape = (conc_input_dim, hidden_dim)
        np_W = np.hstack([np_uniform(shape, random_state),
                          np_uniform(shape, random_state),
                          np_uniform(shape, random_state)])
        np_b = np_zeros((3 * shape[1],))
        np_Urz = np.hstack([np_ortho((shape[1], shape[1]), random_state),
                            np_ortho((shape[1], shape[1]), random_state), ])
        np_U = np_ortho((shape[1], shape[1]), random_state)
        add_arrays_to_graph([np_W, np_b, np_Urz, np_U], list_of_names, graph,
                            strict=strict)
    else:
        if strict:
            raise AttributeError(
                "Name %s already found in graph with strict mode!" % name)

    W, b, Urz, U = fetch_from_graph(list_of_names, graph)
    projected_input = tensor.dot(conc_input, W) + b

    def _slice(arr, n):
        # First slice is tensor_dim - 1 sometimes with scan...
        # need to be *very* careful and test with strict=False and reusing stuff
        # since shape is redefined in if not names_in_graph...
        dim = shape[1]
        if arr.ndim == 3:
            return arr[:, :, n * dim:(n + 1) * dim]
        return arr[:, n * dim:(n + 1) * dim]

    def step(x_t, m_t, h_tm1, U):
        projected_gates = tensor.dot(h_tm1, Urz)
        r = tensor.nnet.sigmoid(_slice(x_t, 0) + _slice(projected_gates, 0))
        z = tensor.nnet.sigmoid(_slice(x_t, 1) + _slice(projected_gates, 1))
        candidate_h_t = tensor.tanh(_slice(x_t, 2) + tensor.dot(r * h_tm1, U))
        h_ti = z * h_tm1 + (1. - z) * candidate_h_t
        h_t = m_t[:, None] * h_ti + (1 - m_t)[:, None] * h_tm1
        return h_t

    h, updates = theano.scan(step, name=name + '_gru_recurrent_scan',
                             sequences=[projected_input, mask],
                             outputs_info=[h0_sym],
                             non_sequences=[U])
    return h


def bidirectional_gru_recurrent_layer(list_of_inputs, mask, hidden_dim, graph,
                                      name, random_state, strict=True):
    h_f = gru_recurrent_layer(list_of_inputs, mask, hidden_dim, graph,
                              name + "_f", random_state, strict=strict)
    h_r = gru_recurrent_layer([i[::-1] for i in list_of_inputs], mask[::-1],
                              hidden_dim, graph, name + "_r", random_state,
                              strict=strict)
    h = concatenate([h_f, h_r[::-1]], graph, name=name + "_conc",
                    axis=h_f.ndim - 1)
    return h


def shift_layer(list_of_inputs, graph, name):
    """
    Shifts along the first axis by one step, filling with 0 in the first step
    """
    conc_input = concatenate(list_of_inputs, graph, name + "_shifted",
                             axis=list_of_inputs[0].ndim - 1)
    shifted = tensor.zeros_like(conc_input)
    shifted = tensor.set_subtensor(shifted[1:], conc_input[:-1])
    return shifted


def conditional_gru_recurrent_layer(list_of_outputs, list_of_hiddens,
                                    output_mask, hidden_dim, graph, name,
                                    random_state, strict=True):
    """
    Feed list_of_outputs as unshifted outputs desired. Internally the node
    will shift by one time step.

    hidden_context is the hidden states from the encoder,
    in this case only useful to get the last hidden state.
    """
    # an easy interface to conditional gru recurrent nets
    # If the expressions are not the same length and batch size it won't work
    max_ndim = max([out.ndim for out in list_of_outputs])
    if max_ndim > 3:
        raise ValueError("Input with ndim > 3 detected!")

    conc_output = concatenate(list_of_outputs, graph, name + "_cond_gru_step",
                              axis=list_of_outputs[0].ndim - 1)
    conc_hidden = concatenate(list_of_hiddens, graph, name + "_cond_gru_hid",
                              axis=list_of_hiddens[0].ndim - 1)
    context = conc_hidden[-1]
    # Decoder initializes hidden state with tanh projection of last hidden
    # context representing p(X_1...X_t)
    conc_hidden_dim = calc_expected_dims(graph, conc_hidden)[-1]
    h0_sym = tanh_layer([context], graph, name + '_h0_proj',
                        proj_dim=conc_hidden_dim, random_state=random_state)
    shifted = tensor.zeros_like(conc_output)
    shifted = tensor.set_subtensor(shifted[1:], conc_output[:-1])
    input_shifted = shifted

    W_name = name + '_cond_gru_rec_step_W'
    b_name = name + '_cond_gru_rec_step_b'
    Urz_name = name + '_cond_gru_rec_step_Urz'
    U_name = name + '_cond_gru_rec_step_U'
    Wg_name = name + '_cond_gru_rec_step_Wg'
    bg_name = name + '_cond_gru_rec_step_bg'
    Wh_name = name + '_cond_gru_rec_step_Wh'
    bh_name = name + '_cond_gru_rec_step_bh'
    list_of_names = [W_name, b_name, Urz_name, U_name, Wg_name, bg_name,
                     Wh_name, bh_name]
    if not names_in_graph(list_of_names, graph):
        assert random_state is not None
        conc_input_dim = calc_expected_dims(graph, input_shifted)[-1]
        shape = (conc_input_dim, hidden_dim)
        np_W = np.hstack([np_uniform(shape, random_state),
                          np_uniform(shape, random_state),
                          np_uniform(shape, random_state)])
        np_b = np_zeros((3 * shape[1],))
        np_Urz = np.hstack([np_ortho((shape[1], shape[1]), random_state),
                            np_ortho((shape[1], shape[1]), random_state), ])
        np_U = np_ortho((shape[1], shape[1]), random_state)
        context_dim = calc_expected_dims(graph, context)[-1]
        np_Wg = np_uniform((context_dim, 2 * shape[1]), random_state)
        np_bg = np_zeros((2 * shape[1],))
        np_Wh = np_uniform((context_dim, shape[1]), random_state)
        np_bh = np_zeros((shape[1],))
        list_of_arrays = [np_W, np_b, np_Urz, np_U, np_Wg, np_bg, np_Wh, np_bh]
        add_arrays_to_graph(list_of_arrays, list_of_names, graph, strict=strict)
    else:
        if strict:
            raise AttributeError(
                "Name %s already found in graph with strict mode!" % name)

    W, b, Urz, U, Wg, bg, Wh, bh = fetch_from_graph(list_of_names, graph)
    projected_input = tensor.dot(input_shifted, W) + b
    projected_context_to_gates = tensor.dot(context, Wg) + bg
    projected_context_to_hidden = tensor.dot(context, Wh) + bh

    def _slice(arr, n):
        # First slice is tensor_dim - 1 sometimes with scan...
        # need to be *very* careful and test with strict=False and reusing stuff
        # since shape is redefined in if not names_in_graph...
        dim = shape[1]
        if arr.ndim == 3:
            return arr[:, :, n * dim:(n + 1) * dim]
        return arr[:, n * dim:(n + 1) * dim]

    def step(x_t, m_t, h_tm1, U, pcg, pch):
        projected_gates = tensor.dot(h_tm1, Urz) + pcg
        r = tensor.nnet.sigmoid(_slice(x_t, 0) + _slice(projected_gates, 0))
        z = tensor.nnet.sigmoid(_slice(x_t, 1) + _slice(projected_gates, 1))
        candidate_h_t = tensor.tanh(_slice(x_t, 2) + r * tensor.dot(
            h_tm1, U) + pch)
        h_ti = z * h_tm1 + (1. - z) * candidate_h_t
        h_t = m_t[:, None] * h_ti + (1 - m_t)[:, None] * h_tm1
        return h_t

    h, updates = theano.scan(step, name=name + '_cond_gru_recurrent_scan',
                             sequences=[projected_input, output_mask],
                             outputs_info=[h0_sym],
                             non_sequences=[U, projected_context_to_gates,
                                            projected_context_to_hidden])
    final_context = context.dimshuffle('x', 0, 1) * tensor.ones_like(h)
    return h, final_context


def conditional_attention_gru_recurrent_layer(list_of_outputs, list_of_hiddens,
                                              output_mask, hidden_mask,
                                              hidden_dim, graph,
                                              name, random_state, strict=True):
    """
    Feed list_of_outputs as unshifted outputs desired. Internally the node
    will shift by one time step.

    hidden_context is the hidden states from the encoder,
    in this case only useful to get the last hidden state.
    """
    # an easy interface to conditional gru recurrent nets
    # If the expressions are not the same length and batch size it won't work
    max_ndim = max([out.ndim for out in list_of_outputs])
    if max_ndim > 3:
        raise ValueError("Input with ndim > 3 detected!")

    conc_output = concatenate(list_of_outputs, graph, name + "_cond_gru_step",
                              axis=list_of_outputs[0].ndim - 1)
    conc_hidden = concatenate(list_of_hiddens, graph, name + "_cond_gru_hid",
                              axis=list_of_hiddens[0].ndim - 1)
    context = conc_hidden.mean(axis=0)
    # Decoder initializes hidden state with tanh projection of last hidden
    # context representing p(X_1...X_t)
    conc_hidden_dim = calc_expected_dims(graph, conc_hidden)[-1]
    h0_sym = tanh_layer([context], graph, name + '_h0_proj',
                        proj_dim=conc_hidden_dim, random_state=random_state)
    shifted = tensor.zeros_like(conc_output)
    shifted = tensor.set_subtensor(shifted[1:], conc_output[:-1])
    input_shifted = shifted
    conc_input_dim = calc_expected_dims(graph, input_shifted)[-1]

    # GRU weights
    W_name = name + '_cond_gru_rec_step_W'
    b_name = name + '_cond_gru_rec_step_b'
    Urz_name = name + '_cond_gru_rec_step_Urz'
    U_name = name + '_cond_gru_rec_step_U'

    W_context_to_hidden_name = name + '_cond_gru_rec_step_W_cth'
    W_context_to_candidate_name = name + '_cond_gru_rec_step_W_ctc'
    # Attention weights
    # Attention over shifted input sequence
    Wi_att_name = name + '_cond_gru_step_Wi_att'
    # Attention over previous hiddens
    Wc_att_name = name + '_cond_gru_step_Wc_att'
    # Attention bias for all, applied to Wc_att
    b_att = name + '_cond_gru_step_b_att'
    # Attention over state
    Ws_att_name = name + '_cond_gru_step_Ws_att'
    # Attention weights into softmax
    Wp_att_name = name + '_cond_gru_step_Wp_att'
    bp_att_name = name + '_cond_gru_step_bp_att'

    list_of_names = [W_name, b_name, Urz_name, U_name,
                     W_context_to_hidden_name, W_context_to_candidate_name,
                     Wi_att_name, Wc_att_name, b_att,
                     Ws_att_name, Wp_att_name, bp_att_name]
    if not names_in_graph(list_of_names, graph):
        assert random_state is not None
        np_W = np_uniform((conc_input_dim, 3 * conc_hidden_dim), random_state)
        np_b = np_zeros((3 * conc_hidden_dim))
        np_Urz = np.hstack([np_ortho((conc_hidden_dim, conc_hidden_dim),
                                     random_state),
                            np_ortho((conc_hidden_dim, conc_hidden_dim),
                                     random_state)])
        np_U = np_ortho((conc_hidden_dim, conc_hidden_dim), random_state)

        np_W_context_to_hidden = np_uniform((conc_hidden_dim,
                                             2 * conc_hidden_dim), random_state)
        np_W_context_to_candidate = np_uniform((conc_hidden_dim,
                                                conc_hidden_dim), random_state)
        # Init attention weights
        np_Wi_att = np_uniform((conc_input_dim, conc_hidden_dim), random_state)
        np_Wc_att = np_ortho((conc_hidden_dim, conc_hidden_dim), random_state)
        np_b_att = np_zeros((conc_hidden_dim,))
        np_Ws_att = np_ortho((conc_hidden_dim, conc_hidden_dim), random_state)
        np_Wp_att = np_uniform((conc_hidden_dim, 1), random_state)
        np_bp_att = np_zeros((1,))

        list_of_arrays = [np_W, np_b, np_Urz, np_U,
                          np_W_context_to_hidden, np_W_context_to_candidate,
                          np_Wi_att, np_Wc_att,
                          np_b_att, np_Ws_att, np_Wp_att, np_bp_att]
        add_arrays_to_graph(list_of_arrays, list_of_names, graph, strict=strict)
    else:
        if strict:
            raise AttributeError(
                "Name %s already found in graph with strict mode!" % name)

    (W, b, Urz, U,
     W_context_to_hidden, W_context_to_candidate,
     Wi_att, Wc_att, b_att, Ws_att,
     Wp_att, bp_att) = fetch_from_graph(list_of_names, graph)
    projected_hidden_attention = tensor.dot(conc_hidden, Wc_att) + b_att
    projected_input_attention = tensor.dot(input_shifted, Wi_att)
    projected_input = tensor.dot(input_shifted, W) + b

    def _slice(arr, n):
        # First slice is tensor_dim - 1 sometimes with scan...
        # need to be *very* careful and test with strict=False and reusing stuff
        # since shape is redefined in if not names_in_graph...
        dim = conc_hidden_dim
        if arr.ndim == 3:
            return arr[:, :, n * dim:(n + 1) * dim]
        return arr[:, n * dim:(n + 1) * dim]

    sequences = [projected_input, output_mask, projected_input_attention]
    (n_input_steps, n_samples, n_features) = conc_hidden.shape
    ctx0_sym = tensor.cast(tensor.alloc(0., n_samples, n_features),
                           theano.config.floatX)
    att0_sym = tensor.cast(tensor.alloc(0., n_samples, n_input_steps),
                           theano.config.floatX)

    outputs = [h0_sym, ctx0_sym, att0_sym]
    non_sequences = [projected_hidden_attention, conc_hidden,
                     U, W, W_context_to_hidden, W_context_to_candidate,
                     Ws_att, Wp_att, bp_att,
                     Wc_att, Urz, hidden_mask]

    def step(x_t, m_t, att_i_t,
             h_tm1, ctx_tm1, att_w_tm1,
             proj_hid_att, conc_hidden, U, W, W_cth, W_ctc, Ws_att,
             Wp_att, bp_att, Wc_att, Urz, hidden_mask):
        att_s = tensor.dot(h_tm1, Ws_att)
        att = proj_hid_att + att_s[None, :, :]
        att += att_i_t
        att = tensor.tanh(att)
        att_w_t = tensor.dot(att, Wp_att) + bp_att
        att_w_t = att_w_t.reshape((att_w_t.shape[0], att_w_t.shape[1]))  # ?
        att_w_t_max = (att_w_t * hidden_mask).max(axis=0, keepdims=True)
        att_w_t = tensor.exp(att_w_t - att_w_t_max)
        att_w_t = hidden_mask * att_w_t
        att_w_t = att_w_t / att_w_t.sum(axis=0, keepdims=True)
        ctx_t = (conc_hidden * att_w_t[:, :, None]).sum(axis=0)

        projected_state = tensor.dot(h_tm1, Urz)
        projected_state += tensor.dot(ctx_t, W_cth)

        r = tensor.nnet.sigmoid(_slice(x_t, 0) + _slice(projected_state, 0))
        z = tensor.nnet.sigmoid(_slice(x_t, 1) + _slice(projected_state, 1))
        candidate_h_t = tensor.tanh(_slice(x_t, 2) + r * tensor.dot(
            h_tm1, U) + tensor.dot(ctx_t, W_ctc))

        h_ti = z * h_tm1 + (1. - z) * candidate_h_t
        h_t = m_t[:, None] * h_ti + (1 - m_t)[:, None] * h_tm1
        return h_t, ctx_t, att_w_t.T

    """
    # Single step call
    s0 = [s[0] for s in sequences]
    outs_t = step(*(s0 + outputs + non_sequences))
    """

    (h, context, attention_weights), updates = theano.scan(
        step, name=name + '_cond_gru_recurrent_scan',
        sequences=sequences,
        outputs_info=outputs,
        non_sequences=non_sequences)
    return h, context, attention_weights


def lstm_recurrent_layer(list_of_inputs, mask, hidden_dim, graph, name,
                         random_state, strict=True):
    ndim = [len(calc_expected_dims(graph, inp)) for inp in list_of_inputs]
    check = [n for n in ndim if n != 3]
    if len(check) > 0:
        raise ValueError("Input with ndim != 3 detected!")

    # shape[0] is fake, but shape[1] and shape[2] are fine
    conc_input = concatenate(list_of_inputs, graph, name + "_input",
                             axis=list_of_inputs[0].ndim - 1)
    shape = calc_expected_dims(graph, conc_input)
    h0 = np_zeros((shape[1], hidden_dim))
    c0 = np_zeros((shape[1], hidden_dim))
    list_of_names = [name + '_h0', name + '_c0']
    add_arrays_to_graph([h0, c0], list_of_names, graph)
    h0_sym, c0_sym = fetch_from_graph(list_of_names, graph)

    W_name = name + '_lstm_rec_step_W'
    b_name = name + '_lstm_rec_step_b'
    U_name = name + '_lstm_rec_step_U'
    list_of_names = [W_name, b_name, U_name]
    if not names_in_graph(list_of_names, graph):
        assert random_state is not None
        conc_input_dim = int(sum([calc_expected_dims(graph, inp)[-1]
                                  for inp in list_of_inputs]))
        shape = (conc_input_dim, hidden_dim)
        np_W = np.hstack([np_uniform(shape, random_state),
                          np_uniform(shape, random_state),
                          np_uniform(shape, random_state),
                          np_uniform(shape, random_state)])
        np_b = np_zeros((4 * shape[1],))
        np_U = np.hstack([np_ortho((shape[1], shape[1]), random_state),
                          np_ortho((shape[1], shape[1]), random_state),
                          np_ortho((shape[1], shape[1]), random_state),
                          np_ortho((shape[1], shape[1]), random_state)])
        add_arrays_to_graph([np_W, np_b, np_U], list_of_names, graph,
                            strict=strict)
    else:
        if strict:
            raise AttributeError(
                "Name %s already found in graph with strict mode!" % name)

    W, b, U = fetch_from_graph(list_of_names, graph)
    projected_input = tensor.dot(conc_input, W) + b

    def _slice(arr, n):
        # First slice is tensor_dim - 1 sometimes with scan...
        # need to be *very* careful and test with strict=False and reusing stuff
        # since shape is redefined in if not names_in_graph...
        dim = shape[1]
        if arr.ndim == 3:
            return arr[:, :, n * dim:(n + 1) * dim]
        return arr[:, n * dim:(n + 1) * dim]

    def step(x_t, m_t, h_tm1, c_tm1, U):
        projected_gates = tensor.dot(h_tm1, U) + x_t
        i = tensor.nnet.sigmoid(_slice(projected_gates, 0))
        o = tensor.nnet.sigmoid(_slice(projected_gates, 1))
        f = tensor.nnet.sigmoid(_slice(projected_gates, 2))
        c = tensor.tanh(_slice(projected_gates, 3))
        c_ti = f * c_tm1 + i * c
        c_t = m_t[:, None] * c_ti + (1 - m_t)[:, None] * c_tm1

        h_ti = o * tensor.tanh(c_t)
        h_t = m_t[:, None] * h_ti + (1 - m_t)[:, None] * h_tm1
        return h_t, c_t

    (h, c), updates = theano.scan(step, name=name + '_lstm_recurrent_scan',
                                  sequences=[projected_input, mask],
                                  outputs_info=[h0_sym, c0_sym],
                                  non_sequences=[U])
    return h
