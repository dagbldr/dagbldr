# Author: Kyle Kastner
# License: BSD 3-clause
import numpy as np
from scipy import linalg
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams
from ..utils import concatenate
from ..utils import calc_expected_dims, names_in_graph, add_arrays_to_graph
from ..utils import fetch_from_graph, add_random_to_graph


def np_zeros(shape):
    """ Builds a numpy variable filled with zeros """
    return np.zeros(shape).astype(theano.config.floatX)


def np_rand(shape, random_state):
    # Make sure bounds aren't the same
    return random_state.uniform(low=-0.08, high=0.08, size=shape).astype(
        theano.config.floatX)


def np_randn(shape, random_state):
    """ Builds a numpy variable filled with random normal values """
    return (0.01 * random_state.randn(*shape)).astype(theano.config.floatX)


def np_tanh_fan(shape, random_state):
    # The . after the 6 is critical! shape has dtype int...
    bound = np.sqrt(6. / np.sum(shape))
    return random_state.uniform(low=-bound, high=bound,
                                size=shape).astype(theano.config.floatX)


def np_sigmoid_fan(shape, random_state):
    return 4 * np_tanh_fan(shape, random_state)


def np_ortho(shape, random_state):
    """ Builds a theano variable filled with orthonormal random values """
    g = random_state.randn(*shape)
    o_g = linalg.svd(g)[0]
    return o_g.astype(theano.config.floatX)


def softplus(X):
    return tensor.nnet.softplus(X)


def relu(X):
    return X * (X > 1)


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
    conc_input = concatenate(list_of_inputs, graph, name, axis=-1)
    dropped = dropout(conc_input, random_state, on_off_switch, p=dropout_prob)
    return dropped


def projection_layer(list_of_inputs, graph, name, proj_dim=None,
                     random_state=None, strict=True, init_func=np_tanh_fan,
                     func=linear):
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
    if func is not None:
        final = func(output)
    else:
        final = output
    return final


def linear_layer(list_of_inputs, graph, name, proj_dim=None, random_state=None,
                 strict=True, init_func=np_tanh_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=linear)


def sigmoid_layer(list_of_inputs, graph, name, proj_dim=None, random_state=None,
                  strict=True, init_func=np_sigmoid_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=tensor.nnet.sigmoid)


def tanh_layer(list_of_inputs, graph, name, proj_dim=None, random_state=None,
               strict=True, init_func=np_tanh_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=tensor.tanh)


def softplus_layer(list_of_inputs, graph, name, proj_dim=None,
                   random_state=None, strict=True,
                   init_func=np_tanh_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=softplus)


def exp_layer(list_of_inputs, graph, name, proj_dim=None, random_state=None,
              strict=True, init_func=np_tanh_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=tensor.exp)


def relu_layer(list_of_inputs, graph, name, proj_dim=None, random_state=None,
               strict=True, init_func=np_tanh_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=relu)


def softmax_layer(list_of_inputs, graph, name, proj_dim=None, random_state=None,
                  strict=True, init_func=np_tanh_fan):
    return projection_layer(
        list_of_inputs=list_of_inputs, graph=graph, name=name,
        proj_dim=proj_dim, random_state=random_state,
        strict=strict, init_func=init_func, func=softmax)


def softmax_sample_layer(list_of_multinomial_inputs, graph, name,
                         random_state=None):
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        print("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    theano_rng = MRG_RandomStreams(seed=theano_seed)
    conc_multinomial = concatenate(list_of_multinomial_inputs, graph,
                                   name, axis=1)
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
    conc_mu = concatenate(list_of_mu_inputs, graph, name, axis=1)
    conc_sigma = concatenate(list_of_sigma_inputs, graph, name, axis=1)
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
    conc_mu = concatenate(list_of_mu_inputs, graph, name, axis=1)
    conc_log_sigma = concatenate(list_of_log_sigma_inputs, graph, name, axis=1)
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
    conc_input = concatenate(list_of_inputs, graph, name + "_input", axis=-1)
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
        np_W = np_rand(shape, random_state)
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
    conc_input = concatenate(list_of_inputs, graph, name + "_input", axis=-1)
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
        np_W = np.hstack([np_rand(shape, random_state),
                          np_rand(shape, random_state),
                          np_rand(shape, random_state)])
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
        if arr.ndim < 2:
            return arr[n * dim:(n + 1) * dim]
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


def gru_cond_recurrent_layer(list_of_outputs, hidden_context, output_mask,
                             hidden_dim, graph, name, random_state,
                             strict=True):
    """
    Feed list_of_outputs as unshifted outputs desired. Internally the layer
    will shift all of them so that everything is next step prediction.

    hidden_context is the hidden states from the encoder,
    in this case only useful to get the last hidden state.
    """
    # an easy interface to gru conditional recurrent nets
    # If the expressions are not the same length and batch size it won't work
    max_ndim = max([out.ndim for out in list_of_outputs])
    if max_ndim > 3:
        raise ValueError("Input with ndim > 3 detected!")

    conc_output = concatenate(list_of_outputs, graph, name + "_gru_cond_step",
                              axis=-1)
    context = hidden_context[-1]
    # Decoder initializes hidden state with tanh projection of last hidden
    # context representing p(X_1...X_t)
    h0_sym = tanh_layer([context], graph, name + '_h0_proj',
                        proj_dim=hidden_dim, random_state=random_state)

    shifted = tensor.zeros_like(conc_output)
    shifted = tensor.set_subtensor(shifted[1:], conc_output[:-1])
    input_shifted = shifted

    W_name = name + '_gru_cond_rec_step_W'
    b_name = name + '_gru_cond_rec_step_b'
    Urz_name = name + '_gru_cond_rec_step_Urz'
    U_name = name + '_gru_cond_rec_step_U'
    Wg_name = name + '_gru_cond_rec_step_Wg'
    bg_name = name + '_gru_cond_rec_step_bg'
    Wh_name = name + '_gru_cond_rec_step_Wh'
    bh_name = name + '_gru_cond_rec_step_bh'
    list_of_names = [W_name, b_name, Urz_name, U_name, Wg_name, bg_name,
                     Wh_name, bh_name]
    if not names_in_graph(list_of_names, graph):
        assert random_state is not None
        conc_input_dim = calc_expected_dims(graph, input_shifted)[-1]
        shape = (conc_input_dim, hidden_dim)
        np_W = np.hstack([np_rand(shape, random_state),
                          np_rand(shape, random_state),
                          np_rand(shape, random_state)])
        np_b = np_zeros((3 * shape[1],))
        np_Urz = np.hstack([np_ortho((shape[1], shape[1]), random_state),
                            np_ortho((shape[1], shape[1]), random_state), ])
        np_U = np_ortho((shape[1], shape[1]), random_state)
        context_dim = calc_expected_dims(graph, context)[-1]
        np_Wg = np_rand((context_dim, 2 * shape[1]), random_state)
        np_bg = np_zeros((2 * shape[1],))
        np_Wh = np_rand((context_dim, shape[1]), random_state)
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
        if arr.ndim < 2:
            return arr[n * dim:(n + 1) * dim]
        return arr[:, n * dim:(n + 1) * dim]

    def step(x_t, m_t, h_tm1, U, pcg, pch):
        projected_gates = tensor.dot(h_tm1, Urz) + pcg
        r = tensor.nnet.sigmoid(_slice(x_t, 0) + _slice(projected_gates, 0))
        z = tensor.nnet.sigmoid(_slice(x_t, 1) + _slice(projected_gates, 1))
        candidate_h_t = tensor.tanh(_slice(x_t, 2) + tensor.dot(
            r * h_tm1, U) + pch)
        h_ti = z * h_tm1 + (1. - z) * candidate_h_t
        h_t = m_t[:, None] * h_ti + (1 - m_t)[:, None] * h_tm1
        return h_t

    h, updates = theano.scan(step, name=name + '_gru_cond_recurrent_scan',
                             sequences=[projected_input, output_mask],
                             outputs_info=[h0_sym],
                             non_sequences=[U, projected_context_to_gates,
                                            projected_context_to_hidden])
    return h


def lstm_recurrent_layer(list_of_inputs, mask, hidden_dim, graph, name,
                         random_state, strict=True):
    ndim = [len(calc_expected_dims(graph, inp)) for inp in list_of_inputs]
    check = [n for n in ndim if n != 3]
    if len(check) > 0:
        raise ValueError("Input with ndim != 3 detected!")

    # shape[0] is fake, but shape[1] and shape[2] are fine
    conc_input = concatenate(list_of_inputs, graph, name + "_input", axis=-1)
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
        np_W = np.hstack([np_rand(shape, random_state),
                          np_rand(shape, random_state),
                          np_rand(shape, random_state),
                          np_rand(shape, random_state)])
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
        if arr.ndim < 2:
            return arr[n * dim:(n + 1) * dim]
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
