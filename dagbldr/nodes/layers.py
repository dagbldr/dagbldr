# Author: Kyle Kastner
# License: BSD 3-clause
import numpy as np
from scipy import linalg
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams
from ..utils import as_shared, tag_expression, concatenate, expression_shape
from ..utils import calc_expected_dim, names_in_graph, add_arrays_to_graph
from ..utils import fetch_from_graph


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
    return tensor.nnet.softplus(X) + 1E-4


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


def dropout_layer(list_of_inputs, name, on_off_switch, dropout_prob=0.5,
                  random_state=None):
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        print("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    conc_input = concatenate(list_of_inputs, name, axis=-1)
    shape = expression_shape(conc_input)
    dropped = dropout(conc_input, random_state, on_off_switch, p=dropout_prob)
    tag_expression(dropped, name, shape)
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
        conc_input_dim = int(sum([calc_expected_dim(graph, inp)
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
    conc_input = concatenate(list_of_inputs, graph, name, axis=-1)
    output = tensor.dot(conc_input, W) + b
    if func is not None:
        final = func(output)
    else:
        final = output
    # Commenting out, remove when writing tests
    # shape = list(expression_shape(conc_input))
    # Projection is on last axis
    # shape[-1] = proj_dim
    # new_shape = tuple(shape)
    # tag_expression(final, name, new_shape)
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


def softmax_sample_layer(list_of_multinomial_inputs, name, random_state=None):
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        print("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    theano_rng = MRG_RandomStreams(seed=theano_seed)
    conc_multinomial = concatenate(list_of_multinomial_inputs, name, axis=1)
    shape = expression_shape(conc_multinomial)
    conc_multinomial /= len(list_of_multinomial_inputs)
    tag_expression(conc_multinomial, name, shape)
    samp = theano_rng.multinomial(pvals=conc_multinomial,
                                  dtype="int32")
    tag_expression(samp, name, (shape[0], shape[1]))
    return samp


def gaussian_sample_layer(list_of_mu_inputs, list_of_sigma_inputs,
                          name, random_state=None):
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        print("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    theano_rng = MRG_RandomStreams(seed=theano_seed)
    conc_mu = concatenate(list_of_mu_inputs, name, axis=1)
    conc_sigma = concatenate(list_of_sigma_inputs, name, axis=1)
    e = theano_rng.normal(size=(conc_sigma.shape[0],
                                conc_sigma.shape[1]),
                          dtype=conc_sigma.dtype)
    samp = conc_mu + conc_sigma * e
    shape = expression_shape(conc_sigma)
    tag_expression(samp, name, shape)
    return samp


def gaussian_log_sample_layer(list_of_mu_inputs, list_of_log_sigma_inputs,
                              name, random_state=None):
    """ log_sigma_inputs should be from a linear_layer """
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        print("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    theano_rng = MRG_RandomStreams(seed=theano_seed)
    conc_mu = concatenate(list_of_mu_inputs, name, axis=1)
    conc_log_sigma = concatenate(list_of_log_sigma_inputs, name, axis=1)
    e = theano_rng.normal(size=(conc_log_sigma.shape[0],
                                conc_log_sigma.shape[1]),
                          dtype=conc_log_sigma.dtype)

    samp = conc_mu + tensor.exp(0.5 * conc_log_sigma) * e
    shape = expression_shape(conc_log_sigma)
    tag_expression(samp, name, shape)
    return samp


def switch_wrap(switch_func, if_true_var, if_false_var, name):
    switched = tensor.switch(switch_func, if_true_var, if_false_var)
    shape = expression_shape(if_true_var)
    assert shape == expression_shape(if_false_var)
    tag_expression(switched, name, shape)
    return switched


def rnn_scan_wrap(func, sequences=None, outputs_info=None, non_sequences=None,
                  n_steps=None, truncate_gradient=-1, go_backwards=False,
                  mode=None,
                  name=None, profile=False, allow_gc=None, strict=False):
    """ Expects 3D input sequences, dim 0 being the axis of iteration """
    # assumes 0th output of func is hidden state
    # necessary so that values out of scan can be tagged... ugh
    # shape_of_variables eliminates the need for this

    s = expression_shape(sequences[0])
    shape_0 = s[0]
    for n, s in enumerate(sequences):
        s = expression_shape(sequences[n])
        # all sequences should be the same length
        tag_expression(sequences[n], name + "_%s_" % n, s[1:])

    outputs, updates = theano.scan(func, sequences=sequences,
                                   outputs_info=outputs_info,
                                   non_sequences=non_sequences, n_steps=n_steps,
                                   truncate_gradient=truncate_gradient,
                                   go_backwards=go_backwards, mode=mode,
                                   name=name, profile=profile,
                                   allow_gc=allow_gc, strict=strict)
    if type(outputs) is list:
        for n, o in enumerate(outputs):
            s = expression_shape(outputs_info[n])
            # all sequences should be the same length
            shape_1 = s
            shape = (shape_0,) + shape_1
            tag_expression(outputs[n], name + "_%s_" % n, shape)
    else:
        s = expression_shape(outputs_info[0])
        shape_1 = s
        # combine tuples
        shape = (shape_0,) + shape_1
        tag_expression(outputs, name, shape)
    return outputs, updates


def tanh_recurrent_layer(list_of_inputs, list_of_hiddens, graph, name,
                         random_state=None, strict=True):
    # All inputs are assumed 2D as are hiddens
    # Everything is dictated by the size of the hiddens
    W_name = name + '_tanhrec_W'
    b_name = name + '_tanhrec_b'
    U_name = name + '_tanhrec_U'
    list_of_names = [W_name, b_name, U_name]
    if not names_in_graph(list_of_names, graph):
        assert random_state is not None
        conc_input_dim = int(sum([calc_expected_dim(inp)
                                  for inp in list_of_inputs]))
        conc_hidden_dim = int(sum([calc_expected_dim(hid)
                                   for hid in list_of_hiddens]))
        shape = (conc_input_dim, conc_hidden_dim)
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
    # per timestep
    conc_input = concatenate(list_of_inputs, name + "_input", axis=-1)
    conc_hidden = concatenate(list_of_hiddens, name + "_hidden", axis=-1)
    output = tensor.tanh(tensor.dot(conc_input, W) + b +
                         tensor.dot(conc_hidden, U))
    # remember this is number of dims per timestep!
    shape = expression_shape(conc_input)
    tag_expression(output, name, shape)
    return output


def easy_tanh_recurrent(list_of_inputs, mask, hidden_dim, graph, name,
                        random_state,
                        one_step=False):
    # an easy interface to lstm recurrent nets
    shape = expression_shape(list_of_inputs[0])
    # If the expressions are not the same length and batch size it won't work
    max_ndim = max([inp.ndim for inp in list_of_inputs])
    if max_ndim > 3:
        raise ValueError("Input with ndim > 3 detected!")
    elif max_ndim == 2:
        # Simulate batch size 1
        shape = (shape[0], 1, shape[1])

    # an easy interface to tanh recurrent nets
    h0 = np_zeros((shape[1], hidden_dim))
    h0_sym = as_shared(h0, name)
    tag_expression(h0_sym, name, (shape[1], hidden_dim))

    def step(x_t, m_t, h_tm1):
        h_ti = tanh_recurrent_layer([x_t], [h_tm1], graph,
                                    name + '_easy_tanh_rec', random_state)
        h_t = m_t[:, None] * h_ti + (1 - m_t)[:, None] * h_tm1
        return h_t

    if one_step:
        conc_input = concatenate(list_of_inputs, name + "_easy_tanh_step",
                                 axis=-1)
        shape = expression_shape(conc_input)
        sliced = conc_input[0]
        tag_expression(sliced, name, shape[1:])
        shape = expression_shape(mask)
        mask_sliced = mask[0]
        tag_expression(mask_sliced, name + "_mask", shape[1:])
        h = step(sliced, h0_sym, mask_sliced)
        shape = expression_shape(sliced)
        tag_expression(h, name, shape)
    else:
        # the hidden state `h` for the entire sequence
        h, updates = rnn_scan_wrap(step, name=name + '_easy_tanh_scan',
                                   sequences=list_of_inputs + [mask],
                                   outputs_info=[h0_sym])
    return h


def gru_recurrent_layer(list_of_inputs, list_of_hiddens, graph, name,
                        random_state=None, strict=True):
    W_name = name + '_grurec_W'
    b_name = name + '_grurec_b'
    U_name = name + '_grurec_U'
    list_of_names = [W_name, b_name, U_name]
    if not names_in_graph(list_of_names, graph):
        assert random_state is not None
        conc_input_dim = int(sum([calc_expected_dim(inp)
                                  for inp in list_of_inputs]))
        conc_hidden_dim = int(sum([calc_expected_dim(hid)
                                   for hid in list_of_hiddens]))
        shape = (conc_input_dim, conc_hidden_dim)
        np_W = np.hstack([np_rand(shape, random_state),
                          np_rand(shape, random_state),
                          np_rand(shape, random_state)])
        np_b = np_zeros((3 * shape[1],))
        np_U = np.hstack([np_ortho((shape[1], shape[1]), random_state),
                          np_ortho((shape[1], shape[1]), random_state),
                          np_ortho((shape[1], shape[1]), random_state)])
        add_arrays_to_graph([np_W, np_b, np_U], list_of_names, graph,
                            strict=strict)
    else:
        if strict:
            raise AttributeError(
                "Name %s already found in graph with strict mode!" % name)

    def _slice(arr, n):
        # First slice is tensor_dim - 1 sometimes with scan...
        dim = shape[1]
        if arr.ndim < 2:
            return arr[n * dim:(n + 1) * dim]
        return arr[:, n * dim:(n + 1) * dim]
    W, b, U = fetch_from_graph(list_of_names, graph)
    conc_input = concatenate(list_of_inputs, name + "_input", axis=0)
    conc_hidden = concatenate(list_of_hiddens, name + "_hidden", axis=0)
    proj_i = tensor.dot(conc_input, W) + b
    proj_h = tensor.dot(conc_hidden, U)
    r = tensor.nnet.sigmoid(_slice(proj_i, 1)
                            + _slice(proj_h, 1))
    z = tensor.nnet.sigmoid(_slice(proj_i, 2)
                            + _slice(proj_h, 2))
    candidate_h = tensor.tanh(_slice(proj_i, 0) + r * _slice(proj_h, 0))
    output = z * conc_hidden + (1. - z) * candidate_h
    # remember this is number of dims per timestep!
    tag_expression(output, name, (shape[1],))
    return output


def easy_gru_recurrent(list_of_inputs, mask, hidden_dim, graph, name,
                       random_state, one_step=False):
    # an easy interface to lstm recurrent nets
    shape = expression_shape(list_of_inputs[0])
    # If the expressions are not the same length and batch size it won't work
    max_ndim = max([inp.ndim for inp in list_of_inputs])
    if max_ndim > 3:
        raise ValueError("Input with ndim > 3 detected!")
    elif max_ndim == 2:
        # Simulate batch size 1
        shape = (shape[0], 1, shape[1])

    # an easy interface to tanh recurrent nets
    h0 = np_zeros((shape[1], hidden_dim))
    h0_sym = as_shared(h0, name)
    tag_expression(h0_sym, name, (shape[1], hidden_dim))

    def step(x_t, m_t, h_tm1):
        h_ti = gru_recurrent_layer([x_t], [h_tm1], graph,
                                   name + '_easy_gru_rec', random_state)
        h_t = m_t[:, None] * h_ti + (1 - m_t)[:, None] * h_tm1
        return h_t

    if one_step:
        conc_input = concatenate(list_of_inputs, name + "_easy_gru_step",
                                 axis=-1)
        shape = expression_shape(conc_input)
        sliced = conc_input[0]
        tag_expression(sliced, name, shape[1:])
        shape = expression_shape(mask)
        mask_sliced = mask[0]
        tag_expression(mask_sliced, name + "_mask", shape[1:])
        h = step(sliced, h0_sym, mask_sliced)
        shape = expression_shape(sliced)
        tag_expression(h, name, shape)
    else:
        # the hidden state `h` for the entire sequence
        h, updates = rnn_scan_wrap(step, name=name + '_easy_gru_scan',
                                   sequences=list_of_inputs + [mask],
                                   outputs_info=[h0_sym])
    return h


def easy_gru_cond_recurrent(list_of_inputs, list_of_hiddens, mask, hidden_dim,
                            graph, name, random_state, one_step=False):
    """
    Feed list_of_inputs as unshifted outputs desired. Internally the layer
    will shift all inputs so that everything is next step prediction.

    list_of_hiddens is a list of all the hidden states from the encoders,
    in this case only useful to get the last hidden state.
    """
    # an easy interface to gru conditional recurrent nets
    shape = expression_shape(list_of_inputs[0])
    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    # If the expressions are not the same length and batch size it won't work
    max_ndim = max([inp.ndim for inp in list_of_inputs])
    if max_ndim > 3:
        raise ValueError("Input with ndim > 3 detected!")
    elif max_ndim == 2:
        # Simulate batch size 1
        shape = (shape[0], 1, shape[1])

    # Can check length for bidirectionality?
    conc_hiddens = concatenate(list_of_hiddens,
                               name + "easy_gru_step_context", axis=-1)

    shape = expression_shape(conc_hiddens)
    context = conc_hiddens[-1]
    tag_expression(context, name + "_context", shape[1:])
    # Decoder initializes hidden state with projection of last from encode
    h0_sym = tanh_layer([context], graph, name + '_h0_proj',
                        proj_dim=hidden_dim, random_state=random_state)

    """
    # an easy interface to gru recurrent nets
    h0 = np_zeros((shape[1], hidden_dim))
    h0_sym = as_shared(h0, name)
    tag_expression(h0_sym, name, (shape[1], hidden_dim))
    """

    conc_input = concatenate(list_of_inputs, name + "_easy_gru_step", axis=-1)
    shape = expression_shape(conc_input)
    shifted = tensor.zeros_like(conc_input)
    shifted = tensor.set_subtensor(shifted[1:], conc_input[:-1])
    tag_expression(shifted, name + '_shifted_gru_step', shape)

    def step(y_t, m_t, h_tm1):
        import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
        h_ti = gru_recurrent_layer([y_t, context], [h_tm1], graph,
                                   name + '_easy_gru_rec', random_state)
        h_t = m_t[:, None] * h_ti + (1 - m_t)[:, None] * h_tm1
        return h_t

    if one_step:
        shape = expression_shape(shifted)
        sliced = shifted[0]
        tag_expression(sliced, name, shape[1:])
        shape = expression_shape(mask)
        mask_sliced = mask[0]
        tag_expression(mask_sliced, name + "_mask", shape[1:])
        h = step(sliced, h0_sym, mask_sliced)
        shape = expression_shape(sliced)
        tag_expression(h, name, shape)
    else:
        # the hidden state `h` for the entire sequence
        h, updates = rnn_scan_wrap(step, name=name + '_easy_gru_scan',
                                   sequences=[shifted, mask],
                                   outputs_info=[h0_sym])
    return h


def lstm_recurrent_layer(list_of_inputs, list_of_hiddens, list_of_cells,
                         graph, name, random_state=None, strict=True):
    W_name = name + '_lstmrec_W'
    b_name = name + '_lstmrec_b'
    U_name = name + '_lstmrec_U'
    list_of_names = [W_name, b_name, U_name]
    if not names_in_graph(list_of_names, graph):
        assert random_state is not None
        conc_input_dim = int(sum([calc_expected_dim(inp)
                                  for inp in list_of_inputs]))
        conc_hidden_dim = int(sum([calc_expected_dim(hid)
                                   for hid in list_of_hiddens]))
        conc_cell_dim = int(sum([calc_expected_dim(hid)
                                 for hid in list_of_cells]))
        assert conc_hidden_dim == conc_cell_dim
        shape = (conc_input_dim, conc_hidden_dim)
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

    def _slice(arr, n):
        # First slice is tensor_dim - 1 sometimes with scan...
        dim = shape[1]
        if arr.ndim < 2:
            return arr[n * dim:(n + 1) * dim]
        return arr[:, n * dim:(n + 1) * dim]
    W, b, U = fetch_from_graph(list_of_names, graph)
    conc_input = concatenate(list_of_inputs, name + "_input", axis=0)
    conc_hidden = concatenate(list_of_hiddens, name + "_hidden", axis=0)
    conc_cell = concatenate(list_of_cells, name + "_cell", axis=0)
    proj_i = tensor.dot(conc_input, W) + b
    proj_h = tensor.dot(conc_hidden, U)
    # input output forget and cell gates
    ig = tensor.nnet.sigmoid(_slice(proj_i, 0) + _slice(proj_h, 0))
    fg = tensor.nnet.sigmoid(_slice(proj_i, 1) + _slice(proj_h, 1))
    og = tensor.nnet.sigmoid(_slice(proj_i, 2) + _slice(proj_h, 2))
    cg = tensor.tanh(_slice(proj_i, 3) + _slice(proj_h, 3))
    c = fg * conc_cell + ig * cg
    h = og * tensor.tanh(c)
    tag_expression(h, name + "_hidden", (shape[1],))
    tag_expression(c, name + "_cell", (shape[1],))
    return h, c


def easy_lstm_recurrent(list_of_inputs, mask, hidden_dim, graph, name,
                        random_state, one_step=False):
    # an easy interface to lstm recurrent nets
    shape = expression_shape(list_of_inputs[0])
    # If the expressions are not the same length and batch size it won't work
    max_ndim = max([inp.ndim for inp in list_of_inputs])
    if max_ndim > 3:
        raise ValueError("Input with ndim > 3 detected!")
    elif max_ndim == 2:
        # Simulate batch size 1
        shape = (shape[0], 1, shape[1])

    # an easy interface to tanh recurrent nets
    h0 = np_zeros((shape[1], hidden_dim))
    h0_sym = as_shared(h0, name)
    tag_expression(h0_sym, name, (shape[1], hidden_dim))

    c0 = np_zeros((shape[1], hidden_dim))
    c0_sym = as_shared(c0, name)
    tag_expression(c0_sym, name, (shape[1], hidden_dim))

    def step(x_t, m_t, h_tm1, c_tm1):
        h_ti, c_ti = lstm_recurrent_layer([x_t], [h_tm1], [c_tm1], graph,
                                          name + '_easy_lstm_rec', random_state)
        h_t = m_t[:, None] * h_ti + (1 - m_t)[:, None] * h_tm1
        c_t = m_t[:, None] * c_ti + (1 - m_t)[:, None] * c_tm1
        return h_t, c_t

    if one_step:
        conc_input = concatenate(list_of_inputs, name + "_easy_lstm_step",
                                 axis=-1)
        shape = expression_shape(conc_input)
        sliced = conc_input[0]
        tag_expression(sliced, name, shape[1:])
        shape = expression_shape(mask)
        mask_sliced = mask[0]
        tag_expression(mask_sliced, name + "_mask", shape[1:])
        h, c = step(sliced, h0_sym, c0_sym, mask_sliced)
        shape = expression_shape(sliced)
        tag_expression(h, name, shape)
    else:
        # the hidden state `h` for the entire sequence
        [h, c], updates = rnn_scan_wrap(step, name=name + '_easy_lstm_scan',
                                        sequences=list_of_inputs + [mask],
                                        outputs_info=[h0_sym, c0_sym])
    return h
