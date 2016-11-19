# Author: Kyle Kastner
# License: BSD 3-clause
from theano import tensor
import theano
import numpy as np

from ..utils import concatenate, as_shared
from ..core import get_logger, get_type, set_shared
from .nodes import projection
from .nodes import np_tanh_fan_uniform
from .nodes import np_variance_scaled_uniform
from .nodes import np_normal
from .nodes import np_zeros
from .nodes import np_ortho
from .nodes import get_name
from .nodes import linear_activation

logger = get_logger()
_type = get_type()


def simple_fork(list_of_inputs, list_of_input_dims, proj_dim, name=None,
                batch_normalize=False, mode_switch=None,
                random_state=None, strict=True, init_func=np_tanh_fan_uniform):
    if name is None:
        name = get_name()
    else:
        name = name + "_simple_fork"
    ret = projection(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        proj_dim=proj_dim, name=name, batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        strict=strict, init_func=init_func, act_func=linear_activation)
    return ret


def simple(step_input, previous_hidden, hidden_dim, mask=None,
           name=None, random_state=None, strict=True, init_func=np_ortho):
    """
    hidden_dim 1x
    """
    if name is None:
        name = get_name()
    W_name = name + "_simple_recurrent_W"

    if mask is not None:
        raise ValueError("Ragged minibatch processing not yet implemented")

    try:
        W = get_shared(W_name)
        if strict:
            raise AttributeError(
                "Name %s already found in parameters, strict mode!" % name)
    except NameError:
        assert random_state is not None
        np_W = init_func((hidden_dim, hidden_dim), random_state)
        W = as_shared(np_W)
        set_shared(W_name, W)
    return tensor.tanh(step_input + tensor.dot(previous_hidden, W))


def gru_weights(input_dim, hidden_dim, forward_init=None, hidden_init="normal",
                random_state=None):
    if random_state is None:
        raise ValueError("Must pass random_state!")
    shape = (input_dim, hidden_dim)
    if forward_init == "normal":
        W = np.hstack([np_normal(shape, random_state),
                       np_normal(shape, random_state),
                       np_normal(shape, random_state)])
    elif forward_init == "fan":
        W = np.hstack([np_tanh_fan_normal(shape, random_state),
                       np_tanh_fan_normal(shape, random_state),
                       np_tanh_fan_normal(shape, random_state)])
    elif forward_init is None:
        if input_dim == hidden_dim:
            W = np.hstack([np_ortho(shape, random_state),
                           np_ortho(shape, random_state),
                           np_ortho(shape, random_state)])
        else:
            # lecun
            W = np.hstack([np_variance_scaled_uniform(shape, random_state),
                           np_variance_scaled_uniform(shape, random_state),
                           np_variance_scaled_uniform(shape, random_state)])
    else:
        raise ValueError("Unknown forward init type %s" % forward_init)
    b = np_zeros((3 * shape[1],))

    if hidden_init == "normal":
        Wur = np.hstack([np_normal((shape[1], shape[1]), random_state),
                         np_normal((shape[1], shape[1]), random_state), ])
        U = np_normal((shape[1], shape[1]), random_state)
    elif hidden_init == "ortho":
        Wur = np.hstack([np_ortho((shape[1], shape[1]), random_state),
                         np_ortho((shape[1], shape[1]), random_state), ])
        U = np_ortho((shape[1], shape[1]), random_state)
    return W, b, Wur, U


def lstm_weights(input_dim, hidden_dim, forward_init=None, hidden_init="normal",
                 random_state=None):
    if random_state is None:
        raise ValueError("Must pass random_state!")
    shape = (input_dim, hidden_dim)
    if forward_init == "normal":
        W = np.hstack([np_normal(shape, random_state),
                       np_normal(shape, random_state),
                       np_normal(shape, random_state),
                       np_normal(shape, random_state)])
    elif forward_init == "fan":
        W = np.hstack([np_tanh_fan_normal(shape, random_state),
                       np_tanh_fan_normal(shape, random_state),
                       np_tanh_fan_normal(shape, random_state),
                       np_tanh_fan_normal(shape, random_state)])
    elif forward_init is None:
        if input_dim == hidden_dim:
            W = np.hstack([np_ortho(shape, random_state),
                           np_ortho(shape, random_state),
                           np_ortho(shape, random_state),
                           np_ortho(shape, random_state)])
        else:
            # lecun
            W = np.hstack([np_variance_scaled_uniform(shape, random_state),
                           np_variance_scaled_uniform(shape, random_state),
                           np_variance_scaled_uniform(shape, random_state),
                           np_variance_scaled_uniform(shape, random_state)])
    else:
        raise ValueError("Unknown forward init type %s" % forward_init)
    b = np_zeros((4 * shape[1],))
    # Set forget gate bias to 1
    b[shape[1]:2 * shape[1]] += 1.

    if hidden_init == "normal":
        U = np.hstack([np_normal((shape[1], shape[1]), random_state),
                       np_normal((shape[1], shape[1]), random_state),
                       np_normal((shape[1], shape[1]), random_state),
                       np_normal((shape[1], shape[1]), random_state), ])
    elif hidden_init == "ortho":
        U = np.hstack([np_ortho((shape[1], shape[1]), random_state),
                       np_ortho((shape[1], shape[1]), random_state),
                       np_ortho((shape[1], shape[1]), random_state),
                       np_ortho((shape[1], shape[1]), random_state), ])
    return W, b, U


def lstm_fork(list_of_inputs, list_of_input_dims, proj_dim, name=None,
              batch_normalize=False, mode_switch=None,
              random_state=None, strict=True, init_func=np_tanh_fan_uniform):
    if name is None:
        name = get_name()
    else:
        name = name + "_lstm_fork"
    inp_d = np.sum(list_of_input_dims)
    W, b, U = lstm_weights(inp_d, proj_dim, random_state=random_state)

    ret = projection(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        proj_dim=proj_dim, name=name, batch_normalize=batch_normalize,
        mode_switch=mode_switch, random_state=random_state,
        init_weights=W, init_biases=b,
        strict=strict, init_func=init_func, act_func=linear_activation)
    return ret


def lstm(step_input, previous_state, list_of_input_dims, hidden_dim,
         name=None, random_state=None, strict=True, init_func=np_ortho):
    """
    hidden_dim is really 2x hidden_dim
    """
    if name is None:
        name = get_name()
    U_name = name + "_lstm_recurrent_U"
    input_dim = np.sum(list_of_input_dims)
    _, _, np_U = lstm_weights(input_dim, hidden_dim, random_state=random_state)
    U = as_shared(np_U)
    set_shared(U_name, U)

    dim = hidden_dim
    def _s(p, d):
        return p[:, d * dim:(d+1) * dim]

    previous_cell = _s(previous_state, 1)
    previous_st = _s(previous_state, 0)

    preactivation = theano.dot(previous_st, U) + step_input
    sigm = tensor.nnet.sigmoid
    tanh = tensor.tanh
    ig = sigm(_s(preactivation, 0))
    fg = sigm(_s(preactivation, 1))
    og = sigm(_s(preactivation, 2))
    cg = tanh(_s(preactivation, 3))

    cg = fg * previous_cell + ig * cg
    #cg = mask * cg + (1. - mask) * previous_cell

    hg = og * tanh(cg)
    #hg = mask * hg + (1. - mask) * previous_st

    next_state = concatenate([hg, cg], axis=1)
    return next_state


def slice_state(arr, hidden_dim):
    """
    Used to slice the final state of GRU, LSTM to the suitable part
    """
    part = 0
    if arr.ndim == 2:
        return arr[:, part * hidden_dim:(part + 1) * hidden_dim]
    elif arr.ndim == 3:
        return arr[:, :, part * hidden_dim:(part + 1) * hidden_dim]
    else:
        raise ValueError("Unknown dim")


'''
TODO:
def GRU(inp, gate_inp, previous_state, input_dim, hidden_dim, random_state,
        mask=None, name=None, init=None, scale="default", weight_norm=None,
        biases=False):
        if name is not None:
            raise ValueError("Unhandled parameter sharing in GRU")
        if init is None:
            hidden_init = "ortho"
        elif init == "normal":
            hidden_init = "normal"
        else:
            raise ValueError("Not yet configured for other inits")

        ndi = ndim(inp)
        if mask is None:
            if ndi == 2:
                mask = tf.ones_like(inp)
            else:
                raise ValueError("Unhandled ndim")

        ndm = ndim(mask)
        if ndm == (ndi - 1):
            mask = tf.expand_dims(mask, ndm - 1)

        _, _, Wur, U = gru_weights(input_dim, hidden_dim,
                                   hidden_init=hidden_init,
                                   random_state=random_state)
        dim = hidden_dim
        f1 = Linear([previous_state], [2 * hidden_dim], 2 * hidden_dim,
                    random_state, name=(name, "update/reset"), init=[Wur],
                    biases=biases, weight_norm=weight_norm)
        gates = sigmoid(f1 + gate_inp)
        update = gates[:, :dim]
        reset = gates[:, dim:]
        state_reset = previous_state * reset
        f2 = Linear([state_reset], [hidden_dim], hidden_dim,
                    random_state, name=(name, "state"), init=[U], biases=biases,
                    weight_norm=weight_norm)
        next_state = tf.tanh(f2 + inp)
        next_state = next_state * update + previous_state * (1. - update)
        next_state = mask * next_state + (1. - mask) * previous_state
        return next_state


def GRUFork(list_of_inputs, input_dims, output_dim, random_state, name=None,
            init=None, scale="default", weight_norm=None, biases=True):
        if name is not None:
            raise ValueError("Unhandled parameter sharing in GRUFork")
        gates = Linear(list_of_inputs, input_dims, 3 * output_dim,
                       random_state=random_state,
                       name=(name, "gates"), init=init, scale=scale,
                       weight_norm=weight_norm, biases=biases)
        dim = output_dim
        nd = ndim(gates)
        if nd == 2:
            d = gates[:, :dim]
            g = gates[:, dim:]
        elif nd == 3:
            d = gates[:, :, :dim]
            g = gates[:, :, dim:]
        else:
            raise ValueError("Unsupported ndim")
        return d, g
'''
