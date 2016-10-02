# Author: Kyle Kastner
# License: BSD 3-clause
from theano import tensor
from ..utils import concatenate, as_shared
from ..core import get_logger, get_type, set_shared
from .nodes import projection
from .nodes import np_tanh_fan_uniform
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


def simple(step_input, previous_hidden, hidden_dim, name=None,
           random_state=None, strict=True, init_func=np_ortho):
    if name is None:
        name = get_name()
    W_name = name + "_simple_recurrent_W"

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
