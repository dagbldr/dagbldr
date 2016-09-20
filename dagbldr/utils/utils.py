# Author: Kyle Kastner
# License: BSD 3-clause
import re
import numpy as np
import theano
from theano import tensor
from collections import OrderedDict

from ..core import safe_zip
from ..core import get_type
from ..core import get_file_matches
from ..core import get_checkpoint_dir
from ..core import unpickle
from ..core import set_shared_variables_in_function
from ..core import get_lib_shared_params

_type = get_type()


def get_weights(accept_regex="_W", skip_regex="_softmax_"):
    """
    A regex matcher to get weights. To bypass, simply pass None.

    Returns dictionary of {name: param}
    """
    d = get_lib_shared_params()
    if accept_regex is not None:
        ma = re.compile(accept_regex)
    else:
        ma = None
    if skip_regex is not None:
        sk = re.compile(skip_regex)
    else:
        sk = None
    matched_keys = []
    for k in d.keys():
        if ma is not None:
            if ma.search(k):
                if sk is not None:
                    if not sk.search(k):
                        matched_keys.append(k)
                else:
                    matched_keys.append(k)
    matched_weights = OrderedDict()
    for mk in matched_keys:
        matched_weights[mk] = d[mk]
    return matched_weights


def as_shared(arr, **kwargs):
    return tensor.cast(theano.shared(np.cast[_type](arr)), _type)


def concatenate(tensor_list, name, axis=0):
    """
    Wrapper to `theano.tensor.concatenate`, that casts everything to float32!
    """
    out = tensor.cast(tensor.concatenate(tensor_list, axis=axis),
                      dtype=_type)
    # Temporarily commenting out - remove when writing tests
    # conc_dim = int(sum([calc_expected_dim(graph, inp)
    #                for inp in tensor_list]))
    # This may be hosed... need to figure out how to generalize
    # shape = list(expression_shape(tensor_list[0]))
    # shape[axis] = conc_dim
    # new_shape = tuple(shape)
    # tag_expression(out, name, new_shape)
    return out


def interpolate_between_points(arr, n_steps=50):
    """ Helper function for drawing line between points in space """
    assert len(arr) > 2
    assert n_steps > 1
    path = [path_between_points(start, stop, n_steps=n_steps)
            for start, stop in safe_zip(arr[:-1], arr[1:])]
    path = np.vstack(path)
    return path


def path_between_points(start, stop, n_steps=100, dtype=theano.config.floatX):
    """ Helper function for making a line between points in ND space """
    assert n_steps > 1
    step_vector = 1. / (n_steps - 1) * (stop - start)
    steps = np.arange(0, n_steps)[:, None] * np.ones((n_steps, len(stop)))
    steps = steps * step_vector + start
    return steps.astype(dtype)


def create_checkpoint_dict(lcls):
    """
    Create checkpoint dict that contains all local theano functions

    Example usage:
        create_checkpoint_dict(locals())

    Parameters
    ----------
    lcls : dict
        A dictionary containing theano.function instances, normally the
        result of locals()

    Returns
    -------
    checkpoint_dict : dict
        A checkpoint dictionary suitable for passing to a training loop

    """
    print("Creating new checkpoint dictionary")
    checkpoint_dict = {}
    for k, v in lcls.items():
        if isinstance(v, theano.compile.function_module.Function):
            checkpoint_dict[k] = v
    if len(checkpoint_dict.keys()) == 0:
        raise ValueError("No theano functions in lcls!")
    return checkpoint_dict


def create_or_continue_from_checkpoint_dict(lcls, append_name="best"):
    """
    Create or load a checkpoint dict that contains all local theano functions

    Example usage:
        create_or_load_checkpoint_dict(locals(), append_name="best")

    Parameters
    ----------
    lcls : dict, default locals()
        A dictionary containing theano.function instances, normally the
        result of locals()

    append_name : string, default "best"
        The append name to use for the checkpoint

    Returns
    -------
    checkpoint_dict : dict
        A checkpoint dictionary suitable for passing to a training loop

    """
    sorted_paths = get_file_matches("*.npz", append_name)
    if len(sorted_paths) < 1:
        print("No saved results found in %s, creating!" % get_checkpoint_dir())
        return create_checkpoint_dict(lcls)

    last_weights_path = sorted_paths[-1]
    print("Loading in weights from %s" % last_weights_path)
    last_weights = np.load(last_weights_path)

    checkpoint_dict = {}

    sorted_paths = get_file_matches("*_results_*.pkl", append_name)
    last_results_path = sorted_paths[-1]
    print("Loading in results from %s" % last_results_path)

    checkpoint_dict["previous_results"] = unpickle(
        last_results_path)

    for k, v in lcls.items():
        if isinstance(v, theano.compile.function_module.Function):
            matches = [name for name in last_weights.keys() if k in name]
            sorted_matches = sorted(
                matches, key=lambda x: int(x.split("_")[-1]))
            matching_values = [last_weights[s] for s in sorted_matches]
            set_shared_variables_in_function(v, matching_values)
            checkpoint_dict[k] = v
    return checkpoint_dict
