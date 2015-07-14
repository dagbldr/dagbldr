# Author: Kyle Kastner
# License: BSD 3-clause
import numpy as np
import theano
from theano import tensor


def as_shared(arr, name=None):
    """ Quick wrapper for theano.shared """
    if name is not None:
        return theano.shared(value=arr, borrow=True)
    else:
        return theano.shared(value=arr, name=name, borrow=True)


def concatenate(tensor_list, name, axis=0, force_cast_to_float=True):
    """
    Wrapper to `theano.tensor.concatenate`.
    """
    if force_cast_to_float:
        tensor_list = cast_to_float(tensor_list)
    out = tensor.concatenate(tensor_list, axis=axis)
    conc_dim = int(sum([calc_expected_dim(inp)
                   for inp in tensor_list]))
    # This may be hosed... need to figure out how to generalize
    shape = list(expression_shape(tensor_list[0]))
    shape[axis] = conc_dim
    new_shape = tuple(shape)
    tag_expression(out, name, new_shape)
    return out


def theano_repeat(arr, n_repeat, stretch=False):
    """
    Create repeats of 2D array using broadcasting.
    Shape[0] incorrect after this node!
    """
    if arr.dtype not in ["float32", "float64"]:
        arr = tensor.cast(arr, "int32")
    if stretch:
        arg1 = arr.dimshuffle((0, 'x', 1))
        arg2 = tensor.alloc(1., 1, n_repeat, arr.shape[1])
        arg2 = tensor.cast(arg2, arr.dtype)
        cloned = (arg1 * arg2).reshape((n_repeat * arr.shape[0], arr.shape[1]))
    else:
        arg1 = arr.dimshuffle(('x', 0, 1))
        arg2 = tensor.alloc(1., n_repeat, 1, arr.shape[1])
        arg2 = tensor.cast(arg2, arr.dtype)
        cloned = (arg1 * arg2).reshape((n_repeat * arr.shape[0], arr.shape[1]))
    shape = expression_shape(arr)
    name = expression_name(arr)
    # Stretched shapes are *WRONG*
    tag_expression(cloned, name + "_stretched", (shape[0], shape[1]))
    return cloned


def cast_to_float(list_of_inputs):
    """ A cast that preserves name and shape info after cast """
    input_names = [inp.name for inp in list_of_inputs]
    cast_inputs = [tensor.cast(inp, theano.config.floatX)
                   for inp in list_of_inputs]
    for n, inp in enumerate(cast_inputs):
        cast_inputs[n].name = input_names[n]
    return cast_inputs


def interpolate_between_points(arr, n_steps=50):
    """ Helper function for drawing line between points in space """
    assert len(arr) > 2
    assert n_steps > 1
    path = [path_between_points(start, stop, n_steps=n_steps)
            for start, stop in zip(arr[:-1], arr[1:])]
    path = np.vstack(path)
    return path


def path_between_points(start, stop, n_steps=100, dtype=theano.config.floatX):
    """ Helper function for making a line between points in ND space """
    assert n_steps > 1
    step_vector = 1. / (n_steps - 1) * (stop - start)
    steps = np.arange(0, n_steps)[:, None] * np.ones((n_steps, len(stop)))
    steps = steps * step_vector + start
    return steps.astype(dtype)


def names_in_graph(list_of_names, graph):
    """ Return true if all names are in the graph """
    return all([name in graph.keys() for name in list_of_names])


def add_arrays_to_graph(list_of_arrays, list_of_names, graph, strict=True):
    assert len(list_of_arrays) == len(list_of_names)
    arrays_added = []
    for array, name in zip(list_of_arrays, list_of_names):
        if name in graph.keys() and strict:
            raise ValueError("Name %s already found in graph!" % name)
        shared_array = as_shared(array, name=name)
        graph[name] = shared_array
        arrays_added.append(shared_array)


def make_shapename(name, shape):
    if len(shape) == 1:
        # vector, primarily init hidden state for RNN
        return name + "_kdl_" + str(shape[0]) + "x"
    else:
        return name + "_kdl_" + "x".join(map(str, list(shape)))


def parse_shapename(shapename):
    try:
        # Bracket for scan
        shape = shapename.split("_kdl_")[1].split("[")[0].split("x")
    except AttributeError:
        raise AttributeError("Unable to parse shapename. Has the expression "
                             "been tagged with a shape by tag_expression? "
                             " input shapename was %s" % shapename)
    if "[" in shapename.split("_kdl_")[1]:
        # inside scan
        shape = shape[1:]
    name = shapename.split("_kdl_")[0]
    # More cleaning to handle scan
    shape = tuple([int(s) for s in shape if s != ''])
    return name, shape


def add_datasets_to_graph(list_of_datasets, list_of_names, graph, strict=True,
                          list_of_test_values=None):
    assert len(list_of_datasets) == len(list_of_names)
    datasets_added = []
    for n, (dataset, name) in enumerate(zip(list_of_datasets, list_of_names)):
        if dataset.dtype != "int32":
            if len(dataset.shape) == 1:
                sym = tensor.vector()
            elif len(dataset.shape) == 2:
                sym = tensor.matrix()
            elif len(dataset.shape) == 3:
                sym = tensor.tensor3()
            else:
                raise ValueError("dataset %s has unsupported shape" % name)
        elif dataset.dtype == "int32":
            if len(dataset.shape) == 1:
                sym = tensor.ivector()
            elif len(dataset.shape) == 2:
                sym = tensor.imatrix()
            elif len(dataset.shape) == 3:
                sym = tensor.itensor3()
            else:
                raise ValueError("dataset %s has unsupported shape" % name)
        else:
            raise ValueError("dataset %s has unsupported dtype %s" % (
                name, dataset.dtype))
        if list_of_test_values is not None:
            sym.tag.test_value = list_of_test_values[n]
        tag_expression(sym, name, dataset.shape)
        datasets_added.append(sym)
    graph["__datasets_added__"] = datasets_added
    return datasets_added


def tag_expression(expression, name, shape):
    expression.name = make_shapename(name, shape)


def expression_name(expression):
    return parse_shapename(expression.name)[0]


def expression_shape(expression):
    return parse_shapename(expression.name)[1]


def calc_expected_dim(expression):
    # super intertwined with add_datasets_to_graph
    # Expect variables representing datasets in graph!!!
    # Function graph madness
    # Shape format is HxWxZ
    shape = expression_shape(expression)
    dim = shape[-1]
    return dim


def fetch_from_graph(list_of_names, graph):
    """ Returns a list of shared variables from the graph """
    if "__datasets_added__" not in graph.keys():
        # Check for dataset in graph
        raise AttributeError("No dataset in graph! Make sure to add "
                             "the dataset using add_datasets_to_graph")
    return [graph[name] for name in list_of_names]


def get_params_and_grads(graph, cost):
    grads = []
    params = []
    for k, p in graph.items():
        if k[:2] == "__":
            # skip private tags
            continue
        print("Computing grad w.r.t %s" % k)
        grad = tensor.grad(cost, p)
        params.append(p)
        grads.append(grad)
    return params, grads
