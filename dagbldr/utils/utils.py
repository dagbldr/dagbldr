# Author: Kyle Kastner
# License: BSD 3-clause
import numpy as np
import theano
from theano import tensor
from theano.scan_module.scan_utils import infer_shape
from theano.gof.fg import MissingInputError
from collections import OrderedDict

TAG_ID = "_dagbldr_"
DATASETS_ID = "__datasets__"
RANDOM_ID = "__random__"


def safe_zip(*args):
    """Like zip, but ensures arguments are of same length.

       Borrowed from pylearn2
    """
    base = len(args[0])
    for i, arg in enumerate(args[1:]):
        if len(arg) != base:
            raise ValueError("Argument 0 has length %d but argument %d has "
                             "length %d" % (base, i+1, len(arg)))
    return zip(*args)


def as_shared(arr, name=None):
    """ Quick wrapper for theano.shared """
    if name is not None:
        return theano.shared(value=arr, borrow=True)
    else:
        return theano.shared(value=arr, name=name, borrow=True)


def concatenate(tensor_list, graph, name, axis=0, force_cast_to_float=True):
    """
    Wrapper to `theano.tensor.concatenate`.
    """
    if force_cast_to_float:
        tensor_list = cast_to_float(tensor_list)
    out = tensor.concatenate(tensor_list, axis=axis)
    # Temporarily commenting out - remove when writing tests
    # conc_dim = int(sum([calc_expected_dim(graph, inp)
    #                for inp in tensor_list]))
    # This may be hosed... need to figure out how to generalize
    # shape = list(expression_shape(tensor_list[0]))
    # shape[axis] = conc_dim
    # new_shape = tuple(shape)
    # tag_expression(out, name, new_shape)
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


def make_shapename(name, shape):
    if len(shape) == 1:
        # vector, primarily init hidden state for RNN
        return name + TAG_ID + str(shape[0]) + "x"
    else:
        return name + TAG_ID + "x".join(map(str, list(shape)))


def parse_shapename(shapename):
    try:
        # Bracket for scan
        shape = shapename.split(TAG_ID)[1].split("[")[0].split("x")
    except AttributeError:
        raise AttributeError("Unable to parse shapename. Has the expression "
                             "been tagged with a shape by tag_expression? "
                             " input shapename was %s" % shapename)
    if "[" in shapename.split(TAG_ID)[1]:
        # inside scan
        shape = shape[1:]
    name = shapename.split(TAG_ID)[0]
    # More cleaning to handle scan
    shape = tuple([int(s) for s in shape if s != ''])
    return name, shape


def names_in_graph(list_of_names, graph):
    """ Return true if all names are in the graph """
    return all([name in graph.keys() for name in list_of_names])


def add_arrays_to_graph(list_of_arrays, list_of_names, graph, strict=True):
    assert type(graph) is OrderedDict
    arrays_added = []
    for array, name in safe_zip(list_of_arrays, list_of_names):
        if name in graph.keys() and strict:
            raise ValueError("Name %s already found in graph!" % name)
        shared_array = as_shared(array, name=name)
        graph[name] = shared_array
        arrays_added.append(shared_array)


def add_fixed_to_graph(list_of_fixed_numpy, list_of_shapes,
                       list_of_names, graph, strict=True):
    assert type(graph) is OrderedDict
    shared_added = []
    if RANDOM_ID not in graph.keys():
        graph[RANDOM_ID] = []
    for n, (fixed, shape, name) in enumerate(safe_zip(list_of_fixed_numpy,
                                                      list_of_shapes,
                                                      list_of_names)):
        shared_array = as_shared(fixed, name=name)
        tag_expression(shared_array, name, shape)
        shared_added.append(shared_array)
    graph[RANDOM_ID] += shared_added
    return shared_added


def add_random_to_graph(list_of_random, list_of_shapes, list_of_names,
                        graph, strict=True):
    assert type(graph) is OrderedDict
    random_added = []
    if RANDOM_ID not in graph.keys():
        graph[RANDOM_ID] = []
    for n, (random, shape, name) in enumerate(safe_zip(list_of_random,
                                                       list_of_shapes,
                                                       list_of_names)):
        tag_expression(random, name, shape)
        random_added.append(random)
    graph[RANDOM_ID] += random_added
    return random_added


def add_datasets_to_graph(list_of_datasets, list_of_names, graph, strict=True,
                          list_of_test_values=None):
    assert type(graph) is OrderedDict
    datasets_added = []
    for n, (dataset, name) in enumerate(safe_zip(list_of_datasets,
                                                 list_of_names)):
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
    graph[DATASETS_ID] = datasets_added
    if len(datasets_added) == 1:
        # Make returned value easier to access
        datasets_added = datasets_added[0]
    return datasets_added


def tag_expression(expression, name, shape):
    expression.name = make_shapename(name, shape)


def expression_name(expression):
    return parse_shapename(expression.name)[0]


def expression_shape(expression):
    return parse_shapename(expression.name)[1]


def alt_shape_of_variables(inputs, outputs, input_shapes):
    # Thanks to P. Lamblin, F. Bastien for help to make this work
    # mapping from initial to cloned var
    equiv = theano.gof.graph.clone_get_equiv(inputs, outputs)
    cloned_inputs = [equiv[inp] for inp in inputs]
    cloned_outputs = [equiv[out] for out in outputs]
    cloned_shapes = {equiv[k]: v for k, v in input_shapes.items()}
    fgraph = theano.FunctionGraph(cloned_inputs, cloned_outputs, clone=False)
    if not hasattr(fgraph, 'shape_feature'):
        fgraph.attach_feature(tensor.opt.ShapeFeature())

    kept_input = [n for n, f in enumerate(fgraph.inputs)
                  if f in fgraph.shape_feature.shape_of.keys()]

    input_dims = [dimension for kept_idx in kept_input
                  for dimension in fgraph.shape_feature.shape_of[
                      fgraph.inputs[kept_idx]]]
    """
    # The old way
    output_dims = [dimension for f, shape in
                   fgraph.shape_feature.shape_of.items()
                   for dimension in shape
                   if f in fgraph.outputs]
    """

    output_dims = list(*infer_shape(cloned_outputs, cloned_inputs,
                                    [input_shapes[k] for k in inputs]))

    try:
        compute_shapes = theano.function(input_dims,
                                         output_dims,
                                         mode=theano.Mode(optimizer=None),
                                         on_unused_input="ignore")

        numeric_input_dims = [dim for kept_idx in kept_input
                              for dim in cloned_shapes[fgraph.inputs[kept_idx]]]

        numeric_output_dims = compute_shapes(*numeric_input_dims)
    except MissingInputError:
        # need to add fake datasets and masks to input args for ?? reasons
        # unfortunate things might start happening if intermediate vars named
        # example that activated this code path
        # data -> linear -> tanh_rnn_layer
        dataset_and_mask_indices = [n for n, f in enumerate(fgraph.inputs)
                                    if f.name is not None]
        compute_shapes = theano.function(
            [fgraph.inputs[i] for i in dataset_and_mask_indices] + input_dims,
            output_dims,
            mode=theano.Mode(optimizer=None),
            on_unused_input="ignore")

        numeric_input_dims = [dim for kept_idx in kept_input
                              for dim in cloned_shapes[fgraph.inputs[kept_idx]]]
        fake_numeric_data = [np.ones(
            cloned_shapes[fgraph.inputs[i]]).astype(fgraph.inputs[i].dtype)
            for i in dataset_and_mask_indices]

        numeric_inputs = fake_numeric_data + numeric_input_dims

        numeric_output_dims = compute_shapes(*numeric_inputs)

    final_shapes = {}
    # This assumes only 1 OUTPUT!!!
    for var in fgraph.outputs:
        final_shapes[var] = np.array(numeric_output_dims).ravel()

    shapes = dict((outputs[n], tuple(np.array(final_shapes[co]).ravel()))
                  for n, co in enumerate(cloned_outputs))
    return shapes


def calc_expected_dims(graph, expression):
    # Intertwined with add_datasets_to_graph and add_random_to_graph
    # Expect variables representing datasets, shared, and random vars in graph
    # Named shape format is HxWxZ
    if expression in graph[DATASETS_ID]:
        # The expression input is a datastet - use tagged info directly
        dims = expression_shape(expression)
    elif RANDOM_ID in graph.keys() and expression in graph[RANDOM_ID]:
        # The expression input is a random variable - use tagged info directly
        dims = expression_shape(expression)
    else:
        # Assume anything not a random value or dataset is shared
        # Use short-circuit AND to avoid key-error if no random values in graph
        all_shared = [s for s in graph.values()
                      if s != graph[DATASETS_ID] and (
                          RANDOM_ID not in graph.keys()
                          or s != graph[RANDOM_ID])]
        #  == may not be good comparison in all cases
        all_random = [r for r in graph.values()
                      if (RANDOM_ID in graph.keys() and r == graph[RANDOM_ID])]
        # Flatten list of lists for random
        all_random = [ri for r in all_random for ri in r]
        all_inputs = graph[DATASETS_ID] + all_shared + all_random
        # Get shapes or fake shapes for all of the inputs
        dataset_shapes = [expression_shape(d) for d in graph[DATASETS_ID]]
        shared_shapes = [s.get_value().shape for s in all_shared]
        random_shapes = [expression_shape(r) for r in all_random]
        all_input_shapes = dataset_shapes + shared_shapes + random_shapes
        # Fake length of 2
        fake_shapes = [(2,) + s[1:] for s in all_input_shapes]
        all_outputs = [expression]
        fake_dict = dict(zip(all_inputs, fake_shapes))
        calc_shapes = alt_shape_of_variables(all_inputs, all_outputs, fake_dict)
        dims = calc_shapes[expression]
    return dims


def fetch_from_graph(list_of_names, graph):
    """ Returns a list of shared variables from the graph """
    if DATASETS_ID not in graph.keys():
        # Check for dataset in graph
        raise AttributeError("No dataset in graph! Make sure to add "
                             "the dataset using add_datasets_to_graph")
    return [graph[name] for name in list_of_names]


def get_params_and_grads(graph, cost):
    grads = []
    params = []
    for k, p in graph.items():
        if k == DATASETS_ID:
            # skip datasets
            continue
        if k == RANDOM_ID:
            # skip random
            continue
        print("Computing grad w.r.t %s" % k)
        grad = tensor.grad(cost, p)
        params.append(p)
        grads.append(grad)
    return params, grads
