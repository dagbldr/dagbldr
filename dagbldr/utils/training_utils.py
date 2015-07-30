# Author: Kyle Kastner
# License: BSD 3-clause
from __future__ import print_function
import numpy as np
import numbers
import theano
import sys
import warnings
import time
import pprint
try:
    import cPickle as pickle
except ImportError:
    import pickle
from collections import defaultdict
from functools import reduce


def minibatch_indices(itr, minibatch_size):
    """ Generate indices for slicing 2D and 3D arrays in minibatches"""
    is_three_d = False
    if type(itr) is np.ndarray:
        if len(itr.shape) == 3:
            is_three_d = True
    elif not isinstance(itr[0], numbers.Real):
        # Assume 3D list of list of list
        # iterable of iterable of iterable, feature dim must be consistent
        is_three_d = True

    if is_three_d:
        if type(itr) is np.ndarray:
            minibatch_indices = np.arange(0, itr.shape[1], minibatch_size)
        else:
            # multi-list
            minibatch_indices = np.arange(0, len(itr), minibatch_size)
        minibatch_indices = np.asarray(list(minibatch_indices) + [len(itr)])
        start_indices = minibatch_indices[:-1]
        end_indices = minibatch_indices[1:]
        return list(zip(start_indices, end_indices))
    else:
        minibatch_indices = np.arange(0, len(itr), minibatch_size)
        minibatch_indices = np.asarray(list(minibatch_indices) + [len(itr)])
        start_indices = minibatch_indices[:-1]
        end_indices = minibatch_indices[1:]
        return list(zip(start_indices, end_indices))


def make_character_level_from_text(text):
    """ Create mapping and inverse mappings for text -> one_hot_char

    Parameters
    ----------
    text : iterable of strings

    Returns
    -------
    cleaned : list of list of ints, length (len(text), )
         The original text, converted into list of list of integers

    mapper_func : function
         A function that can be used to map text into the correct form

    inverse_mapper_func : function
        A function that can be used to invert the output of mapper_func

    mapper : dict
        Dictionary containing the mapping of char -> integer

    """

    # Try to catch invalid input
    try:
        ord(text[0])
        raise ValueError("Text should be iterable of strings")
    except TypeError:
        pass
    all_chars = reduce(lambda x, y: set(x) | set(y), text, set())
    mapper = {k: n + 2 for n, k in enumerate(list(all_chars))}
    # 1 is EOS
    mapper["EOS"] = 1
    # 0 is UNK/MASK - unused here but needed in general
    mapper["UNK"] = 0
    inverse_mapper = {v: k for k, v in mapper.items()}

    def mapper_func(text_line):
        return [mapper[c] if c in mapper.keys() else mapper["UNK"]
                for c in text_line] + [mapper["EOS"]]

    def inverse_mapper_func(symbol_line):
        return "".join([inverse_mapper[s] for s in symbol_line
                        if s != mapper["EOS"]])

    # Remove blank lines
    cleaned = [mapper_func(t) for t in text if t != ""]
    return cleaned, mapper_func, inverse_mapper_func, mapper


def convert_to_one_hot(itr, n_classes, dtype="int32"):
    """ Convert 1D or 2D iterators of class to 2D or 3D iterators of one hot
        class indicators.

        Parameters
        ----------
        itr : iterator
            itr can be list of list, 1D or 2D np.array. In all cases, the
            fundamental element must have type int32 or int64.

        n_classes : int
           number of classes to expand itr to - this will become shape[-1] of
           the returned array.

        dtype : optional, default "int32"
           dtype for the returned array.

        Returns
        -------
        one_hot : array
           A 2D or 3D numpy array of one_hot values. List of list or 2D
           np.array will return a 3D numpy array, while 1D itr or list will
           return a 2D one_hot.

    """
    is_two_d = False
    error_msg = """itr not understood. convert_to_one_hot accepts\n
                   list of list of int, 1D or 2D numpy arrays of\n
                   dtype int32 or int64"""
    if type(itr) is np.ndarray:
        if len(itr.shape) == 2:
            is_two_d = True
        if itr.dtype not in [np.int32, np.int64]:
            raise ValueError(error_msg)
    elif not isinstance(itr[0], numbers.Real):
        # Assume list of list
        # iterable of iterable, feature dim must be consistent
        is_two_d = True
    else:
        raise ValueError(error_msg)

    if is_two_d:
        lengths = [len(i) for i in itr]
        one_hot = np.zeros((max(lengths), len(itr), n_classes), dtype=dtype)
        for n in range(len(itr)):
            one_hot[np.arange(lengths[n]), n, itr[n]] = 1
    else:
        one_hot = np.zeros((len(itr), n_classes), dtype=dtype)
        one_hot[np.arange(len(itr)), itr] = 1
    return one_hot


def save_checkpoint(save_path, items_dict):
    """ Simple wrapper for checkpoint dictionaries """
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(40000)
    with open(save_path, mode="wb") as f:
        pickle.dump(items_dict, f)
    sys.setrecursionlimit(old_recursion_limit)


def load_checkpoint(save_path):
    """ Simple pickle wrapper for checkpoint dictionaries """
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(40000)
    with open(save_path, mode="rb") as f:
        items_dict = pickle.load(f)
    sys.setrecursionlimit(old_recursion_limit)
    return items_dict


def print_status_func(epoch_results):
    """ Print the last results from a results dictionary """
    n_epochs_seen = max([len(l) for l in epoch_results.values()])
    last_results = {k: v[-1] for k, v in epoch_results.items()}
    pp = pprint.PrettyPrinter()
    epochline = "Epoch %i" % n_epochs_seen
    breakline = "".join(["-"] * (len(epochline) + 1))
    print(breakline)
    print(epochline)
    print(breakline)
    pp.pprint(last_results)


def checkpoint_status_func(save_path, checkpoint_dict, epoch_results,
                           nan_check=True):
    """ Saves a checkpoint dict """
    checkpoint_dict["previous_epoch_results"] = epoch_results
    nan_test = [(k, True) for k, e_v in epoch_results.items()
                for v in e_v if np.isnan(v)]
    if nan_check and len(nan_test) > 0:
        nan_keys = set([tup[0] for tup in nan_test])
        raise ValueError("Found NaN values in the following keys ",
                         "%s, exiting training without saving" % nan_keys)
    save_checkpoint(save_path, checkpoint_dict)
    print_status_func(epoch_results)


def early_stopping_status_func(valid_cost, save_path, checkpoint_dict,
                               epoch_results):
    """
    Adds valid_cost to epoch_results and saves model if best valid
    Assumes checkpoint_dict is a defaultdict(list)

    Example usage for early stopping on validation set:

    def status_func(status_number, epoch_number, epoch_results):
        valid_results = iterate_function(
            cost_function, [X_clean_valid, y_clean_valid], minibatch_size,
            list_of_output_names=["valid_cost"],
            list_of_minibatch_functions=[text_minibatcher], n_epochs=1,
            shuffle=False)
        early_stopping_status_func(valid_results["valid_cost"][-1],
                                save_path, checkpoint_dict, epoch_results)

    status_func can then be fed to iterate_function for training with early
    stopping.
    """
    # Quick trick to avoid 0 length list
    old = min(epoch_results["valid_cost"] + [np.inf])
    epoch_results["valid_cost"].append(valid_cost)
    new = min(epoch_results["valid_cost"])
    if new < old:
        print("Saving checkpoint based on validation score")
        checkpoint_status_func(save_path, checkpoint_dict, epoch_results)
    else:
        print_status_func(epoch_results)


def default_status_func(status_number, epoch_number, epoch_results):
    """ Default status function for iterate_function. Prints epoch info.

    This is exactly equivalent to defining your own status_function as such:
        def status_func(status_number, epoch_number, epoch_results):
            print_status_func(epoch_results)

    Parameters
    ----------
    status_number

    epoch_number

    epoch_results

    """
    print_status_func(epoch_results)


def even_slice(arr, size):
    """ Force array to be even by slicing off the end """
    extent = -(len(arr) % size)
    if extent == 0:
        extent = None
    return arr[:extent]


def make_minibatch(arg, start, stop):
    """ Does not handle off-size minibatches

        Returns list of single element for API compat with mask generators
    """
    if len(arg.shape) == 3:
        return [arg[:, start:stop]]
    else:
        return [arg[start:stop]]


def gen_text_minibatch_func(one_hot_size):
    """
    Returns a function that will turn a text minibatch into one_hot form.

    For use with iterate_function list_of_minibatch_functions argument.

    Example:
    n_chars = 84
    text_minibatcher = gen_text_minibatch_func(n_chars)
    valid_results = iterate_function(
        cost_function, [X_clean_valid, y_clean_valid], minibatch_size,
        list_of_output_names=["valid_cost"],
        list_of_minibatch_functions=[text_minibatcher], n_epochs=1,
        shuffle=False)
    """
    def apply(arg, start, stop):
        sli = arg[start:stop]
        expanded = convert_to_one_hot(sli, one_hot_size)
        lengths = [len(s) for s in sli]
        mask = np.zeros((max(lengths), len(sli)), dtype=theano.config.floatX)
        for n, l in enumerate(lengths):
            mask[np.arange(l), n] = 1.
        return expanded, mask
    return apply


def iterate_function(func, list_of_minibatch_args, minibatch_size,
                     list_of_non_minibatch_args=None,
                     list_of_minibatch_functions=[make_minibatch],
                     list_of_preprocessing_functions=None,
                     list_of_output_names=None,
                     n_epochs=1000,
                     n_epoch_status=.02,
                     epoch_status_func=default_status_func,
                     n_minibatch_status=.02,
                     previous_epoch_results=None,
                     shuffle=False, random_state=None):
    """
    Minibatch arguments should come first.

    Constant arguments which should not be iterated can be passed as
    list_of_non_minibatch_args.

    If list_of_minibatch_functions is length 1, will be replicated to length of
    list_of_args - applying the same function to all minibatch arguments in
    list_of_args. Otherwise, this should be the same length as list_of_args

    list_of_output_names simply names the output of the passed in function.
    Should be the same length as the number of outputs from the function.

    status_func is a function run periodically (based on n_status_points),
    which allows for validation, early stopping, checkpointing, etc.

    previous_epoch_results allows for continuing from saved checkpoints

    n_minibatch_status

    shuffle and random_state are used to determine if minibatches are run
    in sequence or selected randomly each epoch.

    By far the craziest function in this library.

    Example validation function:
    n_chars = 84
    text_minibatcher = gen_text_minibatch_func(n_chars)

    cost_function returns one value, the cost for that minibatch

    valid_results = iterate_function(
        cost_function, [X_clean_valid, y_clean_valid], minibatch_size,
        list_of_output_names=["valid_cost"],
        list_of_minibatch_functions=[text_minibatcher], n_epochs=1,
        shuffle=False)

    Example training loop:

    fit_function returns 3 values, nll, kl and the total cost

    epoch_results = iterate_function(fit_function, [X, y], minibatch_size,
                                 list_of_output_names=["nll", "kl", "cost"],
                                 n_epochs=2000,
                                 status_func=status_func,
                                 previous_epoch_results=previous_epoch_results,
                                 shuffle=True,
                                 random_state=random_state)
    """
    if previous_epoch_results is None:
        epoch_results = defaultdict(list)
    else:
        epoch_results = previous_epoch_results
    # Input checking and setup
    if shuffle:
        assert random_state is not None

    for arg in list_of_minibatch_args:
        assert len(arg) == len(list_of_minibatch_args[0])

    indices = minibatch_indices(list_of_minibatch_args[0], minibatch_size)
    if len(list_of_minibatch_args[0]) % minibatch_size != 0:
        warnings.warn("Length of dataset should be evenly divisible by "
                      "minibatch_size.", UserWarning)

    if n_epoch_status <= 0:
        raise ValueError("n_epoch_status must be > 0")
    elif n_epoch_status < 1:
        n_epoch_status = int(n_epoch_status * n_epochs)
        if n_epoch_status < 1:
            # Update once per epoch
            n_epoch_status = n_epochs
    assert n_epoch_status > 0
    assert n_epochs >= n_epoch_status

    if n_minibatch_status <= 0:
        raise ValueError("n_minibatch_status must be > 0")
    elif n_minibatch_status < 1:
        n_minibatch_status = int(n_minibatch_status * len(indices))
        if n_minibatch_status < 1:
            # fall back to 1 update
            n_minibatch_status = len(indices)
    assert n_minibatch_status > 0

    status_points = list(range(n_epochs))
    if len(status_points) >= n_epoch_status:
        intermediate_points = status_points[::n_epoch_status]
        status_points = intermediate_points + [status_points[-1]]
    else:
        status_points = range(len(status_points))

    if len(list_of_minibatch_functions) == 1:
        list_of_minibatch_functions = list_of_minibatch_functions * len(
            list_of_minibatch_args)
    else:
        assert len(list_of_minibatch_functions) == len(list_of_minibatch_args)

    if list_of_preprocessing_functions is not None and len(
      list_of_preprocessing_functions) == 1:
        list_of_preprocessing_functions = list_of_preprocessing_functions * len(
            list_of_minibatch_args)
    elif list_of_preprocessing_functions is not None:
        assert len(list_of_preprocessing_functions) == len(
            list_of_minibatch_args)
    else:
        assert list_of_preprocessing_functions is None

    # Function loop
    global_start = time.time()
    for e in range(n_epochs):
        epoch_start = time.time()
        results = defaultdict(list)
        if shuffle:
            random_state.shuffle(indices)
        for minibatch_count, (i, j) in enumerate(indices):
            minibatch_args = []
            for n, arg in enumerate(list_of_minibatch_args):
                if list_of_preprocessing_functions is not None:
                    minibatch_args += [list_of_preprocessing_functions[n](
                        *list_of_minibatch_functions[n](arg, i, j))]
                else:
                    # list of minibatch_functions can't always be the right size
                    # (enc-dec with mask coming from mb func)
                    minibatch_args += list_of_minibatch_functions[n](arg, i, j)
            if list_of_non_minibatch_args is not None:
                all_args = minibatch_args + list_of_non_minibatch_args
            else:
                all_args = minibatch_args
            minibatch_results = func(*all_args)
            if type(minibatch_results) is not list:
                minibatch_results = [minibatch_results]
            for n, k in enumerate(minibatch_results):
                if list_of_output_names is not None:
                    assert len(list_of_output_names) == len(minibatch_results)
                    results[list_of_output_names[n]].append(
                        minibatch_results[n])
                else:
                    results[n].append(minibatch_results[n])
            if minibatch_count % n_minibatch_status == 0:
                print("minibatch %i/%i" % (minibatch_count, len(indices) - 1))
        epoch_stop = time.time()
        output = {r: np.mean(results[r]) for r in results.keys()}
        output["mean_minibatch_time_s"] = (epoch_stop - epoch_start) / float(
            minibatch_count + 1)
        output["mean_sample_time_s"] = (epoch_stop - epoch_start) / float(
            len(list_of_minibatch_args[0]) * (minibatch_count + 1))
        output["this_epoch_time_s"] = epoch_stop - epoch_start
        output["mean_epoch_time_s"] = (epoch_stop - global_start) / float(e + 1)
        output["total_training_time_s"] = epoch_stop - global_start
        output["total_number_of_updates"] = (e + 1) * (minibatch_count + 1)
        for k in output.keys():
            epoch_results[k].append(output[k])
        if e in status_points:
            if epoch_status_func is not None:
                epoch_number = e
                status_number = np.searchsorted(status_points, e)
                epoch_status_func(status_number, epoch_number, epoch_results)
    return epoch_results
