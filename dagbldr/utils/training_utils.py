# Author: Kyle Kastner
# License: BSD 3-clause
import numpy as np
import numbers
import theano
import sys
try:
    import cPickle as pickle
except ImportError:
    import pickle
from collections import defaultdict


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
        return zip(start_indices, end_indices)
    else:
        minibatch_indices = np.arange(0, len(itr), minibatch_size)
        minibatch_indices = np.asarray(list(minibatch_indices) + [len(itr)])
        start_indices = minibatch_indices[:-1]
        end_indices = minibatch_indices[1:]
        return zip(start_indices, end_indices)


def convert_to_one_hot(itr, n_classes, dtype="int32"):
    """ Convert 2D or 3D iterators to one_hot. Primarily for text. """
    is_three_d = False
    if type(itr) is np.ndarray:
        if len(itr.shape) == 3:
            is_three_d = True
    elif not isinstance(itr[0], numbers.Real):
        # Assume 3D list of list of list
        # iterable of iterable of iterable, feature dim must be consistent
        is_three_d = True

    if is_three_d:
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
    print("Epoch %i: %s" % (n_epochs_seen, last_results))


def checkpoint_status_func(save_path, checkpoint_dict, epoch_results):
    """ Saves a checkpoint dict """
    checkpoint_dict["previous_epoch_results"] = epoch_results
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


def even_slice(arr, size):
    """ Force array to be even by slicing off the end """
    extent = -(len(arr) % size)
    if extent == 0:
        extent = None
    return arr[:extent]


def make_minibatch(arg, start, stop):
    """ Does not handle off-size minibatches """
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
                     list_of_output_names=None,
                     n_epochs=1000, n_status=50, status_func=None,
                     previous_epoch_results=None,
                     shuffle=False, random_state=None):
    """
    Minibatch arguments should come first.

    Constant arguments which should not be iterated can be passed as
    list_of_non_minibatch_args.

    If list_of_minbatch_functions is length 1, will be replicated to length of
    list_of_args - applying the same function to all minibatch arguments in
    list_of_args. Otherwise, this should be the same length as list_of_args

    list_of_output_names simply names the output of the passed in function.
    Should be the same length as the number of outputs from the function.

    status_func is a function run periodically (based on n_status_points),
    which allows for validation, early stopping, checkpointing, etc.

    previous_epoch_results allows for continuing from saved checkpoints

    shuffle and random_state are used to determine if minibatches are run
    in sequence or selected randomly each epoch.

    By far the craziest function in this file.

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
    status_points = list(range(n_epochs))
    if len(status_points) >= n_status:
        intermediate_points = status_points[::n_epochs // n_status]
        status_points = intermediate_points + [status_points[-1]]
    else:
        status_points = range(len(status_points))

    for arg in list_of_minibatch_args:
        assert len(arg) == len(list_of_minibatch_args[0])

    indices = minibatch_indices(list_of_minibatch_args[0], minibatch_size)
    if len(list_of_minibatch_args[0]) % minibatch_size != 0:
        print ("length of dataset should be evenly divisible by "
               "minibatch_size.")
    if len(list_of_minibatch_functions) == 1:
        list_of_minibatch_functions = list_of_minibatch_functions * len(
            list_of_minibatch_args)
    else:
        assert len(list_of_minibatch_functions) == len(list_of_minibatch_args)
    # Function loop
    for e in range(n_epochs):
        results = defaultdict(list)
        if shuffle:
            random_state.shuffle(indices)
        for i, j in indices:
            minibatch_args = []
            for n, arg in enumerate(list_of_minibatch_args):
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
        avg_output = {r: np.mean(results[r]) for r in results.keys()}
        for k in avg_output.keys():
            epoch_results[k].append(avg_output[k])
        if e in status_points:
            if status_func is not None:
                epoch_number = e
                status_number = np.searchsorted(status_points, e)
                status_func(status_number, epoch_number, epoch_results)
    return epoch_results
