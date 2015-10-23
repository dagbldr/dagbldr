# Author: Kyle Kastner
# License: BSD 3-clause
from __future__ import print_function
import __main__ as main
import os
import re
import shutil
import numpy as np
import glob
import numbers
import theano
import sys
import warnings
import inspect
import zipfile
import time
import pprint
try:
    import cPickle as pickle
except ImportError:
    import pickle
from collections import defaultdict
from functools import reduce
from .plot_utils import _filled_js_template_from_results_dict

# TODO: Fetch from env
NUM_SAVED_TO_KEEP = 2


def get_checkpoint_dir(checkpoint_dir=None, folder=None, create_dir=True):
    """ Get checkpoint directory path """
    if not checkpoint_dir:
        checkpoint_dir = os.getenv("DAGBLDR_MODELS", os.path.join(
            os.path.expanduser("~"), "dagbldr_models"))
    if folder is None:
        checkpoint_name = main.__file__.split(".")[0]
        checkpoint_dir = os.path.join(checkpoint_dir, checkpoint_name)
    else:
        checkpoint_dir = os.path.join(checkpoint_dir, folder)
    if not os.path.exists(checkpoint_dir) and create_dir:
        os.makedirs(checkpoint_dir)
    return checkpoint_dir


def _in_nosetest():
    return sys.argv[0].endswith('nosetests')


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


def whitespace_tokenizer(line):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', line) if x.strip()]


def make_word_level_from_text(text, tokenizer="default"):
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
    all_words = reduce(lambda x, y: set(
        whitespace_tokenizer(x)) | set(whitespace_tokenizer(y)), text, set())
    mapper = {k: n + 2 for n, k in enumerate(list(all_words))}
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
    if type(itr) is np.ndarray and itr.dtype not in [np.object]:
        if len(itr.shape) == 2:
            is_two_d = True
        if itr.dtype not in [np.int32, np.int64]:
            raise ValueError(error_msg)
    elif not isinstance(itr[0], numbers.Real):
        # Assume list of list
        # iterable of iterable, feature dim must be consistent
        is_two_d = True
    elif itr.dtype in [np.object]:
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
        pickle.dump(items_dict, f, protocol=-1)
    sys.setrecursionlimit(old_recursion_limit)


def load_checkpoint(save_path):
    """ Simple pickle wrapper for checkpoint dictionaries """
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(40000)
    with open(save_path, mode="rb") as f:
        items_dict = pickle.load(f)
    sys.setrecursionlimit(old_recursion_limit)
    return items_dict


def load_last_checkpoint(append_name=None):
    """ Simple pickle wrapper for checkpoint dictionaries """
    save_paths = glob.glob(os.path.join(get_checkpoint_dir(), "*.pkl"))
    if len(save_paths) == 0:
        # No saved checkpoint, return empty dict
        return {}
    if append_name is not None:
        save_paths = [s.split(append_name)[:-1] + s.split(append_name)[-1:]
                      for s in save_paths]
    sorted_paths = _get_file_matches("*.pkl", "best")
    last_checkpoint_path = sorted_paths[-1]
    print("Loading checkpoint from %s" % last_checkpoint_path)
    return load_checkpoint(last_checkpoint_path)


def _write_results_as_html(results_dict, save_path, default_show="all"):
    as_html = _filled_js_template_from_results_dict(
        results_dict, default_show=default_show)
    with open(save_path, "w") as f:
        f.writelines(as_html)


def _get_file_matches(glob_ext, append_name):
    all_files = glob.glob(
        os.path.join(get_checkpoint_dir(), glob_ext))
    if append_name is None:
        # This 3 is definitely brittle - need better checks
        selected = [f for n, f in enumerate(all_files)
                    if len(f.split(os.sep)[-1].split("_")) == 3]
    else:
        selected = [f for n, f in enumerate(all_files)
                    if append_name in f.split(os.sep)[-1]]

    def key_func(x):
        return int(x.split(os.sep)[-1].split(".")[0].split("_")[-1])

    int_selected = []
    for s in selected:
        try:
            key_func(s)
            int_selected.append(s)
        except ValueError:
            pass

    selected = sorted(int_selected, key=key_func)
    return selected


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def _remove_old_files(sorted_files_list):
    n_saved_to_keep = NUM_SAVED_TO_KEEP
    if len(sorted_files_list) > n_saved_to_keep:
        times = [os.path.getctime(f) for f in sorted_files_list]
        times_rank = argsort(times)
        for t, f in zip(times_rank, sorted_files_list):
            if t not in range(0, len(times))[-n_saved_to_keep:]:
                os.remove(f)


def _cleanup_monitors(partial_match, append_name=None):
    selected_monitors = _get_file_matches(
        "*" + partial_match + "*.html", append_name)
    _remove_old_files(selected_monitors)


def _cleanup_checkpoints(append_name=None):
    selected_checkpoints = _get_file_matches("*.pkl", append_name)
    _remove_old_files(selected_checkpoints)


def _zip_dir(src, dst):
    zf = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
    abs_src = os.path.abspath(src)
    exclude_exts = [".js", ".pyc", ".html", ".txt", ".csv", ".gz"]
    for root, dirs, files in os.walk(src):
        for fname in files:
            if all([e not in fname for e in exclude_exts]):
                absname = os.path.abspath(os.path.join(root, fname))
                arcname = "dagbldr" + os.sep + absname[len(abs_src) + 1:]
                zf.write(absname, arcname)
    zf.close()


def _archive_dagbldr():
    checkpoint_dir = get_checkpoint_dir()
    code_snapshot_dir = os.path.join(checkpoint_dir, "code_snapshot")
    if not os.path.exists(code_snapshot_dir):
        os.mkdir(code_snapshot_dir)
    try:
        theano_flags = os.environ["THEANO_FLAGS"]
        command_string = 'THEANO_FLAGS="' + theano_flags + '" ' + "python "
        command_string += " ".join(sys.argv)
    except KeyError:
        command_string = "python " + " ".join(sys.argv)
    command_script_path = os.path.join(code_snapshot_dir, "run.sh")
    if not os.path.exists(command_script_path):
        with open(command_script_path, 'w') as f:
            f.writelines(command_string)
    save_script_path = os.path.join(code_snapshot_dir, main.__file__)
    training_utils_dir = inspect.getfile(inspect.currentframe())
    lib_dir = str(os.sep).join(training_utils_dir.split(os.sep)[:-2])
    save_lib_path = os.path.join(code_snapshot_dir, "dagbldr_archive.zip")
    existing_reports = glob.glob(os.path.join(checkpoint_dir, "*.html"))
    existing_models = glob.glob(os.path.join(checkpoint_dir, "*.pkl"))
    empty = all([len(l) == 0 for l in (existing_reports, existing_models)])
    if not os.path.exists(save_script_path) or empty:
        print("Saving code archive %s at %s" % (lib_dir, save_lib_path))
        script_location = os.path.abspath(sys.argv[0])
        shutil.copy2(script_location, save_script_path)
        _zip_dir(lib_dir, save_lib_path)


def monitor_status_func(results_dict, append_name=None,
                        status_type="checkpoint",
                        nan_check=True, print_output=True):
    """ Dump the last results from a results dictionary """
    n_seen = max([len(l) for l in results_dict.values()])
    last_results = {k: v[-1] for k, v in results_dict.items()}
    # This really, really assumes a 1D numpy array (1,) or (1, 1)
    last_results = {k: float("%.15f" % v.ravel()[-1])
                    if isinstance(v, (np.generic, np.ndarray))
                    else float("%.15f" % v)
                    for k, v in last_results.items()}
    pp = pprint.PrettyPrinter()
    filename = main.__file__
    fileline = "Script %s" % str(filename)
    if status_type == "checkpoint":
        statusline = "Checkpoint %i" % n_seen
    else:
        raise ValueError("Unknown status_type %s" % status_type)
    breakline = "".join(["-"] * (len(statusline) + 1))
    if print_output:
        print(breakline)
        print(fileline)
        print(statusline)
        print(breakline)
        pp.pprint(last_results)
    if status_type == "checkpoint":
        save_path = os.path.join(get_checkpoint_dir(),
                                 "model_checkpoint_%i.html" % n_seen)

    if append_name is not None:
        split = save_path.split("_")
        save_path = "_".join(
            split[:-1] + [append_name] + split[-1:])
    if not _in_nosetest():
        # Don't dump if testing!
        # Only enable user defined keys
        nan_test = [(k, True) for k, r_v in results_dict.items()
                    for v in r_v if np.isnan(v)]
        if nan_check and len(nan_test) > 0:
            nan_keys = set([tup[0] for tup in nan_test])
            raise ValueError("Found NaN values in the following keys ",
                             "%s, exiting training" % nan_keys)
        show_keys = [k for k in results_dict.keys()
                     if "_auto" not in k]
        _write_results_as_html(results_dict, save_path,
                               default_show=show_keys)
        if status_type == "checkpoint":
            _cleanup_monitors("checkpoint", append_name)


def checkpoint_status_func(checkpoint_dict, epoch_results,
                           append_name=None, nan_check=True):
    """ Saves a checkpoint dict """
    checkpoint_dict["previous_epoch_results"] = epoch_results
    nan_test = [(k, True) for k, e_v in epoch_results.items()
                for v in e_v if np.isnan(v)]
    if nan_check and len(nan_test) > 0:
        nan_keys = set([tup[0] for tup in nan_test])
        raise ValueError("Found NaN values in the following keys ",
                         "%s, exiting training without saving" % nan_keys)

    n_epochs_seen = max([len(l) for l in epoch_results.values()])
    save_path = os.path.join(get_checkpoint_dir(),
                             "model_checkpoint_%i.pkl" % n_epochs_seen)
    if append_name is not None:
        split = save_path.split("_")
        save_path = "_".join(
            split[:-1] + [append_name] + split[-1:])
    if not _in_nosetest():
        # Don't dump if testing!
        save_checkpoint(save_path, checkpoint_dict)
        _cleanup_checkpoints(append_name)
    monitor_status_func(epoch_results, append_name=append_name)


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
    monitor_status_func(epoch_results)


def even_slice(arr, size):
    """ Force array to be even by slicing off the end """
    extent = -(len(arr) % size)
    if extent == 0:
        extent = None
    return arr[:extent]


def make_minibatch(arg, slice_or_indices_list):
    """ Does not handle off-size minibatches
        returns list of [arg, mask] mask of ones if 3D
        else [arg]

    """
    if len(arg.shape) == 3:
        sliced = arg[:, slice_or_indices_list, :]
        return [sliced, np.ones_like(sliced[:, :, 0].astype(
            theano.config.floatX))]
    elif len(arg.shape) == 2:
        return [arg[slice_or_indices_list, :]]
    else:
        return [arg[slice_or_indices_list]]


def make_masked_minibatch(arg, slice_or_indices_list):
    """ Create masked minibatches
        returns list of [arg, mask]
    """
    sliced = arg[slice_or_indices_list]
    is_two_d = True
    if len(sliced[0].shape) > 1:
        is_two_d = False

    if is_two_d:
        d0 = [s.shape[0] for s in sliced]
        max_len = max(d0)
        batch_size = len(sliced)
        data = np.zeros((batch_size, max_len)).astype(sliced[0].dtype)
        mask = np.zeros((batch_size, max_len)).astype(theano.config.floatX)
        for n, s in enumerate(sliced):
            data[n, :len(s)] = s
            mask[n, :len(s)] = 1
    else:
        d0 = [s.shape[0] for s in sliced]
        d1 = [s.shape[1] for s in sliced]
        max_len = max(d0)
        batch_size = len(sliced)
        dim = d1[0]
        same_dim = all([d == dim for d in d1])
        assert same_dim
        data = np.zeros((max_len, batch_size, dim)).astype(theano.config.floatX)
        mask = np.zeros((max_len, batch_size)).astype(theano.config.floatX)
        for n, s in enumerate(sliced):
            data[:len(s), n] = s
            mask[:len(s), n] = 1
    return [data, mask]


def make_embedding_minibatch(arg, slice_type):
    if type(slice_type) is not slice:
        raise ValueError("Text formatters for list of list can only use "
                         "slice objects")
    sli = arg[slice_type]
    lengths = [len(s) for s in sli]
    maxlen = max(lengths)
    mask = np.zeros((max(lengths), len(sli)), dtype=theano.config.floatX)
    expanded = [np.zeros((maxlen,), dtype="int32") for s in sli]
    for n, l in enumerate(lengths):
        mask[:l, n] = 1.
        expanded[n][:l] = sli[n]
    return expanded, mask


def gen_make_one_hot_minibatch(n_targets):
    """ returns function that returns list """
    def make_one_hot_minibatch(arg, slice_or_indices_list):
        non_one_hot_minibatch = make_minibatch(
            arg, slice_or_indices_list)[0].squeeze()
        return [convert_to_one_hot(non_one_hot_minibatch, n_targets)]
    return make_one_hot_minibatch


def gen_make_masked_one_hot_minibatch(n_targets):
    """ returns function that returns list """
    def make_masked_one_hot_minibatch(arg, slice_or_indices_list):
        non_one_hot_minibatch = make_minibatch(
            arg, slice_or_indices_list)[0].squeeze()
        max_len = max([len(i) for i in non_one_hot_minibatch])
        mask = np.zeros((max_len, len(non_one_hot_minibatch))).astype(
            theano.config.floatX)
        for n, i in enumerate(non_one_hot_minibatch):
            mask[:len(i), n] = 1.
        return [convert_to_one_hot(non_one_hot_minibatch, n_targets), mask]
    return make_masked_one_hot_minibatch


def gen_make_list_one_hot_minibatch(n_targets):
    """
    Returns a function that will turn a list into a minibatch of one_hot form.

    For use with iterate_function list_of_minibatch_functions argument.

    Example:
    n_chars = 84
    text_minibatcher = gen_make_text_minibatch_func(n_chars)
    valid_results = iterate_function(
        cost_function, [X_clean_valid, y_clean_valid], minibatch_size,
        list_of_output_names=["valid_cost"],
        list_of_minibatch_functions=[text_minibatcher], n_epochs=1,
        shuffle=False)
    """
    def make_list_one_hot_minibatch(arg, slice_type):
        if type(slice_type) is not slice:
            raise ValueError("Text formatters for list of list can only use "
                             "slice objects")
        sli = arg[slice_type]
        expanded = convert_to_one_hot(sli, n_targets)
        lengths = [len(s) for s in sli]
        mask = np.zeros((max(lengths), len(sli)), dtype=theano.config.floatX)
        for n, l in enumerate(lengths):
            mask[np.arange(l), n] = 1.
        return expanded, mask
    return make_list_one_hot_minibatch


def _make_minibatch_from_indices(indices, minibatch_size):
    if len(indices) % minibatch_size != 0:
        warnings.warn("WARNING:Length of dataset should be evenly divisible"
                      "by minibatch_size - slicing to match.", UserWarning)
        indices = even_slice(indices,
                             len(indices) - len(indices) % minibatch_size)
        assert(len(indices) % minibatch_size == 0)
    minibatch_indices = [indices[i:i + minibatch_size]
                         for i in np.arange(0, len(indices), minibatch_size)]
    # Check for contiguity to avoid unnecessary copies
    minibatch_indices = [slice(mi[0], mi[-1] + 1, 1)
                         if np.all(
                             np.abs(np.array(mi) - np.arange(
                                 mi[0], mi[-1] + 1, 1))
                             < 1E-8)
                         else mi
                         for mi in minibatch_indices]
    return minibatch_indices


def _apply_function_over_minibatch(function, list_of_minibatch_args,
                                   list_of_minibatch_functions, mi):
    minibatch_args = []
    for n, arg in enumerate(list_of_minibatch_args):
        # list of minibatch_functions can't always be the right size
        # (enc-dec with mask coming from mb func)
        r = list_of_minibatch_functions[n](arg, mi)
        # support embeddings
        if type(r[0]) is list:
            minibatch_args += r[0]
            minibatch_args += r[1:]
        else:
            minibatch_args += r
    all_args = minibatch_args
    minibatch_results = function(*all_args)
    if type(minibatch_results) is not list:
        minibatch_results = [minibatch_results]
    return minibatch_results


def _save_best_functions(train_function, valid_function, optimizer_object=None,
                         fname="__functions.pkl"):
    if not _in_nosetest():
        checkpoint_dir = get_checkpoint_dir()
        save_path = os.path.join(checkpoint_dir, fname)
        save_checkpoint(save_path, {"train_function": train_function,
                                    "valid_function": valid_function,
                                    "optimizer_object": optimizer_object})


def _load_best_functions(fname="__functions.pkl"):
    if not _in_nosetest():
        checkpoint_dir = get_checkpoint_dir()
        save_path = os.path.join(checkpoint_dir, fname)
        chk = load_checkpoint(save_path)
        return chk["train_function"], chk["valid_function"], chk["optimizer_object"]


def _save_best_results(results, fname="__results.pkl"):
    if not _in_nosetest():
        checkpoint_dir = get_checkpoint_dir()
        save_path = os.path.join(checkpoint_dir, fname)
        save_checkpoint(save_path, results)


def _load_best_results(fname="__results.pkl"):
    if not _in_nosetest():
        checkpoint_dir = get_checkpoint_dir()
        save_path = os.path.join(checkpoint_dir, fname)
        return load_checkpoint(save_path)


def _init_results_dict():
    results = defaultdict(list)
    results["total_number_of_updates_auto"] = [0]
    results["total_number_of_epochs_auto"] = [0]
    results["current_patience_auto"] = [-1]
    return results


def _iterate_function(train_function, valid_function,
                      train_indices, valid_indices,
                      list_of_minibatch_args, minibatch_size,
                      monitor_function=None,
                      monitor_indices="valid",
                      checkpoint_dict=None,
                      list_of_minibatch_functions=[make_minibatch],
                      list_of_train_output_names=None,
                      valid_output_name=None,
                      valid_frequency="valid_length",
                      n_epochs=100,
                      optimizer_object=None,
                      patience_based_stopping=False,
                      patience_minimum="valid_length",
                      patience_increase=2,
                      patience_improvement=0.995,
                      previous_results=None,
                      shuffle=False, random_state=None,
                      verbose=False):
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
    """
    if previous_results is None:
        previous_results = defaultdict(list)

    # Input checking and setup
    if shuffle:
        assert random_state is not None

    for arg in list_of_minibatch_args:
        assert len(arg) == len(list_of_minibatch_args[0])

    # Bad things happen if this is out of bounds
    final_index = max(max(train_indices), max(valid_indices))
    assert final_index < len(list_of_minibatch_args[0])

    train_minibatch_indices = _make_minibatch_from_indices(train_indices,
                                                           minibatch_size)
    valid_minibatch_indices = _make_minibatch_from_indices(valid_indices,
                                                           minibatch_size)

    if valid_frequency == "valid_length":
        valid_frequency = len(valid_minibatch_indices)
    elif valid_frequency == "train_length":
        valid_frequency = len(train_minibatch_indices)
    else:
        assert valid_frequency >= 1

    if patience_minimum == "valid_length":
        patience_minimum = len(valid_indices)
    else:
        assert patience_minimum > 1

    if len(list_of_minibatch_functions) == 1:
        list_of_minibatch_functions = list_of_minibatch_functions * len(
            list_of_minibatch_args)
    else:
        assert len(list_of_minibatch_functions) == len(list_of_minibatch_args)

    # Function loop
    global_start = time.time()
    if not _in_nosetest():
        _archive_dagbldr()
        # add calling commandline arguments here...

    if len(previous_results.keys()) != 0:
        last_sample_count = previous_results[
            "total_number_of_samples_auto"][-1]
        last_update_count = previous_results[
            "total_number_of_updates_auto"][-1]
        last_epoch_count = previous_results["total_number_of_epochs_auto"][-1]
    else:
        last_sample_count = 0
        last_update_count = 0
        last_epoch_count = 0

    last_valid_count = 0
    if len(previous_results.keys()) != 0:
        old_patience = previous_results["current_patience_auto"][-1]
        patience = patience_increase * old_patience
    else:
        patience = patience_minimum
    done_looping = False
    results = _init_results_dict()
    for e in range(n_epochs):
        new_epoch_count = last_epoch_count + 1
        if patience_based_stopping and done_looping:
            break
        epoch_start = time.time()
        if shuffle:
            random_state.shuffle(train_minibatch_indices)

        for minibatch_count, mi in enumerate(train_minibatch_indices):
            train_minibatch_results = _apply_function_over_minibatch(
                train_function, list_of_minibatch_args,
                list_of_minibatch_functions, mi)
            for n, k in enumerate(train_minibatch_results):
                if list_of_train_output_names is not None:
                    assert len(list_of_train_output_names) == len(
                        train_minibatch_results)
                    results[list_of_train_output_names[n]].append(
                        train_minibatch_results[n])
                else:
                    results[n].append(train_minibatch_results[n])
            # assumes this is a slice object which is *almost* always true
            new_sample_count = last_sample_count + (mi.stop - mi.start)
            new_update_count = last_update_count + 1
            if new_update_count >= (last_valid_count + valid_frequency):
                last_valid_count = new_update_count
                # Validation and monitoring here...
                if monitor_function is not None:
                    print("Running monitor at update %i" % last_update_count)
                    if monitor_indices == "valid":
                        monitor_minibatch_indices = valid_minibatch_indices
                    elif monitor_indices == "train":
                        monitor_minibatch_indices = train_minibatch_indices
                    elif monitor_indices == "full":
                        monitor_minibatch_indices = train_minibatch_indices
                        monitor_minibatch_indices += valid_minibatch_indices
                    else:
                        raise ValueError("monitor_indices %s not supported" %
                                         monitor_indices)
                    for mi in monitor_minibatch_indices:
                        monitor_results = _apply_function_over_minibatch(
                            monitor_function, list_of_minibatch_args,
                            list_of_minibatch_functions, mi)
                print("Computing validation at update %i" % last_update_count)
                valid_results = defaultdict(list)
                for minibatch_count, mi in enumerate(valid_minibatch_indices):
                    valid_minibatch_results = _apply_function_over_minibatch(
                        valid_function, list_of_minibatch_args,
                        list_of_minibatch_functions, mi)
                    valid_results[valid_output_name] += valid_minibatch_results

                # Monitoring output
                output = {r: np.mean(results[r]) for r in results.keys()}
                output["total_number_of_samples_auto"] = new_sample_count
                output["total_number_of_updates_auto"] = new_update_count
                output["current_patience_auto"] = patience
                valid_cost = np.mean(valid_results[valid_output_name])
                epoch_stop = time.time()
                output["minibatch_size_auto"] = minibatch_size
                output["train_minibatch_count_auto"] = len(
                    train_minibatch_indices)
                output["valid_minibatch_count_auto"] = len(
                    valid_minibatch_indices)
                output["train_sample_count_auto"] = len(train_indices)
                output["valid_sample_count_auto"] = len(valid_indices)
                output["start_time_s_auto"] = global_start
                output["this_epoch_time_s_auto"] = epoch_stop - epoch_start
                output["total_number_of_epochs_auto"] = new_epoch_count
                for k in output.keys():
                    previous_results[k].append(output[k])

                if checkpoint_dict is not None:
                    # Quick trick to avoid 0 length list
                    old = min(previous_results[valid_output_name] + [np.inf])
                    previous_results[valid_output_name].append(valid_cost)
                    new = min(previous_results[valid_output_name])
                    if new < old:
                        print("Saving checkpoint based on validation score")
                        checkpoint_status_func(checkpoint_dict,
                                               previous_results,
                                               append_name="best")
                        _save_best_functions(train_function, valid_function,
                                             optimizer_object)
                        _save_best_results(previous_results)
                        if new < old * patience_improvement:
                            patience = max(patience,
                                           new_sample_count * patience_increase)
                    else:
                        checkpoint_status_func(checkpoint_dict,
                                               previous_results)
                results = _init_results_dict()
            last_update_count = new_update_count
            last_sample_count = new_sample_count
            if last_sample_count > patience:
                done_looping = True
        last_epoch_count = new_epoch_count
    return previous_results


def fixed_n_epochs_trainer(train_function, valid_function,
                           train_indices, valid_indices,
                           checkpoint_dict,
                           list_of_minibatch_args, minibatch_size,
                           monitor_function=None,
                           monitor_indices="valid",
                           list_of_minibatch_functions=[make_minibatch],
                           list_of_train_output_names=None,
                           valid_output_name=None,
                           valid_frequency="valid_length",
                           n_epochs=1000,
                           optimizer_object=None,
                           previous_results=None,
                           shuffle=False, random_state=None,
                           verbose=False):

    epoch_results = _iterate_function(
        train_function, valid_function,
        train_indices, valid_indices,
        list_of_minibatch_args, minibatch_size,
        monitor_function=monitor_function,
        monitor_indices=monitor_indices,
        checkpoint_dict=checkpoint_dict,
        list_of_minibatch_functions=list_of_minibatch_functions,
        list_of_train_output_names=list_of_train_output_names,
        valid_output_name=valid_output_name,
        valid_frequency=valid_frequency,
        n_epochs=n_epochs,
        optimizer_object=optimizer_object,
        patience_based_stopping=False,
        previous_results=previous_results,
        shuffle=shuffle,
        random_state=random_state,
        verbose=verbose)
    train_function, valid_function, opt = _load_best_functions()
    previous_results = _load_best_results()
    checkpoint_dict = load_last_checkpoint("best")
    return previous_results


def early_stopping_trainer(train_function, valid_function,
                           train_indices, valid_indices,
                           checkpoint_dict,
                           list_of_minibatch_args, minibatch_size,
                           monitor_function=None,
                           monitor_indices="valid",
                           list_of_minibatch_functions=[make_minibatch],
                           list_of_train_output_names=None,
                           valid_output_name=None,
                           valid_frequency="valid_length",
                           n_epochs=1000,
                           optimizer_object=None,
                           previous_results=None,
                           shuffle=False, random_state=None,
                           verbose=False):

    if optimizer_object is not None:
        n_halvings = 3
        assert hasattr(optimizer_object, 'learning_rate')
    else:
        n_halvings = 1

    if previous_results is None:
        previous_results = defaultdict(list)
    for i in range(n_halvings):
        epoch_results = _iterate_function(
            train_function, valid_function,
            train_indices, valid_indices,
            list_of_minibatch_args, minibatch_size,
            monitor_function=monitor_function,
            monitor_indices=monitor_indices,
            checkpoint_dict=checkpoint_dict,
            list_of_minibatch_functions=list_of_minibatch_functions,
            list_of_train_output_names=list_of_train_output_names,
            valid_output_name=valid_output_name,
            valid_frequency=valid_frequency,
            n_epochs=n_epochs,
            optimizer_object=optimizer_object,
            patience_based_stopping=True,
            previous_results=previous_results,
            shuffle=shuffle,
            random_state=random_state,
            verbose=verbose)
        if not _in_nosetest():
            train_function, valid_function, opt = _load_best_functions()
            if opt is not None:
                optimizer_object = opt
            previous_results = _load_best_results()
            checkpoint_dict = load_last_checkpoint("best")
            if optimizer_object is not None:
                old_lr = optimizer_object.learning_rate.get_value()
                if len(previous_results["learning_rate_auto"]) == 0:
                    previous_results["learning_rate_auto"] = [old_lr] * len(
                        previous_results["current_patience_auto"])
                else:
                    old_length = len(previous_results["learning_rate_auto"])
                    new_length = len(previous_results["current_patience_auto"])
                    previous_results["learning_rate_auto"] += [old_lr] * (
                        new_length - old_length)
                optimizer_object.learning_rate.set_value(old_lr / 2.)
            # final checkpoint
            checkpoint_status_func(checkpoint_dict, previous_results)
    return previous_results
