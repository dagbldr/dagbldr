# Authors: Kyle Kastner
from __future__ import print_function
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
try:
    import Queue
except ImportError:
    import queue as Queue
try:
    import urllib.request as urllib  # for backwards compatibility
except ImportError:
    import urllib2 as urllib
try:
    import cPickle as pickle
except ImportError:
    import pickle
import heapq
import copy
import threading
import logging
import uuid
from collections import OrderedDict
import socket
import random
import os
import glob
import subprocess
import numpy as np
from itertools import cycle
import __main__ as main
import re
import shutil
import numbers
import theano
import sys
import warnings
import inspect
import zipfile
import time
import pprint
from collections import defaultdict
from functools import reduce
from ..externals import dill


logging.basicConfig(level=logging.INFO,
                    format='%(message)s')
logger = logging.getLogger(__name__)

string_f = StringIO()
ch = logging.StreamHandler(string_f)
# Automatically put the HTML break characters on there
formatter = logging.Formatter('%(message)s<br>')
ch.setFormatter(formatter)
logger.addHandler(ch)


def get_logger():
    """
    Fetch the global dagbldr logger.
    """
    return logger


FINALIZE_TRAINING = False


def _get_finalize_train():
    return FINALIZE_TRAINING


def _set_finalize_train():
    global FINALIZE_TRAINING
    FINALIZE_TRAINING = True


# Storage of internal shared
_lib_shared_params = OrderedDict()


def get_lib_shared_params():
    return _lib_shared_params


def get_name():
    base = str(uuid.uuid4())
    return base


def get_shared(name):
    if name in _lib_shared_params.keys():
        logger.info("Found name %s in shared parameters" % name)
        return _lib_shared_params[name]
    else:
        raise NameError("Name not found in shared params!")


def set_shared(name, variable):
    if name in _lib_shared_params.keys():
        raise ValueError("Trying to set key %s which already exists!" % name)
    _lib_shared_params[name] = variable


def del_shared():
    for key in _lib_shared_params.keys():
        del _lib_shared_params[key]


def get_params():
    """
    Returns {name: param}
    """
    params = OrderedDict()
    for name in _lib_shared_params.keys():
        params[name] = _lib_shared_params[name]
    return params


_type = "float32"


def get_type():
    return _type


# TODO: Fetch from env
NUM_SAVED_TO_KEEP = 2


# copied from utils to avoid circular deps
def safe_zip(*args):
    """Like zip, but ensures arguments are of same length.

       Borrowed from pylearn2 - copied from utils to avoid circular import
    """
    base = len(args[0])
    for i, arg in enumerate(args[1:]):
        if len(arg) != base:
            raise ValueError("Argument 0 has length %d but argument %d has "
                             "length %d" % (base, i+1, len(arg)))
    return zip(*args)


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


def in_nosetest():
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


def dpickle(save_path, pickle_item):
    """ Simple wrapper for checkpoint dictionaries """
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(40000)
    with open(save_path, mode="wb") as f:
        dill.dump(pickle_item, f, protocol=-1)
    sys.setrecursionlimit(old_recursion_limit)


def dunpickle(save_path):
    """ Simple pickle wrapper for checkpoint dictionaries """
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(40000)
    with open(save_path, mode="rb") as f:
        pickle_item = dill.load(f)
    sys.setrecursionlimit(old_recursion_limit)
    return pickle_item


def get_shared_variables_from_function(func):
    """
    Get all shared variables out of a compiled Theano function

    Parameters
    ----------
    func : theano function

    Returns
    -------
    shared_variables : list
        A list of theano shared variables
    """
    shared_variable_indices = [n for n, var in enumerate(func.maker.inputs)
                               if isinstance(var.variable,
                                             theano.compile.SharedVariable)]
    shared_variables = [func.maker.inputs[i].variable
                        for i in shared_variable_indices]
    return shared_variables


def get_values_from_function(func):
    """
    Get all shared values out of a compiled Theano function

    Parameters
    ----------
    func : theano function

    Returns
    -------
    list_of_values : list
        A list of numpy arrays
    """
    return [v.get_value() for v in get_shared_variables_from_function(func)]


def set_shared_variables_in_function(func, list_of_values):
    """
    Set all shared variables in a compiled Theano function

    Parameters
    ----------
    func : theano function

    list_of_values : list
        List of numpy arrays to add into shared variables
    """
    # TODO : Add checking that sizes are OK
    shared_variable_indices = [n for n, var in enumerate(func.maker.inputs)
                               if isinstance(var.variable,
                                             theano.compile.SharedVariable)]
    shared_variables = [func.maker.inputs[i].variable
                        for i in shared_variable_indices]
    [s.set_value(v) for s, v in safe_zip(shared_variables, list_of_values)]


def load_checkpoint(saved_checkpoint_path):
    """ Simple pickle wrapper for checkpoint dictionaries """
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(40000)
    with open(saved_checkpoint_path, mode="rb") as f:
        pickle_item = dill.load(f)
    sys.setrecursionlimit(old_recursion_limit)
    return pickle_item


def load_last_checkpoint(append_name=None):
    """ Simple pickle wrapper for checkpoint dictionaries """
    save_paths = glob.glob(os.path.join(get_checkpoint_dir(), "*.pkl"))
    save_paths = [s for s in save_paths if "results" not in s]
    if append_name is not None:
        save_paths = [s.split(append_name)[:-1] + s.split(append_name)[-1:]
                      for s in save_paths]
    sorted_paths = get_file_matches("*.pkl", "best")
    sorted_paths = [s for s in sorted_paths if "results" not in s]
    if len(sorted_paths) == 0:
        raise ValueError("No checkpoint found in %s" % get_checkpoint_dir())
    last_checkpoint_path = sorted_paths[-1]
    logger.info("Loading checkpoint from %s" % last_checkpoint_path)
    return load_checkpoint(last_checkpoint_path)


def write_results_as_html(results_dict, save_path, default_show="all"):
    as_html = filled_js_template_from_results_dict(
        results_dict, default_show=default_show)
    with open(save_path, "w") as f:
        f.writelines(as_html)


def get_file_matches(glob_ext, append_name):
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


def remove_old_files(sorted_files_list):
    n_saved_to_keep = NUM_SAVED_TO_KEEP
    if len(sorted_files_list) > n_saved_to_keep:
        times = [os.path.getctime(f) for f in sorted_files_list]
        times_rank = argsort(times)
        for t, f in zip(times_rank, sorted_files_list):
            if t not in range(0, len(times))[-n_saved_to_keep:]:
                os.remove(f)


def cleanup_monitors(partial_match, append_name=None):
    selected_monitors = get_file_matches(
        "*" + partial_match + "*.html", append_name)
    remove_old_files(selected_monitors)


def cleanup_checkpoints(append_name=None):
    selected_checkpoints = get_file_matches("*.pkl", append_name)
    remove_old_files(selected_checkpoints)
    selected_checkpoints = get_file_matches("*.npz", append_name)
    remove_old_files(selected_checkpoints)


def zip_dir(src, dst):
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


def archive_dagbldr():
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
        logger.info("Saving code archive %s at %s" % (lib_dir, save_lib_path))
        script_location = os.path.abspath(sys.argv[0])
        shutil.copy2(script_location, save_script_path)
        zip_dir(lib_dir, save_lib_path)


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
        logger.info(breakline)
        logger.info(fileline)
        logger.info(statusline)
        logger.info(breakline)
        logger.info(pp.pformat(last_results))
    if status_type == "checkpoint":
        save_path = os.path.join(get_checkpoint_dir(),
                                 "model_checkpoint_%i.html" % n_seen)

    if append_name is not None:
        split = save_path.split("_")
        save_path = "_".join(
            split[:-1] + [append_name] + split[-1:])
    if not in_nosetest():
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
        write_results_as_html(results_dict, save_path,
                              default_show=show_keys)
        if status_type == "checkpoint":
            cleanup_monitors("checkpoint", append_name)


def checkpoint_status_func(training_loop, results,
                           append_name=None, nan_check=True):
    """ Saves a checkpoint dict """
    checkpoint_dict = training_loop.checkpoint_dict
    checkpoint_dict["previous_results"] = results
    nan_test = [(k, True) for k, e_v in results.items()
                for v in e_v if np.isnan(v)]
    if nan_check and len(nan_test) > 0:
        nan_keys = set([tup[0] for tup in nan_test])
        raise ValueError("Found NaN values in the following keys ",
                         "%s, exiting training without saving" % nan_keys)

    n_seen = max([len(l) for l in results.values()])
    checkpoint_save_path = os.path.join(
        get_checkpoint_dir(), "model_checkpoint_%i.pkl" % n_seen)
    weight_save_path = os.path.join(
        get_checkpoint_dir(), "model_weights_%i.npz" % n_seen)
    results_save_path = os.path.join(
        get_checkpoint_dir(), "model_results_%i.pkl" % n_seen)
    if append_name is not None:
        def mkpath(name):
            split = name.split("_")
            return "_".join(split[:-1] + [append_name] + split[-1:])
        checkpoint_save_path = mkpath(checkpoint_save_path)
        weight_save_path = mkpath(weight_save_path)
        results_save_path = mkpath(results_save_path)
    if not in_nosetest():
        # Don't dump if testing!
        save_checkpoint(checkpoint_save_path, training_loop)
        save_weights(weight_save_path, checkpoint_dict)
        save_results(results_save_path, results)
        cleanup_checkpoints(append_name)
    monitor_status_func(results, append_name=append_name)


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

    if hasattr(arg, 'shape'):
        # should handle numpy arrays and hdf5
        if is_two_d:
            data = arg[slice_or_indices_list]
            mask = np.ones_like(data[:, 0]).astype(theano.config.floatX)
        else:
            data = arg[:, slice_or_indices_list]
            mask = np.ones_like(data[:, :, 0]).astype(theano.config.floatX)
        return [data, mask]

    if is_two_d:
        # list of lists
        d0 = [s.shape[0] for s in sliced]
        max_len = max(d0)
        batch_size = len(sliced)
        data = np.zeros((batch_size, max_len)).astype(sliced[0].dtype)
        mask = np.zeros((batch_size, max_len)).astype(theano.config.floatX)
        for n, s in enumerate(sliced):
            data[n, :len(s)] = s
            mask[n, :len(s)] = 1
    else:
        # list of arrays
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


def make_minibatch_from_indices(indices, minibatch_size):
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


def apply_function_over_minibatch(function, list_of_minibatch_args,
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


def save_best_functions(train_function, valid_function, optimizer_object=None,
                         fname="__functions.pkl"):
    if not in_nosetest():
        checkpoint_dir = get_checkpoint_dir()
        save_path = os.path.join(checkpoint_dir, fname)
        dpickle(save_path, {"train_function": train_function,
                           "valid_function": valid_function,
                           "optimizer_object": optimizer_object})


def load_best_functions(fname="__functions.pkl"):
    if not in_nosetest():
        checkpoint_dir = get_checkpoint_dir()
        save_path = os.path.join(checkpoint_dir, fname)
        chk = dunpickle(save_path)
        return (chk["train_function"], chk["valid_function"],
                chk["optimizer_object"])


def save_best_results(results, fname="__results.pkl"):
    if not in_nosetest():
        checkpoint_dir = get_checkpoint_dir()
        save_path = os.path.join(checkpoint_dir, fname)
        dpickle(save_path, results)


def load_best_results(fname="__results.pkl"):
    if not in_nosetest():
        checkpoint_dir = get_checkpoint_dir()
        save_path = os.path.join(checkpoint_dir, fname)
        return dunpickle(save_path)


def init_results_dict():
    results = defaultdict(list)
    results["total_number_of_epochs_auto"] = [0]
    return results


def plot_training_epochs(epochs_dict, plot_name, plot_limit=None,
                         turn_on_agg=True):
    # plot_limit can be a positive integer, negative integer, or float in 0 - 1
    # float between 0 and 1 assumed to be percentage of total to keep
    if turn_on_agg:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # colors from seaborn flatui
    color_list = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e",
                  "#2ecc71"]
    colors = cycle(color_list)
    for key in epochs_dict.keys():
        if plot_limit < 1 and plot_limit > 0:
            plot_limit = int(plot_limit * len(epochs_dict[key]))
        plt.plot(epochs_dict[key][:plot_limit], color=colors.next())
        plt.title(str(key))
        plt.savefig(plot_name + "_" + str(key) + ".png")
        plt.close()


def plot_images_as_subplots(list_of_plot_args, plot_name, width, height,
                            invert_y=False, invert_x=False,
                            figsize=None, turn_on_agg=True):
    if turn_on_agg:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lengths = [len(a) for a in list_of_plot_args]
    if len(list(filter(lambda x: x != lengths[0], lengths))) > 0:
        raise ValueError("list_of_plot_args has elements of different lengths!")

    if figsize is None:
        f, axarr = plt.subplots(lengths[0], len(lengths))
    else:
        f, axarr = plt.subplots(lengths[0], len(lengths), figsize=figsize)
    for n, v in enumerate(list_of_plot_args):
        for i, X_i in enumerate(v):
            axarr[i, n].matshow(X_i.reshape(width, height), cmap="gray",
                                interpolation="none")
            axarr[i, n].axis('off')
            if invert_y:
                axarr[i, n].set_ylim(axarr[i, n].get_ylim()[::-1])
            if invert_x:
                axarr[i, n].set_xlim(axarr[i, n].get_xlim()[::-1])
    plt.tight_layout()
    plt.savefig(plot_name + ".png")


def make_gif(arr, gif_name, plot_width, plot_height, resize_scale_width=5,
             resize_scale_height=5, list_text_per_frame=None, invert_y=False,
             invert_x=False, list_text_per_frame_color=None, delay=1,
             grayscale=False, loop=False, turn_on_agg=True):
    """ Make a gif from a series of pngs using matplotlib matshow """
    if turn_on_agg:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Plot temporaries for making gif
    # use random code to try and avoid deleting surprise files...
    random_code = random.randrange(2 ** 32)
    pre = str(random_code)
    for n, arr_i in enumerate(arr):
        plt.matshow(arr_i.reshape(plot_width, plot_height), cmap="gray",
                    interpolation="none")
        if invert_y:
            ax = plt.gca()
            ax.set_ylim(ax.get_ylim()[::-1])
        if invert_x:
            ax = plt.gca()
            ax.set_xlim(ax.get_xlim()[::-1])

        plt.axis('off')
        if list_text_per_frame is not None:
            text = list_text_per_frame[n]
            if list_text_per_frame_color is not None:
                color = list_text_per_frame_color[n]
            else:
                color = "white"
            plt.text(0, plot_height, text, color=color,
                     fontsize=2 * plot_height)
        # This looks rediculous but should count the number of digit places
        # also protects against multiple runs
        # plus 1 is to maintain proper ordering
        plotpath = '__%s_giftmp_%s.png' % (str(n).zfill(len(
            str(len(arr))) + 1), pre)
        plt.savefig(plotpath)
        plt.close()

    # make gif
    assert delay >= 1
    gif_delay = int(delay)
    basestr = "convert __*giftmp_%s.png -delay %s " % (pre, str(gif_delay))
    if loop:
        basestr += "-loop 1 "
    else:
        basestr += "-loop 0 "
    if grayscale:
        basestr += "-depth 8 -type Grayscale -depth 8 "
    basestr += "-resize %sx%s " % (str(int(resize_scale_width * plot_width)),
                                   str(int(resize_scale_height * plot_height)))
    basestr += gif_name
    print("Attempting gif")
    print(basestr)
    subprocess.call(basestr, shell=True)
    filelist = glob.glob("__*giftmp_%s.png" % pre)
    for f in filelist:
        os.remove(f)


def get_resource_dir(name, resource_dir=None, folder=None, create_dir=True):
    """ Get dataset directory path """
    if not resource_dir:
        resource_dir = os.getenv("DAGBLDR_MODELS", os.path.join(
            os.path.expanduser("~"), "dagbldr_models"))
    if folder is None:
        resource_dir = os.path.join(resource_dir, name)
    else:
        resource_dir = os.path.join(resource_dir, folder)
    if create_dir:
        if not os.path.exists(resource_dir):
            os.makedirs(resource_dir)
    return resource_dir


def get_script_name():
    script_path = os.path.abspath(sys.argv[0])
    # Assume it ends with .py ...
    script_name = script_path.split(os.sep)[-1]
    return script_name


def archive(tag=None):
    script_name = get_script_name()[:-3]
    save_path = get_resource_dir(script_name)
    if tag is None:
        save_script_path = os.path.join(save_path, get_script_name())
    else:
        save_script_path = os.path.join(save_path, tag + "_" + get_script_name())

    logger.info("Saving code archive for %s" % (save_path))
    script_location = os.path.abspath(sys.argv[0])
    shutil.copy2(script_location, save_script_path)

    lib_location = os.path.realpath(__file__)
    lib_name = lib_location.split(os.sep)[-1]
    if tag is None:
        save_lib_path = os.path.join(save_path, lib_name)
    else:
        save_lib_path = os.path.join(save_path, tag + "_" + lib_name)
    shutil.copy2(lib_location, save_lib_path)


def coroutine(func):
    def start(*args,**kwargs):
        cr = func(*args,**kwargs)
        cr.next()
        return cr
    return start


def save_weights(save_path, items_dict, use_resource_dir=True):
    logger.info("Not saving weights due to copy issues in npz")
    return
    weights_dict = {}
    # k is the function name, v is a theano function
    for k, v in items_dict.items():
        if isinstance(v, theano.compile.function_module.Function):
            # w is all the numpy values from a function
            w = get_values_from_function(v)
            for n, w_v in enumerate(w):
                weights_dict[k + "_%i" % n] = w_v
    if use_resource_dir:
        # Assume it ends with .py ...
        script_name = get_script_name()[:-3]
        save_path = os.path.join(get_resource_dir(script_name), save_path)
    logger.info("Saving weights to %s" % save_weights_path)
    if len(weights_dict.keys()) > 0:
        np.savez(save_path, **weights_dict)
    else:
        logger.info("Possible BUG: no theano functions found in items_dict, "
              "unable to save weights!")
    logger.info("Weight saving complete %s" % save_path)


def download(url, server_fname, local_fname=None, progress_update_percentage=5,
             bypass_certificate_check=False):
    """
    An internet download utility modified from
    http://stackoverflow.com/questions/22676/
    how-do-i-download-a-file-over-http-using-python/22776#22776
    """
    if bypass_certificate_check:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        u = urllib.urlopen(url, context=ctx)
    else:
        u = urllib.urlopen(url)
    if local_fname is None:
        local_fname = server_fname
    full_path = local_fname
    meta = u.info()
    with open(full_path, 'wb') as f:
        try:
            file_size = int(meta.get("Content-Length"))
        except TypeError:
            logger.info("WARNING: Cannot get file size, displaying bytes instead!")
            file_size = 100
        logger.info("Downloading: %s Bytes: %s" % (server_fname, file_size))
        file_size_dl = 0
        block_sz = int(1E7)
        p = 0
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            if (file_size_dl * 100. / file_size) > p:
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl *
                                               100. / file_size)
                logger.info(status)
                p += progress_update_percentage



def filled_js_template_from_results_dict(results_dict, default_show="all"):
    # Uses arbiter strings in the template to split the template and stick
    # values in
    partial_path = get_resource_dir("js_plot_dependencies")
    full_path = os.path.join(partial_path, "master.zip")
    url = "http://github.com/kastnerkyle/simple_template_plotter/archive/master.zip"
    if not os.path.exists(full_path):
        logger.info("Downloading plotter template code from %s" % url)
        download(url, full_path)
        zip_ref = zipfile.ZipFile(full_path, 'r')
        zip_ref.extractall(partial_path)
        zip_ref.close()

    js_path = os.path.join(partial_path, "simple_template_plotter-master")
    template_path =  os.path.join(js_path, "template.html")
    f = open(template_path, mode='r')
    all_template_lines = f.readlines()
    f.close()
    imports_split_index = [n for n, l in enumerate(all_template_lines)
                           if "IMPORTS_SPLIT" in l][0]
    data_split_index = [n for n, l in enumerate(all_template_lines)
                        if "DATA_SPLIT" in l][0]
    log_split_index = [n for n, l in enumerate(all_template_lines)
                       if "LOGGING_SPLIT" in l][0]
    first_part = all_template_lines[:imports_split_index]
    imports_part = []
    js_files_path = os.path.join(js_path, "js")
    js_file_names = ["jquery-1.9.1.js", "knockout-3.0.0.js",
                     "highcharts.js", "exporting.js"]
    js_files = [os.path.join(js_files_path, jsf) for jsf in js_file_names]
    for js_file in js_files:
        with open(js_file, "r") as f:
            imports_part.extend(
                ["<script>\n"] + f.readlines() + ["</script>\n"])
    post_imports_part = all_template_lines[
        imports_split_index + 1:data_split_index]
    log_part = all_template_lines[data_split_index + 1:log_split_index]
    last_part = all_template_lines[log_split_index + 1:]

    def gen_js_field_for_key_value(key, values, show=True):
        assert type(values) is list
        if isinstance(values[0], (np.generic, np.ndarray)):
            values = [float(v.ravel()) for v in values]
        maxlen = 1500
        if len(values) > maxlen:
            values = list(np.interp(np.linspace(0, len(values), maxlen),
                          np.arange(len(values)), values))
        show_key = "true" if show else "false"
        return "{\n    name: '%s',\n    data: %s,\n    visible: %s\n},\n" % (
            str(key), str(values), show_key)
    data_part = [gen_js_field_for_key_value(k, results_dict[k], True)
                 if k in default_show or default_show == "all"
                 else gen_js_field_for_key_value(k, results_dict[k], False)
                 for k in sorted(results_dict.keys())]
    all_filled_lines = first_part + imports_part + post_imports_part
    all_filled_lines = all_filled_lines + data_part + log_part
    # add logging output
    tmp = copy.copy(string_f)
    tmp.seek(0)
    log_output = tmp.readlines()
    del tmp
    all_filled_lines = all_filled_lines + log_output + last_part
    return all_filled_lines


def save_results_as_html(save_path, results_dict, use_resource_dir=True,
                         default_no_show="_auto"):
    show_keys = [k for k in results_dict.keys()
                 if default_no_show not in k]
    as_html = filled_js_template_from_results_dict(
        results_dict, default_show=show_keys)
    if use_resource_dir:
        # Assume it ends with .py ...
        script_name = get_script_name()[:-3]
        save_path = os.path.join(get_resource_dir(script_name), save_path)
    logger.info("Saving HTML results %s" % save_path)
    with open(save_path, "w") as f:
        f.writelines(as_html)
    logger.info("Completed HTML results saving %s" % save_path)


@coroutine
def threaded_html_writer(maxsize=25):
    """
    Expects to be sent a tuple of (save_path, results_dict)
    """
    messages = Queue.PriorityQueue(maxsize=maxsize)
    def run_thread():
        while True:
            p, item = messages.get()
            if item is GeneratorExit:
                return
            else:
                save_path, results_dict = item
                save_results_as_html(save_path, results_dict)
    threading.Thread(target=run_thread).start()
    try:
        n = 0
        while True:
            item = (yield)
            messages.put((n, item))
            n -= 1
    except GeneratorExit:
        messages.put((1, GeneratorExit))


@coroutine
def threaded_weights_writer(maxsize=25):
    """
    Expects to be sent a tuple of (save_path, checkpoint_dict)
    """
    messages = Queue.PriorityQueue(maxsize=maxsize)
    def run_thread():
        while True:
            p, item = messages.get()
            if item is GeneratorExit:
                return
            else:
                save_path, items_dict = item
                save_weights(save_path, items_dict)
    threading.Thread(target=run_thread).start()
    try:
        n = 0
        while True:
            item = (yield)
            messages.put((n, item))
            n -= 1
    except GeneratorExit:
        messages.put((1, GeneratorExit))


def save_checkpoint(save_path, pickle_item, use_resource_dir=True):
    if use_resource_dir:
        # Assume it ends with .py ...
        script_name = get_script_name()[:-3]
        save_path = os.path.join(get_resource_dir(script_name), save_path)
    sys.setrecursionlimit(40000)
    logger.info("Saving checkpoint to %s" % save_path)
    with open(save_path, mode="wb") as f:
        dill.dump(pickle_item, f, protocol=-1)
    logger.info("Checkpoint saving complete %s" % save_path)


@coroutine
def threaded_checkpoint_writer(maxsize=25):
    """
    Expects to be sent a tuple of (save_path, checkpoint_dict)
    """
    messages = Queue.PriorityQueue(maxsize=maxsize)
    def run_thread():
        while True:
            p, item = messages.get()
            if item is GeneratorExit:
                return
            else:
                save_path, pickle_item = item
                save_checkpoint(save_path, pickle_item)
    threading.Thread(target=run_thread).start()
    try:
        n = 0
        while True:
            item = (yield)
            messages.put((n, item))
            n -= 1
    except GeneratorExit:
        messages.put((1, GeneratorExit))


@coroutine
def threaded_timed_writer(sleep_time=15 * 60):
    """
    Expects to be sent a tuple of
    (objective,
    ((results_save_path, results_dict),
     (weights_save_path, checkpoint_dict),
     (checkpoint_save_path, checkpoint_dict)))

    Alternatively, pass None to bypass saving for that entry.

    (objective,
    ((results_save_path, results_dict),
     None,
     None))
    """
    messages = Queue.PriorityQueue()

    def run_thread():
        # always save the very first one
        last_time = time.time() - (sleep_time + 1)
        while True:
            if messages.qsize() > 5:
                mi = messages.queue.index(max(messages.queue))
                del messages.queue[mi]
                heapq.heapify(messages.queue)

            time_flag = (time.time() - last_time) > sleep_time
            # check if train loop has set FINALIZE_TRAINING
            # if so, write out the best one and exit
            train_flag = _get_finalize_train()
            if time_flag or train_flag:
                p, item = messages.get()
                last_time = time.time()
                if item is GeneratorExit:
                    return
                else:
                    results_tup, weights_tup, checkpoint_tup = item
                    if results_tup is not None:
                        save_path, results_dict = results_tup
                        save_results_as_html(save_path, results_dict)
                    if weights_tup is not None:
                        save_path, items_dict = weights_tup
                        save_weights(save_path, items_dict)
                    if checkpoint_tup is not None:
                        save_path, pickle_item = checkpoint_tup
                        save_checkpoint(save_path, pickle_item)
                    # write the last one if training is done
                    # but do not stop on a "results only" save
                    artifact_flag = checkpoint_tup is not None or weights_tup is not None
                    if train_flag and artifact_flag:
                        logger.info("Last checkpoint written, exiting save thread")
                        return

    threading.Thread(target=run_thread).start()
    try:
        last_best = np.inf
        n = -1
        while True:
            item = (yield)
            if item[0] < last_best:
                n = n - 1
                last_best = item[0]
                messages.put((n, item[1:]))
            else:
                messages.put((n + 1, item[1:]))
    except GeneratorExit:
        messages.put((1, GeneratorExit))


class TrainingLoop(object):
    """
    Runs the loop - thin wrapper for serializing

    checkpoint_every_n_epochs - useful for reducing disk writes when there are many epochs
    checkpoint_every_n_updates - useful for models where 1 epoch would have many updates
    checkpoint_every_n_seconds - useful for models where 1 epoch takes a long time
    write_every_n_seconds - the frequency at which the best checkpoint according to the train and valid objectives gets written

    monitor frequency
    skip_minimums - skip checkpoints based on minimum training/valid
    skip_intermediates - skip within epoch checkpoints
    skip_most_recents - skip writing most recent results html
    """
    def __init__(self, train_loop_function, train_function, train_itr,
                 valid_loop_function, valid_function, valid_itr,
                 n_epochs, checkpoint_dict,
                 checkpoint_delay=0,
                 checkpoint_every_n_epochs=1,
                 checkpoint_every_n_updates=np.inf,
                 checkpoint_every_n_seconds=np.inf,
                 write_every_n_seconds=15 * 60,
                 monitor_frequency=1000,
                 skip_minimums=False,
                 skip_intermediates=True,
                 skip_most_recents=False):
        self.train_loop_function = train_loop_function
        self.train_function = train_function
        self.train_itr = train_itr

        self.valid_loop_function = valid_loop_function
        self.valid_function = valid_function
        self.valid_itr = valid_itr

        self.n_epochs = n_epochs
        self.checkpoint_dict = checkpoint_dict

        # These parameters should be serialized
        self.checkpoint_delay = checkpoint_delay
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.checkpoint_every_n_updates = checkpoint_every_n_updates
        self.checkpoint_every_n_seconds = checkpoint_every_n_seconds
        self.write_every_n_seconds = write_every_n_seconds
        self.monitor_frequency = monitor_frequency
        self.skip_minimums = skip_minimums
        self.skip_intermediates = skip_intermediates
        self.skip_most_recents = skip_most_recents

        # tracker to ensure restarting at the correct minibatch
        self.num_train_minibatches_run = -1

    def __getstate__(self):
        skiplist = [self.train_loop_function,
                    self.train_function,
                    self.train_itr,
                    self.valid_loop_function,
                    self.valid_function,
                    self.valid_itr,
                    self.n_epochs,
                    self.checkpoint_dict]
        return {k:v for k, v in self.__dict__.items() if v not in skiplist}

    def refresh(self, train_loop_function, train_function, train_itr,
                valid_loop_function, valid_function, valid_itr,
                n_epochs,
                checkpoint_dict):
        # Must refresh after reloading from pkl
        self.train_loop_function = train_loop_function
        self.train_function = train_function
        self.train_itr = train_itr

        self.valid_loop_function = valid_loop_function
        self.valid_function = valid_function
        self.valid_itr = valid_itr
        self.n_epochs = n_epochs
        self.checkpoint_dict = checkpoint_dict

    def run(self):
        run_loop(self.train_loop_function, self.train_function, self.train_itr,
                 self.valid_loop_function, self.valid_function, self.valid_itr,
                 self.n_epochs,
                 self.checkpoint_dict,
                 self.checkpoint_delay,
                 self.checkpoint_every_n_epochs,
                 self.checkpoint_every_n_updates,
                 self.checkpoint_every_n_seconds,
                 self.write_every_n_seconds,
                 self.monitor_frequency,
                 self.skip_minimums,
                 self.skip_intermediates,
                 self.skip_most_recents,
                 self.num_train_minibatches_run,
                 self)


def run_loop(train_loop_function, train_function, train_itr,
             valid_loop_function, valid_function, valid_itr,
             n_epochs, checkpoint_dict,
             checkpoint_delay=10, checkpoint_every_n_epochs=1,
             checkpoint_every_n_updates=np.inf,
             checkpoint_every_n_seconds=10 * 60,
             write_every_n_seconds=15 * 60,
             monitor_frequency=1000, skip_minimums=False,
             skip_intermediates=True, skip_most_recents=False,
             skip_n_train_minibatches=-1,
             stateful_object=None):
    """
    TODO: add all logging info into the js report
    TODO: add upload fields to add data to an html and save a copy
    loop function should return a list of costs
    stateful_object allows to serialize and relaunch in middle of an epoch
    for long training models
    """
    logger.info("Running loop...")
    # Assume keys which are theano functions to ignore!
    ignore_keys = [k for k, v in checkpoint_dict.items()
                   if isinstance(v, theano.compile.function_module.Function)]

    train_loop = train_loop_function
    valid_loop = valid_loop_function
    ident = str(uuid.uuid4())[:8]
    random_state = np.random.RandomState(2177)
    monitor_prob = 1. / monitor_frequency

    non_ignored_keys = [k for k in checkpoint_dict.keys()
                        if k not in ignore_keys]
    if len(non_ignored_keys) > 0:
        overall_train_costs = checkpoint_dict["train_costs"]
        overall_valid_costs = checkpoint_dict["valid_costs"]
        # Auto tracking times
        overall_epoch_deltas = checkpoint_dict["epoch_deltas_auto"]
        overall_epoch_times = checkpoint_dict["epoch_times_auto"]
        overall_train_deltas = checkpoint_dict["train_deltas_auto"]
        overall_train_times = checkpoint_dict["train_times_auto"]
        overall_valid_deltas = checkpoint_dict["valid_deltas_auto"]
        overall_valid_times = checkpoint_dict["valid_times_auto"]
        overall_checkpoint_deltas = checkpoint_dict["checkpoint_deltas_auto"]
        overall_checkpoint_times = checkpoint_dict["checkpoint_times_auto"]
        overall_joint_deltas = checkpoint_dict["joint_deltas_auto"]
        overall_joint_times = checkpoint_dict["joint_times_auto"]
        overall_train_checkpoint = checkpoint_dict["train_checkpoint_auto"]
        overall_valid_checkpoint = checkpoint_dict["valid_checkpoint_auto"]
        keys_checked = ["train_costs",
                        "valid_costs",
                        "epoch_deltas_auto",
                        "epoch_times_auto",
                        "train_deltas_auto",
                        "train_times_auto",
                        "valid_deltas_auto",
                        "valid_times_auto",
                        "checkpoint_deltas_auto",
                        "checkpoint_times_auto",
                        "joint_deltas_auto",
                        "joint_times_auto",
                        "train_checkpoint_auto",
                        "valid_checkpoint_auto"]
        not_handled = [k for k in checkpoint_dict.keys()
                       if k not in keys_checked and k not in ignore_keys]
        if len(not_handled) > 0:
            raise ValueError("Unhandled keys %s in checkpoint_dict, exiting..." % not_handled)

        epoch_time_total = overall_epoch_times[-1]
        train_time_total = overall_train_times[-1]
        valid_time_total = overall_valid_times[-1]
        checkpoint_time_total = overall_checkpoint_times[-1]
        joint_time_total = overall_joint_times[-1]

        start_epoch = len(overall_train_costs)
    else:
        overall_train_costs = []
        overall_valid_costs = []
        overall_train_checkpoint = []
        overall_valid_checkpoint = []

        epoch_time_total = 0
        train_time_total = 0
        valid_time_total = 0
        checkpoint_time_total = 0
        joint_time_total = 0
        overall_epoch_times = []
        overall_epoch_deltas = []
        overall_train_times = []
        overall_train_deltas = []
        overall_valid_times = []
        overall_valid_deltas = []
        # Add zeros to avoid errors
        overall_checkpoint_times = [0]
        overall_checkpoint_deltas = [0]
        overall_joint_times = [0]
        overall_joint_deltas = [0]

        start_epoch = 0

    # save current state of lib and calling script
    archive_dagbldr()

    # Timed versus forced here
    tcw = threaded_timed_writer(write_every_n_seconds)
    vcw = threaded_timed_writer(write_every_n_seconds)
    fcw = threaded_timed_writer(sleep_time=0)

    best_train_checkpoint_pickle = None
    best_train_checkpoint_epoch = 0
    best_valid_checkpoint_pickle = None
    best_train_checkpoint_epoch = 0
    # If there are more than 1M minibatches per epoch this will break!
    # Not reallocating buffer greatly helps fast training models though
    # Also we have bigger problems if there are 1M minibatches per epoch...
    # This will get sliced down to the correct number of minibatches down below
    train_costs = [0.] * 1000000
    valid_costs = [0.] * 1000000
    try:
        for e in range(start_epoch, start_epoch + n_epochs):
            e_i = e + 1
            joint_start = time.time()
            epoch_start = time.time()
            logger.info(" ")
            logger.info("Starting training, epoch %i" % e_i)
            logger.info(" ")
            train_mb_count = 0
            valid_mb_count = 0
            results_dict = {k: v for k, v in checkpoint_dict.items()
                            if k not in ignore_keys}
            this_results_dict = results_dict
            try:
                # train loop
                train_start = time.time()
                last_time_checkpoint = train_start
                while True:
                    if train_mb_count < skip_n_train_minibatches:
                        train_mb_count += 1
                        continue
                    partial_train_costs = train_loop(train_function, train_itr)
                    train_costs[train_mb_count] = np.mean(partial_train_costs)
                    tc = train_costs[train_mb_count]
                    if np.isnan(tc):
                        logger.info("NaN detected in train cost, update %i" % train_mb_count)
                        raise StopIteration("NaN detected in train")

                    train_mb_count += 1
                    if (train_mb_count % checkpoint_every_n_updates) == 0:
                        checkpoint_save_path = "%s_model_update_checkpoint_%i.pkl" % (ident, train_mb_count)
                        weights_save_path = "%s_model_update_weights_%i.npz" % (ident, train_mb_count)
                        results_save_path = "%s_model_update_results_%i.html" % (ident, train_mb_count)
                        # Use pickle to preserve relationships between keys
                        # while still copying buffers
                        copy_pickle = pickle.dumps(checkpoint_dict)
                        copy_dict = pickle.loads(copy_pickle)

                        logger.info(" ")
                        logger.info("Update checkpoint after train mb %i" % train_mb_count)
                        logger.info("Current mean cost %f" % np.mean(partial_train_costs))
                        logger.info(" ")
                        this_results_dict["this_epoch_train_auto"] = train_costs[:train_mb_count]
                        tmb = train_costs[:train_mb_count]
                        running_train_mean = np.cumsum(tmb) / (np.arange(train_mb_count) + 1)
                        # needs to be a list
                        running_train_mean = list(running_train_mean)
                        this_results_dict["this_epoch_train_mean_auto"] = running_train_mean

                        objective = running_train_mean
                        tcw.send((objective,
                                  (results_save_path, this_results_dict),
                                  (checkpoint_save_path, copy_dict),
                                  (weights_save_path, copy_dict)))

                        if stateful_object is not None:
                            stateful_object.num_train_minibatches_run = train_mb_count
                            object_save_path = "%s_model_update_object_%i.pkl" % (ident, train_mb_count)
                            save_checkpoint(object_save_path, stateful_object)
                    elif (time.time() - last_time_checkpoint) >= checkpoint_every_n_seconds:
                        time_diff = time.time() - train_start
                        last_time_checkpoint = time.time()
                        checkpoint_save_path = "%s_model_time_checkpoint_%i.pkl" % (ident, int(time_diff))
                        weights_save_path = "%s_model_time_weights_%i.npz" % (ident, int(time_diff))
                        results_save_path = "%s_model_time_results_%i.html" % (ident, int(time_diff))
                        # Use pickle to preserve relationships between keys
                        # while still copying buffers
                        copy_pickle = pickle.dumps(checkpoint_dict)
                        copy_dict = pickle.loads(copy_pickle)

                        logger.info(" ")
                        logger.info("Time checkpoint after train mb %i" % train_mb_count)
                        logger.info("Current mean cost %f" % np.mean(partial_train_costs))
                        logger.info(" ")
                        this_results_dict["this_epoch_train_auto"] = train_costs[:train_mb_count]
                        tmb = train_costs[:train_mb_count]
                        running_train_mean = np.cumsum(tmb) / (np.arange(train_mb_count) + 1)
                        # needs to be a list
                        running_train_mean = list(running_train_mean)
                        this_results_dict["this_epoch_train_mean_auto"] = running_train_mean

                        objective = running_train_mean
                        tcw.send((objective,
                                  (results_save_path, this_results_dict),
                                  (checkpoint_save_path, copy_dict),
                                  (weights_save_path, copy_dict)))

                        if stateful_object is not None:
                            stateful_object.num_train_minibatches_run = train_mb_count
                            object_save_path = "%s_model_time_object_%i.pkl" % (ident, int(time_diff))
                            save_checkpoint(object_save_path, stateful_object)
                    draw = random_state.rand()
                    if draw < monitor_prob and not skip_intermediates:
                        logger.info(" ")
                        logger.info("Starting train mb %i" % train_mb_count)
                        logger.info("Current mean cost %f" % np.mean(partial_train_costs))
                        logger.info(" ")
                        results_save_path = "%s_intermediate_results.html" % ident
                        this_results_dict["this_epoch_train_auto"] = train_costs[:train_mb_count]

                        objective = np.mean(partial_train_costs)
                        fcw.send((objective,
                                  (results_save_path, this_results_dict),
                                  None,
                                  None))
            except StopIteration:
                # Slice so that only valid data is in the minibatch
                # this also assumes there is not a variable number
                # of minibatches in an epoch!
                train_stop = time.time()
                train_costs = train_costs[:train_mb_count]
                logger.info(" ")
                logger.info("Starting validation, epoch %i" % e_i)
                logger.info(" ")
                valid_start = time.time()
                try:
                    # Valid loop
                    while True:
                        partial_valid_costs = valid_loop(valid_function, valid_itr)
                        valid_costs[valid_mb_count] = np.mean(partial_valid_costs)
                        vc = valid_costs[valid_mb_count]
                        if np.isnan(vc):
                            logger.info("NaN detected in valid cost, minibatch %i" % valid_mb_count)
                            raise StopIteration("NaN detected in valid")
                        valid_mb_count += 1
                        draw = random_state.rand()
                        if draw < monitor_prob and not skip_intermediates:
                            logger.info(" ")
                            logger.info("Valid mb %i" % valid_mb_count)
                            logger.info("Current validation mean cost %f" % np.mean(
                                valid_costs))
                            logger.info(" ")
                            results_save_path = "%s_intermediate_results.html" % ident
                            this_results_dict["this_epoch_valid_auto"] = valid_costs[:valid_mb_count]

                            objective = np.mean(valid_costs)
                            fcw.send((objective,
                                     (results_save_path, this_results_dict),
                                     None,
                                     None))
                except StopIteration:
                    pass
                logger.info(" ")
                valid_stop = time.time()
                epoch_stop = time.time()
                valid_costs = valid_costs[:valid_mb_count]

                # Logging and tracking training statistics
                epoch_time_delta = epoch_stop - epoch_start
                epoch_time_total += epoch_time_delta
                overall_epoch_deltas.append(epoch_time_delta)
                overall_epoch_times.append(epoch_time_total)

                train_time_delta = train_stop - train_start
                train_time_total += train_time_delta
                overall_train_deltas.append(train_time_delta)
                overall_train_times.append(train_time_total)

                valid_time_delta = valid_stop - valid_start
                valid_time_total += valid_time_delta
                overall_valid_deltas.append(valid_time_delta)
                overall_valid_times.append(valid_time_total)

                mean_epoch_train_cost = np.mean(train_costs)
                # np.inf trick to avoid taking the min of length 0 list
                old_min_train_cost = min(overall_train_costs + [np.inf])
                if np.isnan(mean_epoch_train_cost):
                    logger.info("Previous train costs %s" % overall_train_costs[-5:])
                    logger.info("NaN detected in train cost, epoch %i" % e)
                    raise StopIteration("NaN detected in train")
                overall_train_costs.append(mean_epoch_train_cost)

                mean_epoch_valid_cost = np.mean(valid_costs)
                old_min_valid_cost = min(overall_valid_costs + [np.inf])
                if np.isnan(mean_epoch_valid_cost):
                    logger.info("Previous valid costs %s" % overall_valid_costs[-5:])
                    logger.info("NaN detected in valid cost, epoch %i" % e)
                    raise StopIteration("NaN detected in valid")
                overall_valid_costs.append(mean_epoch_valid_cost)

                if mean_epoch_train_cost < old_min_train_cost:
                    overall_train_checkpoint.append(mean_epoch_train_cost)
                else:
                    overall_train_checkpoint.append(old_min_train_cost)

                if mean_epoch_valid_cost < old_min_valid_cost:
                    overall_valid_checkpoint.append(mean_epoch_valid_cost)
                else:
                    overall_valid_checkpoint.append(old_min_valid_cost)

                checkpoint_dict["train_costs"] = overall_train_costs
                checkpoint_dict["valid_costs"] = overall_valid_costs
                # Auto tracking times
                checkpoint_dict["epoch_deltas_auto"] = overall_epoch_deltas
                checkpoint_dict["epoch_times_auto"] = overall_epoch_times
                checkpoint_dict["epoch_count_auto"] = list([i + 1 for i in range(len(overall_epoch_times))])

                checkpoint_dict["train_deltas_auto"] = overall_train_deltas
                checkpoint_dict["train_times_auto"] = overall_train_times

                checkpoint_dict["valid_deltas_auto"] = overall_valid_deltas
                checkpoint_dict["valid_times_auto"] = overall_valid_times

                checkpoint_dict["checkpoint_deltas_auto"] = overall_checkpoint_deltas
                checkpoint_dict["checkpoint_times_auto"] = overall_checkpoint_times

                checkpoint_dict["joint_deltas_auto"] = overall_joint_deltas
                checkpoint_dict["joint_times_auto"] = overall_joint_times

                # Tracking if checkpoints are made
                checkpoint_dict["train_checkpoint_auto"] = overall_train_checkpoint
                checkpoint_dict["valid_checkpoint_auto"] = overall_valid_checkpoint


                script = get_script_name()
                hostname = socket.gethostname()
                logger.info("Host %s, script %s" % (hostname, script))
                logger.info("Epoch %i complete" % e_i)
                logger.info("Epoch mean train cost %f" % mean_epoch_train_cost)
                logger.info("Epoch mean valid cost %f" % mean_epoch_valid_cost)
                logger.info("Previous train costs %s" % overall_train_costs[-5:])
                logger.info("Previous valid costs %s" % overall_valid_costs[-5:])

                results_dict = {k: v for k, v in checkpoint_dict.items()
                                if k not in ignore_keys}

                # Checkpointing part
                checkpoint_start = time.time()
                if e < checkpoint_delay or skip_minimums:
                    pass
                elif mean_epoch_valid_cost < old_min_valid_cost:
                    logger.info("Checkpointing valid...")
                    # Using dumps so relationship between keys in the pickle
                    # is preserved
                    checkpoint_save_path = "%s_model_checkpoint_valid_%i.pkl" % (ident, e_i)
                    weights_save_path = "%s_model_weights_valid_%i.npz" % (ident, e_i)
                    results_save_path = "%s_model_results_valid_%i.html" % (ident, e_i)
                    best_valid_checkpoint_pickle = pickle.dumps(checkpoint_dict)
                    best_valid_checkpoint_epoch = e
                    # preserve key relations
                    copy_dict = pickle.loads(best_valid_checkpoint_pickle)

                    objective = mean_epoch_valid_cost
                    vcw.send((objective,
                             (results_save_path, this_results_dict),
                             (checkpoint_save_path, copy_dict),
                             (weights_save_path, copy_dict)))

                    if mean_epoch_train_cost < old_min_train_cost:
                        checkpoint_save_path = "%s_model_checkpoint_train_%i.pkl" % (ident, e_i)
                        weights_save_path = "%s_model_weights_train_%i.npz" % (ident, e_i)
                        results_save_path = "%s_model_results_train_%i.html" % (ident, e_i)
                        best_train_checkpoint_pickle = pickle.dumps(checkpoint_dict)
                        best_train_checkpoint_epoch = e

                        objective = mean_epoch_train_cost
                        vcw.send((objective,
                                (results_save_path, this_results_dict),
                                (checkpoint_save_path, copy_dict),
                                (weights_save_path, copy_dict)))
                    logger.info("Valid checkpointing complete.")
                elif mean_epoch_train_cost < old_min_train_cost:
                    logger.info("Checkpointing train...")
                    checkpoint_save_path = "%s_model_checkpoint_train_%i.pkl" % (ident, e_i)
                    weights_save_path = "%s_model_weights_train_%i.npz" % (ident, e_i)
                    results_save_path = "%s_model_results_train_%i.html" % (ident, e_i)
                    best_train_checkpoint_pickle = pickle.dumps(checkpoint_dict)
                    best_train_checkpoint_epoch = e
                    # preserve key relations
                    copy_dict = pickle.loads(best_train_checkpoint_pickle)

                    objective = mean_epoch_train_cost
                    vcw.send((objective,
                             (results_save_path, this_results_dict),
                             (checkpoint_save_path, copy_dict),
                             (weights_save_path, copy_dict)))
                    logger.info("Train checkpointing complete.")

                if e < checkpoint_delay:
                    pass
                    # Don't skip force checkpoints after default delay
                    # Printing already happens above
                elif((e % checkpoint_every_n_epochs) == 0) or (e == (n_epochs - 1)):
                    logger.info("Checkpointing force...")
                    checkpoint_save_path = "%s_model_checkpoint_%i.pkl" % (ident, e_i)
                    weights_save_path = "%s_model_weights_%i.npz" % (ident, e_i)
                    results_save_path = "%s_model_results_%i.html" % (ident, e_i)
                    # Use pickle to preserve relationships between keys
                    # while still copying buffers
                    copy_pickle = pickle.dumps(checkpoint_dict)
                    copy_dict = pickle.loads(copy_pickle)

                    objective = mean_epoch_train_cost
                    fcw.send((objective,
                             (results_save_path, results_dict),
                             (weights_save_path, copy_dict),
                             (checkpoint_save_path, copy_dict)))
                    logger.info("Force checkpointing complete.")

                checkpoint_stop = time.time()
                joint_stop = time.time()

                if skip_most_recents:
                    pass
                else:
                    # Save latest
                    results_save_path = "%s_most_recent_results.html" % ident
                    objective = mean_epoch_train_cost
                    fcw.send((objective,
                             (results_save_path, results_dict),
                              None,
                              None))

                # Will show up next go around
                checkpoint_time_delta = checkpoint_stop - checkpoint_start
                checkpoint_time_total += checkpoint_time_delta
                overall_checkpoint_deltas.append(checkpoint_time_delta)
                overall_checkpoint_times.append(checkpoint_time_total)

                joint_time_delta = joint_stop - joint_start
                joint_time_total += joint_time_delta
                overall_joint_deltas.append(joint_time_delta)
                overall_joint_times.append(joint_time_total)
    except KeyboardInterrupt:
        logger.info("Training loop interrupted by user! Saving current best results.")

    if not skip_minimums:
        # Finalize saving best train and valid
        best_valid_checkpoint_dict = pickle.loads(best_valid_checkpoint_pickle)
        best_valid_results_dict = {k: v for k, v in best_valid_checkpoint_dict.items()
                                   if k not in ignore_keys}
        ee = best_valid_checkpoint_epoch
        checkpoint_save_path = "%s_model_checkpoint_valid_%i.pkl" % (ident, ee + 1)
        weights_save_path = "%s_model_weights_valid_%i.npz" % (ident, ee + 1)
        results_save_path = "%s_model_results_valid_%i.html" % (ident, ee + 1)

        objective = -np.inf
        fcw.send((objective,
                 (results_save_path, best_valid_results_dict),
                 (weights_save_path, best_valid_checkpoint_dict),
                 (checkpoint_save_path, best_valid_checkpoint_dict)))

        best_train_checkpoint_dict = pickle.loads(best_train_checkpoint_pickle)
        best_train_results_dict = {k: v for k, v in best_train_checkpoint_dict.items()
                                   if k not in ignore_keys}
        ee = best_train_checkpoint_epoch
        checkpoint_save_path = "%s_model_checkpoint_train_%i.pkl" % (ident, ee + 1)
        weights_save_path = "%s_model_weights_train_%i.npz" % (ident, ee + 1)
        results_save_path = "%s_model_results_train_%i.html" % (ident, ee + 1)

        objective = -np.inf
        fcw.send((objective,
                 (results_save_path, best_train_results_dict),
                 (weights_save_path, best_train_checkpoint_dict),
                 (checkpoint_save_path, best_train_checkpoint_dict)))

    logger.info("Loop finished, closing write threads (this may take a while!)")
    # set FINALIZE_TRAINING so that write threads know it is time to close
    _set_finalize_train()
    tcw.close()
    vcw.close()
    fcw.close()
