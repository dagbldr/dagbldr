# Authors: Kyle Kastner
from __future__ import print_function
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import logging
import uuid
from collections import OrderedDict

# Author: Kyle Kastner
# License: BSD 3-clause
import random
import os
import glob
import subprocess
import numpy as np
from itertools import cycle

# Author: Kyle Kastner
# License: BSD 3-clause
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


def pickle(save_path, pickle_item):
    """ Simple wrapper for checkpoint dictionaries """
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(40000)
    with open(save_path, mode="wb") as f:
        dill.dump(pickle_item, f, protocol=-1)
    sys.setrecursionlimit(old_recursion_limit)


def unpickle(save_path):
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


def save_weights(save_weights_path, items_dict):
    """ Save weights stored in functions contained in items_dict """
    logger.info("Saving weights to %s" % save_weights_path)
    weights_dict = {}
    # k is the function name, v is a theano function
    for k, v in items_dict.items():
        if isinstance(v, theano.compile.function_module.Function):
            # w is all the numpy values from a function
            w = get_values_from_function(v)
            for n, w_v in enumerate(w):
                weights_dict[k + "_%i" % n] = w_v
    if len(weights_dict.keys()) > 0:
        np.savez(save_weights_path, **weights_dict)
    else:
        logger.info("Possible BUG: no theano functions found in items_dict, "
                    "unable to save weights!")


def save_results(save_path, results):
    save_checkpoint(save_path, results)


def save_checkpoint(save_path, pickle_item):
    """ Simple wrapper for checkpoint dictionaries """
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(40000)
    with open(save_path, mode="wb") as f:
        dill.dump(pickle_item, f, protocol=-1)
    sys.setrecursionlimit(old_recursion_limit)


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


def old_checkpoint_status_func(checkpoint_dict, results,
                               append_name=None, nan_check=True):
    """ Saves a checkpoint dict """
    checkpoint_dict = checkpoint_dict
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
        save_checkpoint(checkpoint_save_path, checkpoint_dict)
        save_weights(weight_save_path, checkpoint_dict)
        save_results(results_save_path, results)
        cleanup_checkpoints(append_name)
    monitor_status_func(results, append_name=append_name)


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
        pickle(save_path, {"train_function": train_function,
                           "valid_function": valid_function,
                           "optimizer_object": optimizer_object})


def load_best_functions(fname="__functions.pkl"):
    if not in_nosetest():
        checkpoint_dir = get_checkpoint_dir()
        save_path = os.path.join(checkpoint_dir, fname)
        chk = unpickle(save_path)
        return (chk["train_function"], chk["valid_function"],
                chk["optimizer_object"])


def save_best_results(results, fname="__results.pkl"):
    if not in_nosetest():
        checkpoint_dir = get_checkpoint_dir()
        save_path = os.path.join(checkpoint_dir, fname)
        pickle(save_path, results)


def load_best_results(fname="__results.pkl"):
    if not in_nosetest():
        checkpoint_dir = get_checkpoint_dir()
        save_path = os.path.join(checkpoint_dir, fname)
        return unpickle(save_path)


def init_results_dict():
    results = defaultdict(list)
    results["total_number_of_epochs_auto"] = [0]
    return results


class TrainingLoop(object):
    def __init__(self, train_function, valid_function,
                 train_iterator, valid_iterator,
                 monitor_function=None,
                 monitor_iterator="valid",
                 checkpoint_dict=None,
                 list_of_train_output_names=None,
                 valid_output_name=None,
                 valid_frequency=100,
                 monitor_frequency=100,
                 n_epochs=100,
                 optimizer_object=None,
                 previous_results=None,
                 verbose=False):
        """
        Custom functions for train_function or valid_function *must* return
        a list!
        """
        self.train_function = train_function
        self.valid_function = valid_function
        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.monitor_function = monitor_function
        self.monitor_iterator = monitor_iterator
        self.checkpoint_dict = checkpoint_dict
        self.list_of_train_output_names = list_of_train_output_names
        self.valid_output_name = valid_output_name
        self.monitor_frequency = monitor_frequency
        self.valid_frequency = valid_frequency
        self.n_epochs = n_epochs
        self.optimizer_object = optimizer_object
        self.previous_results = previous_results
        self.verbose = verbose

    def run(self):
        return self._run()

    def _run(self):
        train_function = self.train_function
        valid_function = self.valid_function
        train_iterator = self.train_iterator
        valid_iterator = self.valid_iterator
        monitor_function = self.monitor_function
        monitor_iterator = self.monitor_iterator
        monitor_frequency = self.monitor_frequency
        checkpoint_dict = self.checkpoint_dict
        list_of_train_output_names = self.list_of_train_output_names
        valid_output_name = self.valid_output_name
        valid_frequency = self.valid_frequency
        n_epochs = self.n_epochs
        optimizer_object = self.optimizer_object
        previous_results = self.previous_results
        verbose = self.verbose
        if previous_results is not None:
            raise ValueError("previous_results argument no longer supported! "
                             "checkpoint_dict should contain this information.")

        if "previous_results" in checkpoint_dict.keys():
            previous_results = checkpoint_dict["previous_results"]
        else:
            logger.info("previous_results not found in checkpoint_dict.keys() "
                        "creating new storage for previous_results")
            previous_results = defaultdict(list)

        assert valid_frequency >= 1
        assert monitor_frequency >= 1
        train_minibatch_size = train_iterator.minibatch_size
        valid_minibatch_size = valid_iterator.minibatch_size

        if not in_nosetest():
            archive_dagbldr()
            # add calling commandline arguments here...

        if len(previous_results.keys()) != 0:
            last_epoch_count = previous_results[
                "total_number_of_epochs_auto"][-1]
        else:
            last_epoch_count = 0

        results = init_results_dict()
        total_train_minibatch_count = 0
        total_valid_minibatch_count = 0
        # print parameter info
        logger.info("-------------------")
        logger.info("Parameter name list")
        logger.info("-------------------")
        for key in get_params().keys():
            logger.info(key)
        logger.info("-------------------")
        global_start = time.time()
        previous_validation_time = time.time()
        for e in range(n_epochs):
            new_epoch_count = last_epoch_count + 1
            try:
                # Iterate through train minibatches until StopIteration
                while True:
                    list_of_train_args = next(train_iterator)
                    train_minibatch_results = train_function(
                        *list_of_train_args)
                    total_train_minibatch_count += 1
                    for n, k in enumerate(train_minibatch_results):
                        if list_of_train_output_names is not None:
                            assert len(list_of_train_output_names) == len(
                                train_minibatch_results)
                            results[list_of_train_output_names[n]].append(
                                train_minibatch_results[n])
                        else:
                            results[n].append(train_minibatch_results[n])

                    if total_train_minibatch_count % valid_frequency == 0:
                        logger.info("Computing validation at "
                              "minibatch %i" % total_train_minibatch_count)
                        valid_results = defaultdict(list)
                        try:
                            # Iterate through valid minibatches to StopIteration
                            while True:
                                list_of_valid_args = next(valid_iterator)
                                valid_minibatch_results = valid_function(
                                    *list_of_valid_args)
                                total_valid_minibatch_count += 1
                                valid_results[valid_output_name] += valid_minibatch_results
                        except StopIteration:
                            pass

                        # Monitoring output
                        output = {r: np.mean(results[r]) for r in results.keys()}
                        valid_cost = np.mean(valid_results[valid_output_name])
                        current_validation_time = time.time()

                        output["train_total_sample_count_auto"] = total_train_minibatch_count * train_minibatch_size
                        output["train_total_minibatch_count_auto"] = total_train_minibatch_count
                        output["train_minibatch_size_auto"] = train_minibatch_size

                        output["valid_total_sample_count_auto"] = total_valid_minibatch_count * valid_minibatch_size
                        output["valid_total_minibatch_count_auto"] = total_valid_minibatch_count
                        output["valid_minibatch_size_auto"] = valid_minibatch_size

                        output["start_time_s_auto"] = global_start
                        output["current_time_s_auto"] = current_validation_time
                        output["total_run_time_s_auto"] = current_validation_time - global_start

                        output["time_since_last_validation_s_auto"] = current_validation_time - previous_validation_time
                        previous_validation_time = current_validation_time

                        output["total_number_of_epochs_auto"] = new_epoch_count
                        for k in output.keys():
                            previous_results[k].append(output[k])

                        if checkpoint_dict is not None:
                            # Quick trick to avoid 0 length list
                            old = min(
                                previous_results[valid_output_name] + [np.inf])
                            previous_results[valid_output_name].append(valid_cost)
                            new = min(previous_results[valid_output_name])
                            if new < old:
                                logger.info("Saving checkpoint based on validation score")
                                checkpoint_status_func(self, previous_results,
                                                       append_name="best")
                                save_best_functions(train_function, valid_function,
                                                     optimizer_object)
                                save_best_results(previous_results)
                            else:
                                checkpoint_status_func(self, previous_results)
                        results = init_results_dict()

                    if total_train_minibatch_count % monitor_frequency == 0:
                        if monitor_function is not None:
                            logger.info("Running monitor at "
                                        "update %i" % total_train_minibatch_count)
                            if monitor_iterator == "valid":
                                # Iterate through monitor til StopIteration
                                try:
                                    while True:
                                        list_of_valid_args = next(
                                            valid_iterator)
                                        monitor_results = monitor_function(
                                            *list_of_valid_args)
                                except StopIteration:
                                    pass
                            else:
                                raise ValueError("Unhandled monitor_iterator")

            except StopIteration:
                last_epoch_count = new_epoch_count
        return previous_results


def get_js_path():
    module_path = os.path.dirname(__file__)
    js_path = os.path.join(module_path, "js_plot_dependencies")
    return js_path


def filled_js_template_from_results_dict(results_dict, default_show="all"):
    # Uses arbiter strings in the template to split the template and stick
    # values in
    js_path = get_js_path()
    template_path = os.path.join(js_path, "template.html")
    f = open(template_path, mode='r')
    all_template_lines = f.readlines()
    f.close()
    imports_split_index = [n for n, l in enumerate(all_template_lines)
                           if "IMPORTS_SPLIT" in l][0]
    data_split_index = [n for n, l in enumerate(all_template_lines)
                        if "DATA_SPLIT" in l][0]
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
    last_part = all_template_lines[data_split_index + 1:]

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
    all_filled_lines = all_filled_lines + data_part + last_part
    return all_filled_lines


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
