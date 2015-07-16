# Author: Kyle Kastner
# License: BSD 3-clause
# Ideas from Junyoung Chung and Kyunghyun Cho
# See https://github.com/jych/cle for a library in this style
import numpy as np
from scipy.io import loadmat
import theano
import zipfile
import gzip
import os
import csv
try:
    import cPickle as pickle
except ImportError:
    import pickle


def get_dataset_dir(dataset_name, data_dir=None, folder=None, create_dir=True):
    """ Get dataset directory path """
    if not data_dir:
        data_dir = os.getenv("DAGBLDR_DATA", os.path.join(
            os.path.expanduser("~"), "dagbldr_data"))
    if folder is None:
        data_dir = os.path.join(data_dir, dataset_name)
    else:
        data_dir = os.path.join(data_dir, folder)
    if not os.path.exists(data_dir) and create_dir:
        os.makedirs(data_dir)
    return data_dir


def download(url, server_fname, local_fname=None, progress_update_percentage=5):
    """
    An internet download utility modified from
    http://stackoverflow.com/questions/22676/
    how-do-i-download-a-file-over-http-using-python/22776#22776
    """
    try:
        import urllib
        urllib.urlretrieve('http://google.com')
    except AttributeError:
        import urllib.request as urllib
    u = urllib.urlopen(url)
    if local_fname is None:
        local_fname = server_fname
    full_path = local_fname
    meta = u.info()
    with open(full_path, 'wb') as f:
        try:
            file_size = int(meta.get("Content-Length"))
        except TypeError:
            print("WARNING: Cannot get file size, displaying bytes instead!")
            file_size = 100
        print("Downloading: %s Bytes: %s" % (server_fname, file_size))
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
                print(status)
                p += progress_update_percentage


def check_fetch_uci_words():
    """ Check for UCI vocabulary """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
    url += 'bag-of-words/'
    partial_path = get_dataset_dir("uci_words")
    full_path = os.path.join(partial_path, "uci_words.zip")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        # Download all 5 vocabularies and zip them into a file
        all_vocabs = ['vocab.enron.txt', 'vocab.kos.txt', 'vocab.nips.txt',
                      'vocab.nytimes.txt', 'vocab.pubmed.txt']
        for vocab in all_vocabs:
            dl_url = url + vocab
            download(dl_url, os.path.join(partial_path, vocab),
                     progress_update_percentage=1)

            def zipdir(path, zipf):
                # zipf is zipfile handle
                for root, dirs, files in os.walk(path):
                    for f in files:
                        if "vocab" in f:
                            zipf.write(os.path.join(root, f))

            zipf = zipfile.ZipFile(full_path, 'w')
            zipdir(partial_path, zipf)
            zipf.close()
    return full_path


def fetch_uci_words():
    """ Returns UCI vocabulary text. """
    data_path = check_fetch_uci_words()
    all_data = []
    with zipfile.ZipFile(data_path, "r") as f:
        for name in f.namelist():
            if ".txt" not in name:
                # Skip README
                continue
            data = f.read(name)
            data = data.split("\n")
            data = [l.strip() for l in data if l != ""]
            all_data.extend(data)
    return list(set(all_data))


def check_fetch_lovecraft():
    """ Check for lovecraft data """
    url = 'https://dl.dropboxusercontent.com/u/15378192/lovecraft_fiction.zip'
    partial_path = get_dataset_dir("lovecraft")
    full_path = os.path.join(partial_path, "lovecraft_fiction.zip")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    return full_path


def fetch_lovecraft():
    """ Returns lovecraft text. """
    data_path = check_fetch_lovecraft()
    all_data = []
    with zipfile.ZipFile(data_path, "r") as f:
        for name in f.namelist():
            if ".txt" not in name:
                # Skip README
                continue
            data = f.read(name)
            data = data.split("\n")
            data = [l.strip() for l in data if l != ""]
            all_data.extend(data)
    return all_data


def check_fetch_tfd():
    """ Check that tfd faces are downloaded """
    partial_path = get_dataset_dir("tfd")
    full_path = os.path.join(partial_path, "TFD_48x48.mat")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        raise ValueError("Put TFD_48x48 in %s" % str(partial_path))
    return full_path


def fetch_tfd():
    """ Flattened 48x48 TFD faces with pixel values in [0 - 1]

        n_samples : 102236
        n_features : 2304

        Returns
        -------
        summary : dict
            A dictionary cantaining data and image statistics.

            summary["data"] : array, shape (102236, 2304)
                The flattened data for TFD

    """
    data_path = check_fetch_tfd()
    matfile = loadmat(data_path)
    all_data = matfile['images'].reshape(len(matfile['images']), -1) / 255.
    all_data = all_data.astype(theano.config.floatX)
    return {"data": all_data,
            "mean0": all_data.mean(axis=0),
            "var0": all_data.var(axis=0),
            "mean1": all_data.mean(axis=1),
            "var1": all_data.var(axis=1),
            "mean": all_data.mean(),
            "var": all_data.var()}


def check_fetch_frey():
    """ Check that frey faces are downloaded """
    url = 'http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat'
    partial_path = get_dataset_dir("frey")
    full_path = os.path.join(partial_path, "frey_rawface.mat")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    return full_path


def fetch_frey():
    """ Flattened 20x28 frey faces with pixel values in [0 - 1]

        n_samples : 1965
        n_features : 560

        Returns
        -------
        summary : dict
            A dictionary cantaining data and image statistics.

            summary["data"] : array, shape (1965, 560)

    """
    data_path = check_fetch_frey()
    matfile = loadmat(data_path)
    all_data = (matfile['ff'] / 255.).T
    all_data = all_data.astype(theano.config.floatX)
    return {"data": all_data,
            "mean0": all_data.mean(axis=0),
            "var0": all_data.var(axis=0),
            "mean1": all_data.mean(axis=1),
            "var1": all_data.var(axis=1),
            "mean": all_data.mean(),
            "var": all_data.var()}


def check_fetch_mnist():
    """ Check that mnist is downloaded. May need fixing for py3 compat """
    # py3k version is available at mnist_py3k.pkl.gz ... might need to fix
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    partial_path = get_dataset_dir("mnist")
    full_path = os.path.join(partial_path, "mnist.pkl.gz")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    return full_path


def fetch_mnist():
    """ Flattened 28x28 mnist digits with pixel values in [0 - 1]

        n_samples : 70000
        n_feature : 784

        Returns
        -------
        summary : dict
            A dictionary cantaining data and image statistics.

            summary["data"] : array, shape (70000, 784)
            summary["targets"] : array, shape (70000,)
            summary["train_indices"] : array, shape (50000,)
            summary["valid_indices"] : array, shape (10000,)
            summary["test_indices"] : array, shape (10000,)

    """
    data_path = check_fetch_mnist()
    f = gzip.open(data_path, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()
    train_indices = np.arange(0, len(train_set[0]))
    valid_indices = np.arange(0, len(valid_set[0])) + train_indices[-1] + 1
    test_indices = np.arange(0, len(test_set[0])) + valid_indices[-1] + 1
    return {"data": np.concatenate((train_set[0], valid_set[0], test_set[0]),
                                   axis=0).astype(theano.config.floatX),
            "target": np.concatenate((train_set[1], valid_set[1], test_set[1]),
                                     axis=0).astype(np.int32),
            "train_indices": train_indices.astype(np.int32),
            "valid_indices": valid_indices.astype(np.int32),
            "test_indices": test_indices.astype(np.int32)}


def check_fetch_binarized_mnist():
    raise ValueError("Binarized MNIST has no labels! Do not use")
    """
    # public version
    url = 'https://github.com/mgermain/MADE/releases/download/ICML2015/'
    url += 'binarized_mnist.npz'
    partial_path = get_dataset_dir("binarized_mnist")
    fname = "binarized_mnist.npz"
    full_path = os.path.join(partial_path, fname)
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    # personal version
    url = "https://dl.dropboxusercontent.com/u/15378192/binarized_mnist_%s.npy"
    fname = "binarized_mnist_%s.npy"
    for s in ["train", "valid", "test"]:
        full_path = os.path.join(partial_path, fname % s)
        if not os.path.exists(partial_path):
            os.makedirs(partial_path)
        if not os.path.exists(full_path):
            download(url % s, full_path, progress_update_percentage=1)
    return partial_path
    """


def fetch_binarized_mnist():
    """ Flattened 28x28 mnist digits with pixel of either 0 or 1, sampled from
        binomial distribution defined by the original MNIST values

        n_samples : 70000
        n_features : 784

        Returns
        -------
        summary : dict
            A dictionary cantaining data and image statistics.

            summary["data"] : array, shape (70000, 784)
            summary["targets"] : array, shape (70000,)
            summary["train_indices"] : array, shape (50000,)
            summary["valid_indices"] : array, shape (10000,)
            summary["test_indices"] : array, shape (10000,)

    """
    mnist = fetch_mnist()
    random_state = np.random.RandomState(1999)

    def get_sampled(arr):
        # make sure that a pixel can always be turned off
        return random_state.binomial(1, arr * 255 / 256., size=arr.shape)

    data = get_sampled(mnist["data"]).astype(theano.config.floatX)
    return {"data": data,
            "target": mnist["target"],
            "train_indices": mnist["train_indices"],
            "valid_indices": mnist["valid_indices"],
            "test_indices": mnist["test_indices"]}


def load_iris():
    """Load and return the iris dataset (classification).

    This is basically the sklearn dataset loader, except returning a dictionary.

    n_samples : 150
    n_features : 4

    Returns
    -------
    summary : dict
        A dictionary cantaining data and target labels

        summary["data"] : array, shape (150, 4)
            The data for iris

        summary["target"] : array, shape (150,)
            The classification targets

    """
    module_path = os.path.dirname(__file__)
    with open(os.path.join(module_path, 'data', 'iris.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float)
            target[i] = np.asarray(ir[-1], dtype=np.int)

    return {"data": data, "target": target}


def load_digits():
    """Load and return the digits dataset (classification).

    This is basically the sklearn dataset loader, except returning a dictionary.

    n_samples : 1797
    n_features : 64

    Returns
    -------
    summary : dict
        A dictionary cantaining data and target labels

        summary["data"] : array, shape (1797, 64)
            The data for digits

        summary["target"] : array, shape (1797,)
            The classification targets

    """

    module_path = os.path.dirname(__file__)
    data = np.loadtxt(os.path.join(module_path, 'data', 'digits.csv.gz'),
                      delimiter=',')
    target = data[:, -1]
    flat_data = data[:, :-1]
    return {"data": flat_data, "target": target.astype(np.int)}
