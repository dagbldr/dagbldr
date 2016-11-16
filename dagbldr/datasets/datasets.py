# Author: Kyle Kastner
# License: BSD 3-clause
# Ideas from Junyoung Chung and Kyunghyun Cho
# See https://github.com/jych/cle for a library in this style
import numpy as np
from collections import Counter
from scipy.io import loadmat, wavfile
from scipy.linalg import svd
from functools import reduce
from ..core import whitespace_tokenizer, safe_zip, get_logger
from .preprocessing_utils import stft
import shutil
import string
import tarfile
import fnmatch
import theano
import zipfile
import gzip
import os
import re
import csv
try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = get_logger()

regex = re.compile('[%s]' % re.escape(string.punctuation))

bitmap_characters = np.array([
    0x0,
    0x808080800080000,
    0x2828000000000000,
    0x287C287C280000,
    0x81E281C0A3C0800,
    0x6094681629060000,
    0x1C20201926190000,
    0x808000000000000,
    0x810202010080000,
    0x1008040408100000,
    0x2A1C3E1C2A000000,
    0x8083E08080000,
    0x81000,
    0x3C00000000,
    0x80000,
    0x204081020400000,
    0x1824424224180000,
    0x8180808081C0000,
    0x3C420418207E0000,
    0x3C420418423C0000,
    0x81828487C080000,
    0x7E407C02423C0000,
    0x3C407C42423C0000,
    0x7E04081020400000,
    0x3C423C42423C0000,
    0x3C42423E023C0000,
    0x80000080000,
    0x80000081000,
    0x6186018060000,
    0x7E007E000000,
    0x60180618600000,
    0x3844041800100000,
    0x3C449C945C201C,
    0x1818243C42420000,
    0x7844784444780000,
    0x3844808044380000,
    0x7844444444780000,
    0x7C407840407C0000,
    0x7C40784040400000,
    0x3844809C44380000,
    0x42427E4242420000,
    0x3E080808083E0000,
    0x1C04040444380000,
    0x4448507048440000,
    0x40404040407E0000,
    0x4163554941410000,
    0x4262524A46420000,
    0x1C222222221C0000,
    0x7844784040400000,
    0x1C222222221C0200,
    0x7844785048440000,
    0x1C22100C221C0000,
    0x7F08080808080000,
    0x42424242423C0000,
    0x8142422424180000,
    0x4141495563410000,
    0x4224181824420000,
    0x4122140808080000,
    0x7E040810207E0000,
    0x3820202020380000,
    0x4020100804020000,
    0x3808080808380000,
    0x1028000000000000,
    0x7E0000,
    0x1008000000000000,
    0x3C023E463A0000,
    0x40407C42625C0000,
    0x1C20201C0000,
    0x2023E42463A0000,
    0x3C427E403C0000,
    0x18103810100000,
    0x344C44340438,
    0x2020382424240000,
    0x800080808080000,
    0x800180808080870,
    0x20202428302C0000,
    0x1010101010180000,
    0x665A42420000,
    0x2E3222220000,
    0x3C42423C0000,
    0x5C62427C4040,
    0x3A46423E0202,
    0x2C3220200000,
    0x1C201804380000,
    0x103C1010180000,
    0x2222261A0000,
    0x424224180000,
    0x81815A660000,
    0x422418660000,
    0x422214081060,
    0x3C08103C0000,
    0x1C103030101C0000,
    0x808080808080800,
    0x38080C0C08380000,
    0x324C000000,
], dtype=np.uint64)

bitmap = np.unpackbits(bitmap_characters.view(np.uint8)).reshape(
    bitmap_characters.shape[0], 8, 8)
bitmap = bitmap[:, ::-1, :]
all_vocabulary_chars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTU"
all_vocabulary_chars += "VWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
char_mapping = {c: i for i, c in enumerate(all_vocabulary_chars)}


def string_to_character_index(string):
    return np.asarray([char_mapping[c] for c in string])


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


def _parse_stories(lines, only_supporting=False):
    """ Preprocessing code modified from Keras and Stephen Merity
    http://smerity.com/articles/2015/keras_qa.html
    https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py

    Parse stories provided in the bAbi tasks format

    If only_supporting is true, only the sentences that support the answer are
    kept.
    """
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = whitespace_tokenizer(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = whitespace_tokenizer(line)
            story.append(sent)
    return data


def _get_stories(f, only_supporting=False, max_length=None):
    """ Preprocessing code modified from Keras and Stephen Merity
    http://smerity.com/articles/2015/keras_qa.html
    https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py

    Given a file name, read the file, retrieve the stories, and then convert
    the sentences into a single story.

    If max_length is supplied, any stories longer than max_length tokens will be
    discarded.
    """
    data = _parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data
            if not max_length or len(flatten(story)) < max_length]
    return data


def _vectorize_stories(data, vocab_size, word_idx):
    """ Preprocessing code modified from Keras and Stephen Merity
    http://smerity.com/articles/2015/keras_qa.html
    https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py
    """
    X = []
    Xq = []
    y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        yi = np.zeros(vocab_size)
        yi[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        y.append(yi)
    return X, Xq, np.array(y)


def check_fetch_babi():
    """ Check for babi task data

    "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
    J. Weston, A. Bordes, S. Chopra, T. Mikolov, A. Rush
    http://arxiv.org/abs/1502.05698
    """
    url = "http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"
    partial_path = get_dataset_dir("babi")
    full_path = os.path.join(partial_path, "tasks_1-20_v1-2.tar.gz")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    return full_path


def fetch_babi(task_number=2):
    """ Fetch data for babi tasks described in
    "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
    J. Weston, A. Bordes, S. Chopra, T. Mikolov, A. Rush
    http://arxiv.org/abs/1502.05698

    Preprocessing code modified from Keras and Stephen Merity
    http://smerity.com/articles/2015/keras_qa.html
    https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py

    n_samples : 1000 - 10000 (task dependent)

    Returns
    -------
    summary : dict
        A dictionary cantaining data

        summary["stories"] : list
            List of list of ints

        summary["queries"] : list
            List of list of ints

        summary["target"] : list
            List of list of int

        summary["train_indices"] : array
            Indices for training samples

        summary["valid_indices"] : array
            Indices for validation samples

        summary["vocabulary_size"] : int
            Total vocabulary size
    """

    data_path = check_fetch_babi()
    tar = tarfile.open(data_path)
    if task_number == 2:
        challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
    else:
        raise ValueError("No other supported tasks at this time")
    # QA2 with 1000 samples
    train = _get_stories(tar.extractfile(challenge.format('train')))
    test = _get_stories(tar.extractfile(challenge.format('test')))

    vocab = sorted(reduce(lambda x, y: x | y, (
        set(story + q + [answer]) for story, q, answer in train + test)))
    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    # story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    # query_maxlen = max(map(len, (x for _, x, _ in train + test)))

    X_story, X_query, y_answer = _vectorize_stories(train, vocab_size, word_idx)
    valid_X_story, valid_X_query, valid_y_answer = _vectorize_stories(
        test, vocab_size, word_idx)
    train_indices = np.arange(len(y_answer))
    valid_indices = np.arange(len(valid_y_answer)) + len(y_answer)

    X_story, X_query, y_answer = _vectorize_stories(train + test, vocab_size,
                                                    word_idx)
    return {"stories": X_story,
            "queries": X_query,
            "target": y_answer,
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "vocabulary_size": vocab_size}


def check_fetch_fruitspeech():
    """ Check for fruitspeech data

    Recorded by Hakon Sandsmark
    """
    url = "https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz"
    partial_path = get_dataset_dir("fruitspeech")
    full_path = os.path.join(partial_path, "audio.tar.gz")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    audio_path = os.path.join(partial_path, "audio")
    if not os.path.exists(audio_path):
        tar = tarfile.open(full_path)
        os.chdir(partial_path)
        tar.extractall()
        tar.close()
    return audio_path


def fetch_fruitspeech():
    """ Check for fruitspeech data

    Recorded by Hakon Sandsmark

    Returns
    -------
    summary : dict
        A dictionary cantaining data

        summary["data"] : list
            List of list of ints

        summary["specgrams"] : list
            List of arrays in (n_frames, n_features) format

        summary["target_names"] : list
            List of strings

        summary["target"] : list
            List of list of int

        summary["train_indices"] : array
            Indices for training samples

        summary["valid_indices"] : array
            Indices for validation samples

        summary["vocabulary_size"] : int
            Total vocabulary size

        summary["vocabulary"] : string
            The whole vocabulary as a string
    """

    data_path = check_fetch_fruitspeech()
    audio_matches = []
    for root, dirnames, filenames in os.walk(data_path):
        for filename in fnmatch.filter(filenames, '*.wav'):
            audio_matches.append(os.path.join(root, filename))
    all_chars = []
    all_words = []
    all_data = []
    all_specgram_data = []
    for wav_path in audio_matches:
        # Convert chars to int classes
        word = wav_path.split(os.sep)[-1][:-6]
        chars = string_to_character_index(word)
        fs, d = wavfile.read(wav_path)
        d = d.astype("int32")
        # Preprocessing from A. Graves "Towards End-to-End Speech
        # Recognition"
        Pxx = 10. * np.log10(np.abs(stft(d, fftsize=128))).astype(
            theano.config.floatX)
        all_data.append(d)
        all_specgram_data.append(Pxx)
        all_chars.append(chars)
        all_words.append(word)
    vocabulary_size = len(all_vocabulary_chars)
    # Shuffle data
    all_lists = list(safe_zip(all_data, all_specgram_data, all_chars,
                              all_words))
    random_state = np.random.RandomState(1999)
    random_state.shuffle(all_lists)
    all_data, all_specgram_data, all_chars, all_words = zip(*all_lists)
    wordset = list(set(all_words))
    train_matches = []
    valid_matches = []
    for w in wordset:
        matches = [n for n, i in enumerate(all_words) if i == w]
        # Hold out ~25% of the data, keeping some of every class
        train_matches.append(matches[:-4])
        valid_matches.append(matches[-4:])
    train_indices = np.array(sorted(
        [r for i in train_matches for r in i])).astype("int32")
    valid_indices = np.array(sorted(
        [r for i in valid_matches for r in i])).astype("int32")

    # reorganize into contiguous blocks
    def reorg(list_):
        ret = [list_[i] for i in train_indices] + [
            list_[i] for i in valid_indices]
        return np.asarray(ret)
    all_data = reorg(all_data)
    all_specgram_data = reorg(all_specgram_data)
    all_chars = reorg(all_chars)
    all_words = reorg(all_words)
    # after reorganizing finalize indices
    train_indices = np.arange(len(train_indices))
    valid_indices = np.arange(len(valid_indices)) + len(train_indices)
    return {"data": all_data,
            "specgrams": all_specgram_data,
            "target": all_chars,
            "target_names": all_words,
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "vocabulary_size": vocabulary_size,
            "vocabulary": all_vocabulary_chars}


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
    """ All the fiction text written by H. P. Lovecraft

    n_samples : 40363
    n_chars : 84 (Counting UNK, EOS)
    n_words : 26644 (Counting UNK)

    Returns
    -------
    summary : dict
        A dictionary cantaining data

        summary["data"] : list, shape (40363,)
            List of strings

        summary["words"] : list,
            List of strings

    """
    data_path = check_fetch_lovecraft()
    all_data = []
    all_words = Counter()
    with zipfile.ZipFile(data_path, "r") as f:
        for name in f.namelist():
            if ".txt" not in name:
                # Skip README
                continue
            data = f.read(name)
            data = data.split("\n")
            data = [l.strip() for l in data if l != ""]
            words = [w for l in data for w in regex.sub('', l.lower()).split(
                " ") if w != ""]
            all_data.extend(data)
            all_words.update(words)
    return {"data": all_data,
            "words": all_words.keys()}


def load_mountains():
    """
    H. P. Lovecraft's At The Mountains Of Madness

    Used for tests which need text data

    n_samples : 3575
    n_chars : 75 (Counting UNK, EOS)
    n_words : 6346 (Counting UNK)

    Returns
    -------
    summary : dict
        A dictionary cantaining data

        summary["data"] : list, shape (3575, )
            List of strings

        summary["words"] : list,

    """
    module_path = os.path.dirname(__file__)
    all_words = Counter()
    with open(os.path.join(module_path, 'data', 'mountains.txt')) as f:
        data = f.read()
        data = data.split("\n")
        data = [l.strip() for l in data if l != ""]
        words = [w for l in data for w in regex.sub('', l.lower()).split(
            " ") if l != ""]
        all_words.update(words)
    return {"data": data,
            "words": all_words.keys()}


def check_fetch_fer():
    """ Check that fer faces are downloaded """
    url = 'https://dl.dropboxusercontent.com/u/15378192/fer2013.tar.gz'
    partial_path = get_dataset_dir("fer")
    full_path = os.path.join(partial_path, "fer2013.tar.gz")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    return full_path


def fetch_fer():
    """
    Flattened 48x48 fer faces with pixel values in [0 - 1]

    n_samples : 35888
    n_features : 2304

    Returns
    -------
    summary : dict
        A dictionary cantaining data and image statistics.

        summary["data"] : array, shape (35888, 2304)
            The flattened data for FER

    """
    data_path = check_fetch_fer()
    t = tarfile.open(data_path, 'r')
    f = t.extractfile(t.getnames()[0])
    reader = csv.reader(f)
    valid_indices = 2 * 3859
    data = np.zeros((35888, 48 * 48), dtype="float32")
    target = np.zeros((35888,), dtype="int32")
    header = None
    for n, row in enumerate(reader):
        if n % 1000 == 0:
            print("Reading sample %i" % n)
        if n == 0:
            header = row
        else:
            target[n] = int(row[0])
            data[n] = np.array(map(float, row[1].split(" "))) / 255.
    train_indices = np.arange(23709)
    valid_indices = np.arange(23709, len(data))
    train_mean0 = data[train_indices].mean(axis=0)
    saved_pca_path = os.path.join(get_dataset_dir("fer"), "FER_PCA.npy")
    if not os.path.exists(saved_pca_path):
        print("Saved PCA not found for FER, computing...")
        U, S, V = svd(data[train_indices] - train_mean0, full_matrices=False)
        train_pca = V
        np.save(saved_pca_path, train_pca)
    else:
        train_pca = np.load(saved_pca_path)
    return {"data": data,
            "target": target,
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "mean0": train_mean0,
            "pca_matrix": train_pca}


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
    """
    Flattened 48x48 TFD faces with pixel values in [0 - 1]

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
    train_indices = np.arange(0, 90000)
    valid_indices = np.arange(0, 10000) + len(train_indices) + 1
    test_indices = np.arange(valid_indices[-1] + 1, len(all_data))
    train_data = all_data[train_indices]
    train_mean0 = train_data.mean(axis=0)
    random_state = np.random.RandomState(1999)
    subset_indices = random_state.choice(train_indices, 25000, replace=False)
    saved_pca_path = os.path.join(get_dataset_dir("tfd"), "TFD_PCA.npy")
    if not os.path.exists(saved_pca_path):
        print("Saved PCA not found for TFD, computing...")
        U, S, V = svd(train_data[subset_indices] - train_mean0,
                      full_matrices=False)
        train_pca = V
        np.save(saved_pca_path, train_pca)
    else:
        train_pca = np.load(saved_pca_path)
    return {"data": all_data,
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "test_indices": test_indices,
            "mean0": train_mean0,
            "pca_matrix": train_pca}


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
    """
    Flattened 20x28 frey faces with pixel values in [0 - 1]

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
            "var0": all_data.var(axis=0)}


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
    """
    Flattened 28x28 mnist digits with pixel values in [0 - 1]

    n_samples : 70000
    n_feature : 784

    Returns
    -------
    summary : dict
        A dictionary cantaining data and image statistics.

        summary["data"] : array, shape (70000, 784)
        summary["target"] : array, shape (70000,)
        summary["images"] : array, shape (70000, 1, 28, 28)
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
    data = np.concatenate((train_set[0], valid_set[0], test_set[0]),
                          axis=0).astype(theano.config.floatX)
    target = np.concatenate((train_set[1], valid_set[1], test_set[1]),
                            axis=0).astype(np.int32)
    return {"data": data,
            "target": target,
            "images": data.reshape((len(data), 1, 28, 28)),
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
    """
    Flattened 28x28 mnist digits with pixel of either 0 or 1, sampled from
    binomial distribution defined by the original MNIST values

    n_samples : 70000
    n_features : 784

    Returns
    -------
    summary : dict
        A dictionary cantaining data and image statistics.

        summary["data"] : array, shape (70000, 784)
        summary["target"] : array, shape (70000,)
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


def make_sincos(n_timesteps, n_pairs):
    """
    Generate a 2D array of sine and cosine pairs at random frequencies and
    linear phase offsets depending on position in minibatch.

    Used for simple testing of RNN algorithms.

    Parameters
    ----------
    n_timesteps : int
        number of timesteps

    n_pairs : int
        number of sine, cosine pairs to generate

    Returns
    -------
    pairs : array, shape (n_timesteps, n_pairs, 2)
        A minibatch of sine, cosine pairs with the RNN minibatch converntion
        (timestep, sample, feature).
    """
    n_timesteps = int(n_timesteps)
    n_pairs = int(n_pairs)
    random_state = np.random.RandomState(1999)
    frequencies = 5 * random_state.rand(n_pairs) + 1
    frequency_base = np.arange(n_timesteps) / (2 * np.pi)
    steps = frequency_base[:, None] * frequencies[None]
    phase_offset = np.arange(n_pairs) / (2 * np.pi)
    sines = np.sin(steps + phase_offset)
    cosines = np.sin(steps + phase_offset + np.pi / 2)
    sines = sines[:, :, None]
    cosines = cosines[:, :, None]
    pairs = np.concatenate((sines, cosines), axis=-1).astype(
        theano.config.floatX)
    return pairs


def load_iris():
    """
    Load and return the iris dataset (classification).

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
        data = np.empty((n_samples, n_features), dtype=theano.config.floatX)
        target = np.empty((n_samples,), dtype=np.int32)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=theano.config.floatX)
            target[i] = np.asarray(ir[-1], dtype=np.int32)

    return {"data": data, "target": target}


def load_digits():
    """
    Load and return the digits dataset (classification).

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
    target = data[:, -1].astype("int32")
    flat_data = data[:, :-1].astype(theano.config.floatX)
    return {"data": flat_data, "target": target}


def make_ocr(list_of_strings):
    """
    Create an optical character recognition (OCR) dataset from a list of strings

    n_steps : variable
    n_samples : len(list_of_strings)
    n_features : 8

    Returns
    -------
    summary : dict
        A dictionary containing dataset information

        summary["data"] : array, shape (n_steps, n_samples, 8)
           Array containing list_of_strings, converted to bitmap images

        summary["target"] : array, shape (n_samples, )
            Array of variable length arrays, containing character indices for
            strings in list_of_strings

        summary["train_indices"] : array, shape (n_samples, )
            Indices array of the same length as summary["data"]

        summary["vocabulary"] : string
           All possible character labels as one long string

        summary["vocabulary_size"] : int
           len(summary["vocabulary"])

        summary["target_names"] : list
           list_of_strings stored for ease-of-access

    Notes
    -----
    Much of these bitmap utils modified from Shawn Tan

    https://github.com/shawntan/theano-ctc/
    """
    def string_to_bitmap(string):
        return np.hstack(np.array(
            [bitmap[char_mapping[c]] for c in string])).T[:, ::-1]

    data = []
    target = []
    for n, s in enumerate(list_of_strings):
        X_n = string_to_bitmap(s)
        y_n = string_to_character_index(s)
        data.append(X_n)
        target.append(y_n)
    data = np.asarray(data).transpose(1, 0, 2)
    target = np.asarray(target)
    return {"data": data, "target": target,
            "train_indices": np.arange(len(list_of_strings)),
            "vocabulary": all_vocabulary_chars,
            "vocabulary_size": len(all_vocabulary_chars),
            "target_names": list_of_strings}


def check_fetch_iamondb():
    """ Check for IAMONDB data

        This dataset cannot be downloaded automatically!
    """
    partial_path = get_dataset_dir("iamondb")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    ascii_path = os.path.join(partial_path, "lineStrokes-all.tar.gz")
    lines_path = os.path.join(partial_path, "ascii-all.tar.gz")
    files_path = os.path.join(partial_path, "task1.tar.gz")
    for p in [ascii_path, lines_path, files_path]:
        if not os.path.exists(p):
            files = "lineStrokes-all.tar.gz, ascii-all.tar.gz, and task1.tar.gz"
            url = "http://www.iam.unibe.ch/fki/databases/"
            url += "iam-on-line-handwriting-database/"
            url += "download-the-iam-on-line-handwriting-database"
            err = "Path %s does not exist!" % p
            err += " Download the %s files from %s" % (files, url)
            err += " and place them in the directory %s" % partial_path
            raise ValueError(err)
    return partial_path


def fetch_iamondb():
    from lxml import etree
    partial_path = check_fetch_iamondb()
    pickle_path = os.path.join(partial_path, "iamondb_saved.pkl")
    if not os.path.exists(pickle_path):
        files_path = os.path.join(partial_path, "task1.tar.gz")

        with tarfile.open(files_path) as tf:
            train_file = [fname for fname in tf.getnames()
                          if "trainset" in fname][0]

            def _s(lines):
                return [l.strip().decode("utf-8") for l in lines]

            f = tf.extractfile(train_file)
            train_names = _s(f.readlines())

            valid_files = [fname for fname in tf.getnames()
                           if "testset" in fname]
            valid_names = []
            for v in valid_files:
                f = tf.extractfile(v)
                valid_names.extend(_s(f.readlines()))

        strokes_path = os.path.join(partial_path, "lineStrokes-all.tar.gz")
        ascii_path = os.path.join(partial_path, "ascii-all.tar.gz")
        lsf = tarfile.open(strokes_path)
        af = tarfile.open(ascii_path)
        sf = [fs for fs in lsf.getnames() if ".xml" in fs]

        def construct_ascii_path(f):
            primary_dir = f.split("-")[0]
            if f[-1].isalpha():
                sub_dir = f[:-1]
            else:
                sub_dir = f
            file_path = os.path.join("ascii", primary_dir, sub_dir, f + ".txt")
            return file_path

        def construct_stroke_paths(f):
            primary_dir = f.split("-")[0]
            if f[-1].isalpha():
                sub_dir = f[:-1]
            else:
                sub_dir = f
            files_path = os.path.join("lineStrokes", primary_dir, sub_dir)

            # Dash is crucial to obtain correct match!
            files = [sif for sif in sf if f in sif]
            files = sorted(files, key=lambda x: int(
                x.split(os.sep)[-1].split("-")[-1][:-4]))
            return files

        train_ascii_files = [construct_ascii_path(fta) for fta in train_names]
        valid_ascii_files = [construct_ascii_path(fva) for fva in valid_names]
        train_stroke_files = [construct_stroke_paths(fts)
                              for fts in train_names]
        valid_stroke_files = [construct_stroke_paths(fvs)
                              for fvs in valid_names]

        train_set_files = list(zip(train_stroke_files, train_ascii_files))
        valid_set_files = list(zip(valid_stroke_files, valid_ascii_files))

        dataset_storage = {}
        x_set = []
        y_set = []
        char_set = []
        for sn, se in enumerate([train_set_files, valid_set_files]):
            for n, (strokes_files, ascii_file) in enumerate(se):
                if n % 100 == 0:
                    print("Processing file %i of %i" % (n, len(se)))
                fp = af.extractfile(ascii_file)
                cleaned = [t.strip().decode("utf-8") for t in fp.readlines()
                           if t != '\r\n'
                           and t != ' \n'
                           and t != '\n'
                           and t != ' \r\n']

                # Try using CSR
                idx = [w for w, li in enumerate(cleaned) if li == "CSR:"][0]
                cleaned_sub = cleaned[idx + 1:]
                corrected_sub = []

                for li in cleaned_sub:
                    # Handle edge case with %%%%% meaning new line?
                    if "%" in li:
                        li2 = re.sub('\%\%+', '%', li).split("%")
                        li2 = ''.join([l.strip() for l in li2])
                        corrected_sub.append(li2)
                    else:
                        corrected_sub.append(li)
                corrected_sub = [c for c in corrected_sub if c != '']
                fp.close()

                n_one_hot = 57
                y = [np.zeros((len(li), n_one_hot), dtype='int16')
                     for li in corrected_sub]

                # A-Z, a-z, space, apostrophe, comma, period
                charset = list(range(65, 90 + 1)) + list(range(97, 122 + 1)) + [
                    32, 39, 44, 46]
                tmap = {k: w + 1 for w, k in enumerate(charset)}

                # 0 for UNK/other
                tmap[0] = 0

                def tokenize_ind(line):
                    t = [ord(c) if ord(c) in charset else 0 for c in line]
                    r = [tmap[i] for i in t]
                    return r

                for n, li in enumerate(corrected_sub):
                    y[n][np.arange(len(li)), tokenize_ind(li)] = 1

                x = []
                for stroke_file in strokes_files:
                    fp = lsf.extractfile(stroke_file)
                    tree = etree.parse(fp)
                    root = tree.getroot()
                    # Get all the values from the XML
                    # 0th index is stroke ID, will become up/down
                    s = np.array([[i, int(Point.attrib['x']),
                                   int(Point.attrib['y'])]
                                  for StrokeSet in root
                                  for i, Stroke in enumerate(StrokeSet)
                                  for Point in Stroke])

                    # flip y axis
                    s[:, 2] = -s[:, 2]

                    # Get end of stroke points
                    c = s[1:, 0] != s[:-1, 0]
                    ci = np.where(c == True)[0]
                    nci = np.where(c == False)[0]

                    # set pen down
                    s[0, 0] = 0
                    s[nci, 0] = 0

                    # set pen up
                    s[ci, 0] = 1
                    s[-1, 0] = 1
                    x.append(s)
                    fp.close()

                if len(x) != len(y):
                    x_t = np.vstack((x[-2], x[-1]))
                    x = x[:-2] + [x_t]

                if len(x) == len(y):
                    x_set.extend(x)
                    y_set.extend(y)
                    char_set.extend(corrected_sub)
                else:
                    print("Skipping %i, couldn't make x and y len match!" % n)
            if sn == 0:
                dataset_storage["train_indices"] = np.arange(len(x_set))
            elif sn == 1:
                offset = dataset_storage["train_indices"][-1] + 1
                dataset_storage["valid_indices"] = np.arange(offset, len(x_set))
                dataset_storage["data"] = np.array(x_set)
                dataset_storage["target"] = np.array(y_set)
                dataset_storage["target_phrases"] = char_set
                dataset_storage["vocabulary_size"] = n_one_hot
                c = "".join([chr(a) for a in [ord("-")] + charset])
                dataset_storage["vocabulary"] = c
            else:
                raise ValueError("Undefined number of files")
        f = open(pickle_path, "wb")
        pickle.dump(dataset_storage, f, -1)
        f.close()
    with open(pickle_path, "rb") as f:
        pickle_dict = pickle.load(f)
    return pickle_dict


def check_fetch_bach_chorales_music21():
    """ Move files into dagbldr dir, in case python is on nfs. """
    from music21 import corpus
    all_bach_paths = corpus.getComposer("bach")
    partial_path = get_dataset_dir("bach_chorales_music21")
    for path in all_bach_paths:
        if "riemenschneider" in path:
            continue
        filename = os.path.split(path)[-1]
        local_path = os.path.join(partial_path, filename)
        if not os.path.exists(local_path):
            shutil.copy2(path, local_path)
    return partial_path


def music_21_to_pitch_duration(p):
    """
    Takes in a Music21 score, and outputs two numpy arrays
    One for pitch
    One for duration
    """
    parts = []
    parts_times = []
    for i, pi in enumerate(p.parts):
        part = []
        part_time = []
        for n in pi.stream().flat.notesAndRests:
            if n.isRest:
                part.append(0)
            else:
                part.append(n.midi)
            part_time.append(n.duration.quarterLength)
        parts.append(part)
        parts_times.append(part_time)

    # Create a "block" of events and times
    cumulative_times = map(lambda x: list(np.cumsum(x)), parts_times)
    event_points = sorted(list(set(sum(cumulative_times, []))))
    maxlen = max(map(len, cumulative_times))
    # -1 marks invalid / unused
    part_block = np.zeros((len(p.parts), maxlen)).astype("int32") - 1
    ctime_block = np.zeros((len(p.parts), maxlen)).astype("float32") - 1
    time_block = np.zeros((len(p.parts), maxlen)).astype("float32") - 1
    # create numpy array for easier indexing
    for i in range(len(parts)):
        part_block[i, :len(parts[i])] = parts[i]
        ctime_block[i, :len(cumulative_times[i])] = cumulative_times[i]
        time_block[i, :len(parts_times[i])] = parts_times[i]

    event_block = np.zeros((len(p.parts), len(event_points))) - 1
    etime_block = np.zeros((len(p.parts), len(event_points))) - 1
    for i, e in enumerate(event_points):
        idx = zip(*np.where(ctime_block == e))
        for ix in idx:
            event_block[ix[0], i] = part_block[ix[0], ix[1]]
            etime_block[ix[0], i] = time_block[ix[0], ix[1]]
    return event_block, etime_block


def fetch_bach_chorales_music21():
    """
    Bach chorales, transposed to C major or C minor (depending on original key).
    Only contains chorales with 4 voices populated.
    Requires music21.

    n_timesteps : 34270
    n_features : 4
    n_classes : 12 (duration), 54 (pitch)

    Returns
    -------
    summary : dict
        A dictionary cantaining data and image statistics.

        summary["data_pitch"] : array, shape (34270, 4)
        summary["data_duration"] : array, shape (34270, 4)
        summary["pitch_list"] : list, len 54
        summary["duration_list"] : list, len 12
        summary["major_minor_split"] : int, 16963

    Can split the data to only have major or minor key songs.
    For major, summary["data_pitch"][:summary["major_minor_split"]]
    For minor, summary["data_pitch"][summary["major_minor_split"]:]
    The same operation works for duration.

    pitch_list and duration_list give the mapping back from array value to
    actual data value.
    """

    from music21 import converter, interval, pitch
    data_path = check_fetch_bach_chorales_music21()
    pickle_path = os.path.join(data_path, "__processed_bach.pkl")
    if not os.path.exists(pickle_path):
        logger.info("Pickled file %s not found, creating. This may take a few minutes..." % pickle_path)
        all_transposed_bach_pitch = []
        all_transposed_bach_duration = []
        all_transposed_keys = []
        files = sorted(os.listdir(data_path))
        for n, f in enumerate(files):
            file_path = os.path.join(data_path, f)
            p = converter.parse(file_path)
            k = p.analyze("key")
            i = interval.Interval(k.tonic, pitch.Pitch("C"))
            p = p.transpose(i)
            k = p.analyze("key")
            try:
                pitches, durations = music_21_to_pitch_duration(p)
                if pitches.shape[0] != 4:
                    raise AttributeError("Too many voices, skipping...")
                all_transposed_bach_pitch.append(pitches)
                all_transposed_bach_duration.append(durations)
                all_transposed_keys.append(k.name)
            except AttributeError:
                # Random chord? skip it
                pass
            if n % 25 == 0:
                logger.info("Processed %s, progress %s / %s files complete" % (f, n + 1, len(files)))
        d = {"data_pitch": all_transposed_bach_pitch,
             "data_duration": all_transposed_bach_duration,
             "data_key": all_transposed_keys}
        with open(pickle_path, "wb") as f:
            logger.info("Saving pickle file %s" % pickle_path)
            pickle.dump(d, f, -1)
    else:
        with open(pickle_path, "rb") as f:
            d = pickle.load(f)

    major_pitch = []
    minor_pitch = []
    major_duration = []
    minor_duration = []
    for i in range(len(d["data_key"])):
        k = d["data_key"][i]
        ddp = d["data_pitch"][i]
        ddd = d["data_duration"][i]
        if k == "C major":
            major_pitch.append(ddp)
            major_duration.append(ddd)
        elif k == "C minor":
            minor_pitch.append(ddp)
            minor_duration.append(ddd)
        else:
            raise ValueError("Unknown key %s" % k)

    # now do preproc... and add things into the dict
    dp = np.concatenate(major_pitch + minor_pitch, axis=1)
    dd = np.concatenate(major_duration + minor_duration, axis=1)
    major_minor_split = sum([m.shape[1] for m in major_pitch])

    def replace_with_indices(arr):
        "Inplace but return reference"
        uniques = np.unique(arr)
        classes = np.arange(len(uniques))
        all_idx = [np.where(arr.ravel() == u)[0] for u in uniques]

        for n, (c, idx) in enumerate(zip(classes, all_idx)):
            arr.flat[idx] = float(n)
        return arr
    pitch_list = sorted(np.unique(dp))
    duration_list = sorted(np.unique(dd))

    dp = replace_with_indices(dp)
    dd = replace_with_indices(dd)
    d = {"data_pitch": dp.transpose()[:, ::-1],
         "data_duration": dd.transpose()[:, ::-1],
         "pitch_list": pitch_list,
         "duration_list": duration_list,
         "major_minor_split": major_minor_split}
    return d
