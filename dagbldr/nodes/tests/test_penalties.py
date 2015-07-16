from collections import OrderedDict
import theano

from dagbldr.datasets import load_digits
from dagbldr.utils import add_datasets_to_graph, convert_to_one_hot
from dagbldr.nodes import binary_crossentropy, binary_entropy
from dagbldr.nodes import categorical_crossentropy, abs_error
from dagbldr.nodes import squared_error, gaussian_error, log_gaussian_error
from dagbldr.nodes import masked_cost, gaussian_kl, gaussian_log_kl

# Common between tests
digits = load_digits()
X = digits["data"]
y = digits["target"]
n_classes = len(set(y))
y = convert_to_one_hot(y, n_classes)


def test_binary_crossentropy():
    graph = OrderedDict()
    X_sym = add_datasets_to_graph([X], ["X"], graph)
    cost = binary_crossentropy(.99 * X_sym, X_sym)
    theano.function([X_sym], cost, mode="FAST_COMPILE")


def test_binary_entropy():
    graph = OrderedDict()
    X_sym = add_datasets_to_graph([X], ["X"], graph)
    cost = binary_entropy(X_sym)
    theano.function([X_sym], cost, mode="FAST_COMPILE")


def test_categorical_crossentropy():
    graph = OrderedDict()
    y_sym = add_datasets_to_graph([y], ["y"], graph)
    cost = categorical_crossentropy(.99 * y_sym + .001, y_sym)
    theano.function([y_sym], cost, mode="FAST_COMPILE")


def test_abs_error():
    graph = OrderedDict()
    X_sym = add_datasets_to_graph([X], ["X"], graph)
    cost = abs_error(.5 * X_sym, X_sym)
    theano.function([X_sym], cost, mode="FAST_COMPILE")


def test_squared_error():
    graph = OrderedDict()
    X_sym = add_datasets_to_graph([X], ["X"], graph)
    cost = squared_error(.5 * X_sym, X_sym)
    theano.function([X_sym], cost, mode="FAST_COMPILE")


def test_gaussian_error():
    graph = OrderedDict()
    X_sym = add_datasets_to_graph([X], ["X"], graph)
    cost = gaussian_error(.5 * X_sym, (.5 * X_sym + .001) ** 2, X_sym)
    theano.function([X_sym], cost, mode="FAST_COMPILE")


def test_log_gaussian_error():
    graph = OrderedDict()
    X_sym = add_datasets_to_graph([X], ["X"], graph)
    cost = log_gaussian_error(.5 * X_sym, .5 * X_sym, X_sym)
    theano.function([X_sym], cost, mode="FAST_COMPILE")


def test_masked_cost():
    graph = OrderedDict()
    X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph)
    cost = gaussian_error(.5 * X_sym, .5 * X_sym, X_sym)
    masked = masked_cost(X_sym, y_sym)
    theano.function([X_sym, y_sym], [cost, masked], mode="FAST_COMPILE")


def test_gaussian_kl():
    graph = OrderedDict()
    X_sym = add_datasets_to_graph([X], ["X"], graph)
    fake_sigma = (.5 * X_sym + .001) ** 2
    kl = gaussian_kl([X_sym, X_sym], [fake_sigma, fake_sigma], graph,
                     'gaussian_kl')
    theano.function([X_sym], [kl], mode="FAST_COMPILE")


def test_gaussian_log_kl():
    graph = OrderedDict()
    X_sym = add_datasets_to_graph([X], ["X"], graph)
    kl = gaussian_log_kl([X_sym, X_sym], [X_sym, X_sym], graph,
                         'gaussian_log_kl')
    theano.function([X_sym], [kl], mode="FAST_COMPILE")
