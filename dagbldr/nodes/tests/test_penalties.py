import theano
from theano import tensor

from dagbldr.datasets import load_digits
from dagbldr.utils import convert_to_one_hot
from dagbldr.nodes import binary_crossentropy, binary_entropy
from dagbldr.nodes import categorical_crossentropy, abs_error
from dagbldr.nodes import squared_error, gaussian_error, log_gaussian_error
from dagbldr.nodes import masked_cost, gaussian_kl, gaussian_log_kl

# Common between tests
digits = load_digits()
X = digits["data"].astype("float32")
y = digits["target"]
n_classes = len(set(y))
y = convert_to_one_hot(y, n_classes).astype("float32")
X_sym = tensor.fmatrix()
y_sym = tensor.fmatrix()


def test_binary_crossentropy():
    cost = binary_crossentropy(.99 * X_sym, X_sym)
    theano.function([X_sym], cost, mode="FAST_COMPILE")


def test_binary_entropy():
    cost = binary_entropy(X_sym)
    theano.function([X_sym], cost, mode="FAST_COMPILE")


def test_categorical_crossentropy():
    cost = categorical_crossentropy(.99 * y_sym + .001, y_sym)
    theano.function([y_sym], cost, mode="FAST_COMPILE")


def test_abs_error():
    cost = abs_error(.5 * X_sym, X_sym)
    theano.function([X_sym], cost, mode="FAST_COMPILE")


def test_squared_error():
    cost = squared_error(.5 * X_sym, X_sym)
    theano.function([X_sym], cost, mode="FAST_COMPILE")


def test_gaussian_error():
    cost = gaussian_error(.5 * X_sym, (.5 * X_sym + .001) ** 2, X_sym)
    theano.function([X_sym], cost, mode="FAST_COMPILE")


def test_log_gaussian_error():
    cost = log_gaussian_error(.5 * X_sym, .5 * X_sym, X_sym)
    theano.function([X_sym], cost, mode="FAST_COMPILE")


def test_masked_cost():
    cost = gaussian_error(.5 * X_sym, .5 * X_sym, X_sym)
    masked = masked_cost(X_sym, y_sym)
    theano.function([X_sym, y_sym], [cost, masked], mode="FAST_COMPILE")


def test_gaussian_kl():
    fake_sigma = (.5 * X_sym + .001) ** 2
    kl = gaussian_kl([X_sym, X_sym], [fake_sigma, fake_sigma])
    theano.function([X_sym], [kl], mode="FAST_COMPILE")


def test_gaussian_log_kl():
    kl = gaussian_log_kl([X_sym, X_sym], [X_sym, X_sym])
    theano.function([X_sym], [kl], mode="FAST_COMPILE")
