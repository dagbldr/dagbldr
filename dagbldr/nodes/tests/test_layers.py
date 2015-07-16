from collections import OrderedDict
import numpy as np
import theano
from theano import tensor
from nose.tools import assert_raises
from numpy.testing import assert_almost_equal

from dagbldr.utils import add_datasets_to_graph, convert_to_one_hot
from dagbldr.nodes import projection_layer, linear_layer, softmax_layer
from dagbldr.nodes import sigmoid_layer, tanh_layer, softplus_layer
from dagbldr.nodes import exp_layer, relu_layer, dropout_layer
from dagbldr.datasets import load_digits

# Common between tests
digits = load_digits()
X = digits["data"]
y = digits["target"]
n_classes = len(set(y))
y = convert_to_one_hot(y, n_classes)


def run_common_layer(layer):
    random_state = np.random.RandomState(42)
    graph = OrderedDict()
    X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph)
    single_o = layer([X_sym], graph, 'single', proj_dim=5,
                     random_state=random_state)
    concat_o = layer([X_sym, y_sym], graph, 'concat', proj_dim=5,
                     random_state=random_state)
    # Check that things can be reused
    repeated_o = layer([X_sym], graph, 'single', strict=False)

    # Check that strict mode raises an error if repeated
    assert_raises(AttributeError, layer, [X_sym], graph, 'concat')

    f = theano.function([X_sym, y_sym], [single_o, concat_o, repeated_o],
                        mode="FAST_COMPILE")
    single, concat, repeat = f(X, y)
    assert_almost_equal(single, repeat)


def test_dropout_layer():
    random_state = np.random.RandomState(42)
    graph = OrderedDict()
    X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph)
    on_off = tensor.iscalar()
    dropped = dropout_layer([X_sym], graph, 'dropout', on_off,
                            random_state=random_state)

    f = theano.function([X_sym, on_off], [dropped], mode="FAST_COMPILE")
    drop = f(np.ones_like(X), 1)[0]
    full = f(np.ones_like(X), 0)[0]
    # Make sure drop switch works
    assert_almost_equal((full.sum() / 2) / drop.sum(), 1., decimal=2)


def test_projection_layer():
    run_common_layer(projection_layer)


def test_linear_layer():
    run_common_layer(linear_layer)


def test_sigmoid_layer():
    run_common_layer(sigmoid_layer)


def test_tanh_layer():
    run_common_layer(tanh_layer)


def test_softplus_layer():
    run_common_layer(softplus_layer)


def test_exp_layer():
    run_common_layer(exp_layer)


def test_relu_layer():
    run_common_layer(relu_layer)


def test_softmax_layer():
    run_common_layer(softmax_layer)
