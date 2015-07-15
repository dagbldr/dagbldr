from collections import OrderedDict
import numpy as np
import theano

from dagbldr.utils import add_datasets_to_graph, convert_to_one_hot
from dagbldr.utils import get_params_and_grads
from dagbldr.utils import iterate_function
from dagbldr.nodes import linear_layer, softmax_layer
from dagbldr.nodes import categorical_crossentropy_nll
from dagbldr.optimizers import sgd
from dagbldr.datasets import load_digits

# Common between tests
digits = load_digits()
X = digits["data"]
y = digits["target"]
n_classes = len(set(y))
y = convert_to_one_hot(y, n_classes)


def test_feedforward_classifier():
    minibatch_size = 100
    random_state = np.random.RandomState(1999)
    graph = OrderedDict()

    X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph)

    l1_o = linear_layer([X_sym], graph, 'l1', proj_dim=20,
                        random_state=random_state)
    y_pred = softmax_layer([l1_o], graph, 'pred', n_classes,
                           random_state=random_state)

    cost = categorical_crossentropy_nll(y_pred, y_sym).mean()
    params, grads = get_params_and_grads(graph, cost)
    learning_rate = 0.001
    opt = sgd(params)
    updates = opt.updates(params, grads, learning_rate)

    train_function = theano.function([X_sym, y_sym], [cost], updates=updates,
                                     mode="FAST_COMPILE")

    iterate_function(train_function, [X, y], minibatch_size,
                     list_of_output_names=["cost"], n_epochs=1)


def test_feedforward_theano_mix():
    minibatch_size = 100
    random_state = np.random.RandomState(1999)
    graph = OrderedDict()

    X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph)

    l1_o = linear_layer([X_sym], graph, 'l1', proj_dim=20,
                        random_state=random_state)
    #l1_o = .999 * l1_o
    y_pred = softmax_layer([l1_o], graph, 'pred', n_classes,
                           random_state=random_state)

    cost = categorical_crossentropy_nll(y_pred, y_sym).mean()
    params, grads = get_params_and_grads(graph, cost)
    learning_rate = 0.001
    opt = sgd(params)
    updates = opt.updates(params, grads, learning_rate)

    train_function = theano.function([X_sym, y_sym], [cost], updates=updates,
                                     mode="FAST_COMPILE")

    iterate_function(train_function, [X, y], minibatch_size,
                     list_of_output_names=["cost"], n_epochs=1)
