from collections import OrderedDict
import numpy as np
import theano

from dagbldr.utils import add_datasets_to_graph, convert_to_one_hot
from dagbldr.utils import get_params_and_grads
from dagbldr.utils import early_stopping_trainer
from dagbldr.nodes import linear_layer, softmax_layer
from dagbldr.nodes import categorical_crossentropy
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

    cost = categorical_crossentropy(y_pred, y_sym).mean()
    params, grads = get_params_and_grads(graph, cost)
    learning_rate = 0.001
    opt = sgd(params, learning_rate)
    updates = opt.updates(params, grads)

    fit_function = theano.function([X_sym, y_sym], [cost], updates=updates,
                                   mode="FAST_COMPILE")

    cost_function = theano.function([X_sym, y_sym], [cost],
                                    mode="FAST_COMPILE")

    checkpoint_dict = {}
    train_indices = np.arange(len(X))
    valid_indices = np.arange(len(X))
    early_stopping_trainer(fit_function, cost_function,
                           train_indices, valid_indices,
                           checkpoint_dict, [X, y],
                           minibatch_size,
                           list_of_train_output_names=["cost"],
                           valid_output_name="valid_cost",
                           n_epochs=1)


def test_feedforward_theano_mix():
    minibatch_size = 100
    random_state = np.random.RandomState(1999)
    graph = OrderedDict()

    X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph)

    l1_o = linear_layer([X_sym], graph, 'l1', proj_dim=20,
                        random_state=random_state)
    l1_o = .999 * l1_o
    y_pred = softmax_layer([l1_o], graph, 'pred', n_classes,
                           random_state=random_state)

    cost = categorical_crossentropy(y_pred, y_sym).mean()
    params, grads = get_params_and_grads(graph, cost)
    learning_rate = 0.001
    opt = sgd(params, learning_rate)
    updates = opt.updates(params, grads)

    fit_function = theano.function([X_sym, y_sym], [cost], updates=updates,
                                   mode="FAST_COMPILE")

    cost_function = theano.function([X_sym, y_sym], [cost],
                                    mode="FAST_COMPILE")

    checkpoint_dict = {}
    train_indices = np.arange(len(X))
    valid_indices = np.arange(len(X))
    early_stopping_trainer(fit_function, cost_function,
                           train_indices, valid_indices,
                           checkpoint_dict, [X, y],
                           minibatch_size,
                           list_of_train_output_names=["cost"],
                           valid_output_name="valid_cost",
                           n_epochs=1)
