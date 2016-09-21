import numpy as np
import theano
from theano import tensor

from dagbldr.datasets import minibatch_iterator
from dagbldr.utils import convert_to_one_hot
from dagbldr import get_params, del_shared
from dagbldr.nodes import linear, softmax

from dagbldr.nodes import categorical_crossentropy
from dagbldr.optimizers import sgd
from dagbldr.datasets import load_digits

# Common between tests
digits = load_digits()
X = digits["data"].astype("float32")
y = digits["target"]
n_classes = len(set(y))
y = convert_to_one_hot(y, n_classes).astype("float32")


def test_feedforward_classifier():
    del_shared()
    minibatch_size = 100
    random_state = np.random.RandomState(1999)
    X_sym = tensor.fmatrix()
    y_sym = tensor.fmatrix()

    l1_o = linear([X_sym], [X.shape[1]], proj_dim=20, name='l1',
                  random_state=random_state)
    y_pred = softmax([l1_o], [20], proj_dim=n_classes, name='out',
                     random_state=random_state)

    cost = categorical_crossentropy(y_pred, y_sym).mean()
    params = list(get_params().values())
    grads = theano.grad(cost, params)
    learning_rate = 0.001
    opt = sgd(params, learning_rate)
    updates = opt.updates(params, grads)

    fit_function = theano.function([X_sym, y_sym], [cost], updates=updates,
                                   mode="FAST_COMPILE")

    cost_function = theano.function([X_sym, y_sym], [cost],
                                    mode="FAST_COMPILE")
    train_itr = minibatch_iterator([X, y], minibatch_size, axis=0)
    valid_itr = minibatch_iterator([X, y], minibatch_size, axis=0)
    X_train, y_train = next(train_itr)
    X_train, y_train = next(train_itr)
    X_valid, y_valid = next(valid_itr)
    fit_function(X_train, y_train)
    cost_function(X_valid, y_valid)


def test_feedforward_theano_mix():
    del_shared()
    minibatch_size = 100
    random_state = np.random.RandomState(1999)
    X_sym = tensor.fmatrix()
    y_sym = tensor.fmatrix()

    l1_o = linear([X_sym], [X.shape[1]], proj_dim=20, name='l1',
                  random_state=random_state)
    l1_o = .999 * l1_o
    y_pred = softmax([l1_o], [20], proj_dim=n_classes, name='out',
                     random_state=random_state)

    cost = categorical_crossentropy(y_pred, y_sym).mean()
    params = list(get_params().values())
    grads = theano.grad(cost, params)
    learning_rate = 0.001
    opt = sgd(params, learning_rate)
    updates = opt.updates(params, grads)

    fit_function = theano.function([X_sym, y_sym], [cost], updates=updates,
                                   mode="FAST_COMPILE")

    cost_function = theano.function([X_sym, y_sym], [cost],
                                    mode="FAST_COMPILE")

    train_itr = minibatch_iterator([X, y], minibatch_size, axis=0)
    valid_itr = minibatch_iterator([X, y], minibatch_size, axis=0)
    X_train, y_train = next(train_itr)
    X_valid, y_valid = next(valid_itr)
    fit_function(X_train, y_train)
    cost_function(X_valid, y_valid)
