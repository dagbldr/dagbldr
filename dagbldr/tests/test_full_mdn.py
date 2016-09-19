"""
from collections import OrderedDict
import numpy as np
import theano

from dagbldr.optimizers import sgd
from dagbldr.datasets import minibatch_iterator
from dagbldr.utils import TrainingLoop
from dagbldr.utils import add_datasets_to_graph, get_params_and_grads
from dagbldr.utils import create_checkpoint_dict
from dagbldr.nodes import tanh_layer, gru_recurrent_layer, masked_cost
from dagbldr.nodes import bernoulli_and_correlated_log_gaussian_mixture_layer
from dagbldr.nodes import bernoulli_and_correlated_log_gaussian_mixture_cost
from dagbldr.nodes import log_gaussian_mixture_layer, log_gaussian_mixture_cost

# This example based on a great tutorial on Mixture Density Networks in TF
# http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/


def make_noisy_sinusoid(n_samples=1000):
    random_state = np.random.RandomState(1999)
    x = random_state.uniform(-10, 10, size=(n_samples,))
    r = random_state.normal(size=(n_samples,))
    # Sinusoid with frequency ~0.75, amplitude 7, linear trend of .5
    # and additive noise
    y = np.sin(0.75 * x) * 7 + .5 * x + r
    x = x.astype(theano.config.floatX)
    y = y.astype(theano.config.floatX)
    return x, y

sine_x, sine_y = make_noisy_sinusoid(n_samples=10)
# Swap X and Y to create a one to many relationship

sine_x, sine_y = sine_y, sine_x
# Make 1 minibatch with feature dimension 1
sine_x = sine_x[:, None]
sine_y = sine_y[:, None]

X = sine_x
y = sine_y

random_state = np.random.RandomState(1999)
bern = random_state.rand(X.shape[0], 1)
bern[bern > 0.5] = 1
bern[bern <= 0.5] = 0
x_y = random_state.randn(X.shape[0], 2)
bern = np.hstack((bern, x_y))
bernoulli_X = bern[:-1]
bernoulli_y = bern[1:]


def test_mixture_density():
    # graph holds information necessary to build layers from parents
    random_state = np.random.RandomState(1999)
    graph = OrderedDict()
    X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph)
    n_hid = 20
    n_out = 1
    minibatch_size = len(X)
    train_indices = np.arange(len(X))
    valid_indices = np.arange(len(X))

    l1 = tanh_layer([X_sym], graph, 'l1', proj_dim=n_hid,
                    random_state=random_state)
    coeffs, mus, log_sigmas = log_gaussian_mixture_layer(
        [l1], graph, 'mdn', proj_dim=1, n_components=2,
        random_state=random_state)
    cost = log_gaussian_mixture_cost(coeffs, mus, log_sigmas, y_sym).mean()
    params, grads = get_params_and_grads(graph, cost)

    learning_rate = 1E-6
    opt = sgd(params, learning_rate)
    updates = opt.updates(params, grads)

    fit_function = theano.function([X_sym, y_sym], [cost], updates=updates,
                                   mode="FAST_COMPILE")
    cost_function = theano.function([X_sym, y_sym], [cost],
                                    mode="FAST_COMPILE")

    checkpoint_dict = create_checkpoint_dict(locals())
    train_itr = minibatch_iterator([X, y], minibatch_size, axis=0)
    valid_itr = minibatch_iterator([X, y], minibatch_size, axis=0)
    TL = TrainingLoop(fit_function, cost_function,
                      train_itr, valid_itr,
                      checkpoint_dict=checkpoint_dict,
                      list_of_train_output_names=["train_cost"],
                      valid_output_name="valid_cost",
                      n_epochs=1)
    TL.run()


def test_correlated_mixture_density():
    # graph holds information necessary to build layers from parents
    random_state = np.random.RandomState(1999)
    graph = OrderedDict()
    X_sym, y_sym = add_datasets_to_graph([bernoulli_X, bernoulli_y], ["X", "y"],
                                         graph)
    n_hid = 20
    minibatch_size = len(bernoulli_X)

    l1 = tanh_layer([X_sym], graph, 'l1', proj_dim=n_hid,
                    random_state=random_state)
    rval = bernoulli_and_correlated_log_gaussian_mixture_layer(
        [l1], graph, 'hw', proj_dim=2, n_components=3,
        random_state=random_state)
    binary, coeffs, mus, log_sigmas, corr = rval
    cost = bernoulli_and_correlated_log_gaussian_mixture_cost(
        binary, coeffs, mus, log_sigmas, corr, y_sym).mean()
    params, grads = get_params_and_grads(graph, cost)

    learning_rate = 1E-6
    opt = sgd(params, learning_rate)
    updates = opt.updates(params, grads)

    fit_function = theano.function([X_sym, y_sym], [cost], updates=updates,
                                   mode="FAST_COMPILE")
    cost_function = theano.function([X_sym, y_sym], [cost],
                                    mode="FAST_COMPILE")

    checkpoint_dict = create_checkpoint_dict(locals())
    train_itr = minibatch_iterator([bernoulli_X, bernoulli_y],
                                   minibatch_size, axis=0)
    valid_itr = minibatch_iterator([bernoulli_X, bernoulli_y],
                                   minibatch_size, axis=0)
    TL = TrainingLoop(fit_function, cost_function,
                      train_itr, valid_itr,
                      checkpoint_dict=checkpoint_dict,
                      list_of_train_output_names=["train_cost"],
                      valid_output_name="valid_cost",
                      n_epochs=1)
    TL.run()


def test_rnn_correlated_mixture_density():
    # graph holds information necessary to build layers from parents
    random_state = np.random.RandomState(1999)
    graph = OrderedDict()
    minibatch_size = 5
    X_seq = np.array([bernoulli_X for i in range(minibatch_size)])
    y_seq = np.array([bernoulli_y for i in range(minibatch_size)])
    train_itr = minibatch_iterator([X_seq, y_seq], minibatch_size,
                                   make_mask=True, axis=1)
    X_mb, X_mb_mask, y_mb, y_mb_mask = next(train_itr)
    train_itr.reset()
    datasets_list = [X_mb, X_mb_mask, y_mb, y_mb_mask]
    names_list = ["X", "X_mask", "y", "y_mask"]
    X_sym, X_mask_sym, y_sym, y_mask_sym = add_datasets_to_graph(
        datasets_list, names_list, graph)
    n_hid = 5

    l1 = tanh_layer([X_sym], graph, 'l1', proj_dim=n_hid,
                    random_state=random_state)
    h = gru_recurrent_layer([l1], X_mask_sym, n_hid, graph, 'l1_rec',
                            random_state=random_state)
    rval = bernoulli_and_correlated_log_gaussian_mixture_layer(
        [h], graph, 'hw', proj_dim=2, n_components=3,
        random_state=random_state)
    binary, coeffs, mus, log_sigmas, corr = rval
    cost = bernoulli_and_correlated_log_gaussian_mixture_cost(
        binary, coeffs, mus, log_sigmas, corr, y_sym)
    cost = masked_cost(cost, y_mask_sym).mean()
    cost_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym],
                                    [cost],
                                    mode="FAST_COMPILE")

    checkpoint_dict = create_checkpoint_dict(locals())
    valid_itr = minibatch_iterator([X_seq, y_seq], minibatch_size,
                                   make_mask=True, axis=1)
    # no fit
    TL = TrainingLoop(cost_function, cost_function,
                      train_itr, valid_itr,
                      checkpoint_dict=checkpoint_dict,
                      list_of_train_output_names=["train_cost"],
                      valid_output_name="valid_cost",
                      n_epochs=1)
    TL.run()
"""
