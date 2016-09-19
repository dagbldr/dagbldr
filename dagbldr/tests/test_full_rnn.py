"""
import numpy as np
import theano

from theano.compat.python2x import OrderedDict
from dagbldr.datasets import make_sincos, minibatch_iterator
from dagbldr.optimizers import sgd
from dagbldr.utils import add_datasets_to_graph, get_params_and_grads
from dagbldr.utils import TrainingLoop
from dagbldr.nodes import linear_layer, squared_error, masked_cost
from dagbldr.nodes import tanh_recurrent_layer, gru_recurrent_layer
from dagbldr.nodes import lstm_recurrent_layer


# Generate sinewaves offset in phase
n_timesteps = 50
minibatch_size = 20
data = make_sincos(n_timesteps, minibatch_size)
X = data[:-1]
y = data[1:]

# Use all ones for mask because each one is the same length
X_mask = np.ones_like(X[:, :, 0])
y_mask = np.ones_like(y[:, :, 0])


def test_tanh_rnn():
    # random state so script is deterministic
    random_state = np.random.RandomState(1999)
    # home of the computational graph
    graph = OrderedDict()

    # number of hidden features
    n_hid = 10
    # number of output_features = input_features
    n_out = X.shape[-1]

    # input (where first dimension is time)
    datasets_list = [X, X_mask, y, y_mask]
    names_list = ["X", "X_mask", "y", "y_mask"]
    test_values_list = [X, X_mask, y, y_mask]
    X_sym, X_mask_sym, y_sym, y_mask_sym = add_datasets_to_graph(
        datasets_list, names_list, graph, list_of_test_values=test_values_list)

    # Setup weights
    l1 = linear_layer([X_sym], graph, 'l1_proj', proj_dim=n_hid,
                      random_state=random_state)

    h = tanh_recurrent_layer([l1], X_mask_sym, n_hid, graph, 'l1_rec',
                             random_state)

    # linear output activation
    y_hat = linear_layer([h], graph, 'l2_proj', proj_dim=n_out,
                         random_state=random_state)

    # error between output and target
    cost = squared_error(y_hat, y_sym)
    cost = masked_cost(cost, y_mask_sym).mean()
    # Parameters of the model
    params, grads = get_params_and_grads(graph, cost)

    # Use stochastic gradient descent to optimize
    learning_rate = 0.001
    opt = sgd(params, learning_rate)
    updates = opt.updates(params, grads)

    fit_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym],
                                   [cost], updates=updates, mode="FAST_COMPILE")

    cost_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym],
                                    [cost], mode="FAST_COMPILE")
    checkpoint_dict = {}
    train_itr = minibatch_iterator([X, y], minibatch_size, make_mask=True,
                                   axis=1)
    valid_itr = minibatch_iterator([X, y], minibatch_size, make_mask=True,
                                   axis=1)
    TL = TrainingLoop(fit_function, cost_function,
                      train_itr, valid_itr,
                      checkpoint_dict=checkpoint_dict,
                      list_of_train_output_names=["cost"],
                      valid_output_name="valid_cost",
                      n_epochs=1)
    TL.run()


def test_gru_rnn():
    # random state so script is deterministic
    random_state = np.random.RandomState(1999)
    # home of the computational graph
    graph = OrderedDict()

    # number of hidden features
    n_hid = 10
    # number of output_features = input_features
    n_out = X.shape[-1]

    # input (where first dimension is time)
    datasets_list = [X, X_mask, y, y_mask]
    names_list = ["X", "X_mask", "y", "y_mask"]
    test_values_list = [X, X_mask, y, y_mask]
    X_sym, X_mask_sym, y_sym, y_mask_sym = add_datasets_to_graph(
        datasets_list, names_list, graph, list_of_test_values=test_values_list)

    # Setup weights
    l1 = linear_layer([X_sym], graph, 'l1_proj', n_hid,
                      random_state=random_state)

    h = gru_recurrent_layer([l1], X_mask_sym, n_hid, graph, 'l1_rec',
                            random_state)

    # linear output activation
    y_hat = linear_layer([h], graph, 'l2_proj', n_out,
                         random_state=random_state)

    # error between output and target
    cost = squared_error(y_hat, y_sym)
    cost = masked_cost(cost, y_mask_sym).mean()
    # Parameters of the model
    params, grads = get_params_and_grads(graph, cost)

    # Use stochastic gradient descent to optimize
    learning_rate = 0.01
    opt = sgd(params, learning_rate)
    updates = opt.updates(params, grads)
    fit_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym],
                                   [cost], updates=updates, mode="FAST_COMPILE")

    cost_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym],
                                    [cost], mode="FAST_COMPILE")
    checkpoint_dict = {}
    train_itr = minibatch_iterator([X, y], minibatch_size, make_mask=True,
                                   axis=1)
    valid_itr = minibatch_iterator([X, y], minibatch_size, make_mask=True,
                                   axis=1)
    TL = TrainingLoop(fit_function, cost_function,
                      train_itr, valid_itr,
                      checkpoint_dict=checkpoint_dict,
                      list_of_train_output_names=["cost"],
                      valid_output_name="valid_cost",
                      n_epochs=1)
    TL.run()


def test_lstm_rnn():
    # random state so script is deterministic
    random_state = np.random.RandomState(1999)
    # home of the computational graph
    graph = OrderedDict()

    # number of hidden features
    n_hid = 10
    # number of output_features = input_features
    n_out = X.shape[-1]

    # input (where first dimension is time)
    datasets_list = [X, X_mask, y, y_mask]
    names_list = ["X", "X_mask", "y", "y_mask"]
    test_values_list = [X, X_mask, y, y_mask]
    X_sym, X_mask_sym, y_sym, y_mask_sym = add_datasets_to_graph(
        datasets_list, names_list, graph, list_of_test_values=test_values_list)

    # Setup weights
    l1 = linear_layer([X_sym], graph, 'l1_proj', n_hid,
                      random_state=random_state)

    h = lstm_recurrent_layer([l1], X_mask_sym, n_hid, graph, 'l1_rec',
                             random_state)

    # linear output activation
    y_hat = linear_layer([h], graph, 'l2_proj', n_out,
                         random_state=random_state)

    # error between output and target
    cost = squared_error(y_hat, y_sym)
    cost = masked_cost(cost, y_mask_sym).mean()
    # Parameters of the model
    params, grads = get_params_and_grads(graph, cost)

    # Use stochastic gradient descent to optimize
    learning_rate = 0.01
    opt = sgd(params, learning_rate)
    updates = opt.updates(params, grads)

    fit_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym],
                                   [cost], updates=updates, mode="FAST_COMPILE")

    cost_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym],
                                    [cost], mode="FAST_COMPILE")
    checkpoint_dict = {}
    train_itr = minibatch_iterator([X, y], minibatch_size, make_mask=True,
                                   axis=1)
    valid_itr = minibatch_iterator([X, y], minibatch_size, make_mask=True,
                                   axis=1)
    TL = TrainingLoop(fit_function, cost_function,
                      train_itr, valid_itr,
                      checkpoint_dict=checkpoint_dict,
                      list_of_train_output_names=["cost"],
                      valid_output_name="valid_cost",
                      n_epochs=1)
    TL.run()
"""
