from collections import OrderedDict
import numpy as np
import theano

from dagbldr.datasets import fetch_iamondb
from dagbldr.optimizers import adam, gradient_clipping
from dagbldr.utils import add_datasets_to_graph, get_params_and_grads
from dagbldr.utils import create_checkpoint_dict, make_masked_minibatch
from dagbldr.utils import fixed_n_epochs_trainer
from dagbldr.nodes import relu_layer, lstm_recurrent_layer, masked_cost
from dagbldr.nodes import bernoulli_and_correlated_log_gaussian_mixture_layer
from dagbldr.nodes import bernoulli_and_correlated_log_gaussian_mixture_cost


def plot_scatter_iamondb_example(X, title=None, equal=True):
    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    down = np.where(X[:, 0] == 0)[0]
    up = np.where(X[:, 0] == 1)[0]
    ax.scatter(X[down, 1], X[down, 2], color="steelblue")
    ax.scatter(X[up, 1], X[up, 2], color="darkred")
    if equal:
        ax.set_aspect('equal')
    if title is not None:
        plt.title(title)
    plt.show()


def delta(x):
    return np.hstack((x[1:, 0][:, None], x[1:, 1:] - x[:-1, 1:]))


def undelta(x):
    agg = np.cumsum(x[:, 1:], axis=0)
    return np.hstack((x[:, 0][:, None], agg))


iamondb = fetch_iamondb()
X = iamondb["data"]
X_offset = [delta(x) for x in X]
X = X_offset
Xt = [x[:, 1:] for x in X]
X_len = np.array([len(x) for x in Xt]).sum()
X_mean = np.array([x.sum() for x in Xt]).sum() / X_len
X_sqr = np.array([(x**2).sum() for x in Xt]).sum() / X_len
X_std = np.sqrt(X_sqr - X_mean ** 2)


def normalize(x):
    return np.hstack((x[:, 0][:, None], (x[:, 1:] - X_mean) / (X_std)))

X = np.array([normalize(x) for x in X])
y = np.array([x[1:] for x in X])
X = np.array([x[:-1] for x in X])

train_end = int(.8 * len(X))
valid_end = len(X)
train_indices = np.arange(train_end)
valid_indices = np.arange(train_end, valid_end)

random_state = np.random.RandomState(1999)
minibatch_size = 20
n_hid = 300
rnn_dim = 1200

X_mb, X_mb_mask = make_masked_minibatch(X, slice(0, minibatch_size))
y_mb, y_mb_mask = make_masked_minibatch(y, slice(0, minibatch_size))


datasets_list = [X_mb, X_mb_mask, y_mb, y_mb_mask]
names_list = ["X", "X_mask", "y", "y_mask"]
graph = OrderedDict()
X_sym, X_mask_sym, y_sym, y_mask_sym = add_datasets_to_graph(
    datasets_list, names_list, graph)

l1 = relu_layer([X_sym], graph, 'l1', proj_dim=n_hid,
                random_state=random_state)
h = lstm_recurrent_layer([l1], X_mask_sym, rnn_dim, graph, 'l1_rec',
                         random_state=random_state)
l2 = relu_layer([h], graph, 'l2', proj_dim=n_hid,
                random_state=random_state)
rval = bernoulli_and_correlated_log_gaussian_mixture_layer(
    [l2], graph, 'hw', proj_dim=2, n_components=20, random_state=random_state)
binary, coeffs, mus, sigmas, corr = rval
cost = bernoulli_and_correlated_log_gaussian_mixture_cost(
    binary, coeffs, mus, sigmas, corr, y_sym)
cost = masked_cost(cost, y_mask_sym).sum(axis=0).mean()
params, grads = get_params_and_grads(graph, cost)

learning_rate = 0.0003
opt = adam(params, learning_rate)
clipped_grads = gradient_clipping(grads)
updates = opt.updates(params, clipped_grads)

fit_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym], [cost],
                               updates=updates)
cost_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym], [cost])
predict_function = theano.function([X_sym, X_mask_sym],
                                   [binary, coeffs, mus, sigmas, corr])

checkpoint_dict = create_checkpoint_dict(locals())

epoch_results = fixed_n_epochs_trainer(
    fit_function, cost_function, train_indices, valid_indices,
    checkpoint_dict, [X, y],
    minibatch_size,
    list_of_minibatch_functions=[make_masked_minibatch, make_masked_minibatch],
    list_of_train_output_names=["train_cost"],
    valid_output_name="valid_cost",
    valid_frequency="train_length",
    n_epochs=20)
