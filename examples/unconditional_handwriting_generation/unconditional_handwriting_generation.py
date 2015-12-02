from collections import OrderedDict
import numpy as np
import theano
import copy

from dagbldr.datasets import fetch_iamondb
from dagbldr.optimizers import adadelta
from dagbldr.utils import add_datasets_to_graph, get_params_and_grads
from dagbldr.utils import create_checkpoint_dict, make_masked_minibatch
from dagbldr.utils import fixed_n_epochs_trainer
from dagbldr.nodes import maxout_layer, gru_recurrent_layer, masked_cost
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


iamondb = fetch_iamondb()
X = iamondb["data"]
vocab = iamondb["vocabulary"]
X_mean = np.mean([x.mean(axis=0) for x in X], axis=0)
# Don't mean center the pen points
X_mean[0] = 0.


def center_x(x):
    return x - X_mean

X = np.array([center_x(x) for x in X])

X_min = np.min([x.min(axis=0) for x in X], axis=0)
X_max = np.max([x.max(axis=0) for x in X], axis=0)


def scale_x(x):
    return (x - X_min) / (X_max - X_min)

X = np.array([scale_x(x) for x in X])


def delta_x(x):
    x[1:, 1:] = x[1:, 1:] - x[:-1, 1:]
    x[0, 1:] = 0.
    return x


def undelta_x(x):
    x[:, 1:] = np.cumsum(x[:, 1:], axis=0)
    return x


def invert_hot(y_hot):
    return "".join([vocab[i] for i in y_hot.argmax(axis=1)])


X = np.array([delta_x(x) for x in X])
y = np.array([x[1:] for x in X])
X = np.array([x[:-1] for x in X])

X_r = copy.deepcopy(X)
X_r = np.array([undelta_x(x) for x in X_r])
train_end = int(.8 * len(X))
valid_end = len(X)
train_indices = np.arange(train_end)
valid_indices = np.arange(train_end, valid_end)

random_state = np.random.RandomState(1999)
minibatch_size = 10
n_hid = 1024

X_mb, X_mb_mask = make_masked_minibatch(X, slice(0, minibatch_size))
y_mb, y_mb_mask = make_masked_minibatch(y, slice(0, minibatch_size))


datasets_list = [X_mb, X_mb_mask, y_mb, y_mb_mask]
names_list = ["X", "X_mask", "y", "y_mask"]
graph = OrderedDict()
X_sym, X_mask_sym, y_sym, y_mask_sym = add_datasets_to_graph(
    datasets_list, names_list, graph)

l1 = maxout_layer([X_sym], graph, 'l1', proj_dim=n_hid,
                  random_state=random_state)
h = gru_recurrent_layer([l1], X_mask_sym, n_hid, graph, 'l1_rec',
                        random_state=random_state)
l2 = maxout_layer([h], graph, 'l2', n_hid, random_state=random_state)
rval = bernoulli_and_correlated_log_gaussian_mixture_layer(
    [l2], graph, 'hw', proj_dim=2, n_components=20, random_state=random_state)
binary, coeffs, mus, log_sigmas, corr = rval
cost = bernoulli_and_correlated_log_gaussian_mixture_cost(
    binary, coeffs, mus, log_sigmas, corr, y_sym)
cost = masked_cost(cost, y_mask_sym).mean()
params, grads = get_params_and_grads(graph, cost)

opt = adadelta(params)
updates = opt.updates(params, grads)

fit_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym], [cost],
                               updates=updates)
cost_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym], [cost])
predict_function = theano.function([X_sym, X_mask_sym],
                                   [binary, coeffs, mus, log_sigmas, corr])

checkpoint_dict = create_checkpoint_dict(locals())

epoch_results = fixed_n_epochs_trainer(
    fit_function, cost_function, train_indices, valid_indices,
    checkpoint_dict, [X, y],
    minibatch_size,
    list_of_minibatch_functions=[make_masked_minibatch, make_masked_minibatch],
    list_of_train_output_names=["train_cost"],
    valid_output_name="valid_cost",
    n_epochs=1000)
