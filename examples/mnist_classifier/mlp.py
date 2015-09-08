from collections import OrderedDict
import numpy as np
import theano

from dagbldr.datasets import fetch_mnist
from dagbldr.optimizers import rmsprop
from dagbldr.utils import add_datasets_to_graph, get_params_and_grads
from dagbldr.utils import get_weights_from_graph
from dagbldr.utils import convert_to_one_hot
from dagbldr.utils import early_stopping_trainer
from dagbldr.nodes import tanh_layer, softmax_zeros_layer
from dagbldr.nodes import categorical_crossentropy


mnist = fetch_mnist()
train_indices = mnist["train_indices"]
valid_indices = mnist["valid_indices"]
X = mnist["data"]
y = mnist["target"]
n_targets = 10
y = convert_to_one_hot(y, n_targets)

# graph holds information necessary to build layers from parents
graph = OrderedDict()
X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph)
# random state so script is deterministic
random_state = np.random.RandomState(1999)

minibatch_size = 20
n_hid = 1000

l1 = tanh_layer([X_sym], graph, 'l1', proj_dim=n_hid, random_state=random_state)
y_pred = softmax_zeros_layer([l1], graph, 'y_pred',  proj_dim=n_targets)
nll = categorical_crossentropy(y_pred, y_sym).mean()
weights = get_weights_from_graph(graph)
L2 = sum([(w ** 2).sum() for w in weights])
cost = nll + .0001 * L2


params, grads = get_params_and_grads(graph, cost)

learning_rate = 1E-4
momentum = 0.95
opt = rmsprop(params)
updates = opt.updates(params, grads, learning_rate, momentum)

fit_function = theano.function([X_sym, y_sym], [cost], updates=updates)
cost_function = theano.function([X_sym, y_sym], [cost])
predict_function = theano.function([X_sym], [y_pred])
checkpoint_dict = {}
checkpoint_dict["fit_function"] = fit_function
checkpoint_dict["cost_function"] = cost_function
checkpoint_dict["predict_function"] = predict_function
previous_epoch_results = None


def error(*args):
    xargs = args[:-1]
    y = args[-1]
    final_args = xargs
    y_pred = predict_function(*final_args)[0]
    return 1 - np.mean((np.argmax(
        y_pred, axis=1).ravel()) == (np.argmax(y, axis=1).ravel()))

epoch_results = early_stopping_trainer(
    fit_function, error, checkpoint_dict, [X, y],
    minibatch_size, train_indices, valid_indices,
    fit_function_output_names=["cost"],
    cost_function_output_name="valid_cost",
    n_epochs=1000, previous_epoch_results=previous_epoch_results)
