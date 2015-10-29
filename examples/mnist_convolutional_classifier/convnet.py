from collections import OrderedDict
import numpy as np
import theano

from dagbldr.datasets import fetch_mnist
from dagbldr.optimizers import sgd_nesterov
from dagbldr.utils import add_datasets_to_graph, get_params_and_grads
from dagbldr.utils import convert_to_one_hot, early_stopping_trainer
from dagbldr.nodes import conv2d_layer, pool2d_layer
from dagbldr.nodes import softmax_layer, categorical_crossentropy


mnist = fetch_mnist()
train_indices = mnist["train_indices"]
valid_indices = mnist["valid_indices"]
X = mnist["images"]
y = mnist["target"]
n_targets = 10
y = convert_to_one_hot(y, n_targets)
minibatch_size = 128

# graph holds information necessary to build layers from parents
graph = OrderedDict()
X_sym, y_sym = add_datasets_to_graph([X[:minibatch_size], y[:minibatch_size]],
                                     ["X", "y"], graph)
# random state so script is deterministic
random_state = np.random.RandomState(1999)


l1 = conv2d_layer([X_sym], graph, 'conv1', 8, random_state=random_state)
l2 = pool2d_layer([l1], graph, 'pool1')
l3 = conv2d_layer([l2], graph, 'conv2', 16, random_state=random_state)
l4 = pool2d_layer([l3], graph, 'pool2')
l5 = l4.reshape((l4.shape[0], -1))
y_pred = softmax_layer([l5], graph, 'y_pred', n_targets,
                       random_state=random_state)
nll = categorical_crossentropy(y_pred, y_sym).mean()
cost = nll

params, grads = get_params_and_grads(graph, cost)

learning_rate = 0.001
momentum = 0.9
opt = sgd_nesterov(params, learning_rate, momentum)
updates = opt.updates(params, grads)

fit_function = theano.function([X_sym, y_sym], [cost], updates=updates)
cost_function = theano.function([X_sym, y_sym], [cost])
predict_function = theano.function([X_sym], [y_pred])

checkpoint_dict = {}
checkpoint_dict["fit_function"] = fit_function
checkpoint_dict["cost_function"] = cost_function
checkpoint_dict["predict_function"] = predict_function
previous_results = None


def error(X_mb, y_mb):
    y_pred = predict_function(X_mb)[0]
    return 1 - np.mean((np.argmax(
        y_pred, axis=1).ravel()) == (np.argmax(y_mb, axis=1).ravel()))


epoch_results = early_stopping_trainer(
    fit_function, error, train_indices, valid_indices,
    checkpoint_dict, [X, y],
    minibatch_size,
    list_of_train_output_names=["train_cost"],
    valid_output_name="valid_error",
    n_epochs=1000,
    optimizer_object=opt,
    previous_results=previous_results)
