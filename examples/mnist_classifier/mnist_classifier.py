from collections import OrderedDict
import numpy as np
import theano

from dagbldr.datasets import fetch_binarized_mnist
from dagbldr.optimizers import adam
from dagbldr.utils import add_datasets_to_graph, get_params_and_grads
from dagbldr.utils import convert_to_one_hot
from dagbldr.utils import load_last_checkpoint, early_stopping_trainer
from dagbldr.nodes import relu_layer, softmax_layer
from dagbldr.nodes import categorical_crossentropy


mnist = fetch_binarized_mnist()
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

minibatch_size = 100
n_hid = 512

# q(y_pred | x)
l1 = relu_layer([X_sym], graph, 'l1', n_hid, random_state)
l2 = relu_layer([l1], graph, 'l2', n_hid, random_state)
y_pred = softmax_layer([l2], graph, 'y_pred',  n_targets, random_state)
nll = categorical_crossentropy(y_pred, y_sym).mean()
cost = nll

params, grads = get_params_and_grads(graph, cost)

learning_rate = 0.0002
opt = adam(params)
updates = opt.updates(params, grads, learning_rate)

# Checkpointing
try:
    checkpoint_dict = load_last_checkpoint()
    fit_function = checkpoint_dict["fit_function"]
    cost_function = checkpoint_dict["cost_function"]
    predict_function = checkpoint_dict["predict_function"]
    previous_epoch_results = checkpoint_dict["previous_epoch_results"]
except KeyError:
    fit_function = theano.function([X_sym, y_sym], [cost],
                                   updates=updates)
    cost_function = theano.function([X_sym, y_sym], [cost])
    predict_function = theano.function([X_sym], [y_pred])
    checkpoint_dict = {}
    checkpoint_dict["fit_function"] = fit_function
    checkpoint_dict["cost_function"] = cost_function
    checkpoint_dict["predict_function"] = predict_function
    previous_epoch_results = None

epoch_results = early_stopping_trainer(
    fit_function, cost_function, checkpoint_dict, [X, y],
    minibatch_size, train_indices, valid_indices,
    fit_function_output_names=["cost"],
    cost_function_output_name="valid_cost",
    n_epochs=100, previous_epoch_results=previous_epoch_results,
    shuffle=True, random_state=random_state)
