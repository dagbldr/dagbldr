from collections import OrderedDict
import numpy as np
import theano

from dagbldr.datasets import fetch_mnist
from dagbldr.optimizers import sgd
from dagbldr.utils import add_datasets_to_graph, get_params_and_grads
from dagbldr.utils import get_weights_from_graph
from dagbldr.utils import convert_to_one_hot
from dagbldr.utils import create_checkpoint_dict
from dagbldr.utils import fixed_n_epochs_trainer
from dagbldr.nodes import softmax_zeros_layer
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

minibatch_size = 600

y_pred = softmax_zeros_layer([X_sym], graph, 'y_pred',  proj_dim=n_targets)
nll = categorical_crossentropy(y_pred, y_sym).mean()
weights = get_weights_from_graph(graph)
cost = nll

params, grads = get_params_and_grads(graph, cost)

learning_rate = .13
opt = sgd(params, learning_rate)
updates = opt.updates(params, grads)


fit_function = theano.function([X_sym, y_sym], [cost], updates=updates)
cost_function = theano.function([X_sym, y_sym], [cost])
predict_function = theano.function([X_sym], [y_pred])

checkpoint_dict = create_checkpoint_dict(locals())


def error(*args):
    xargs = args[:-1]
    y = args[-1]
    final_args = xargs
    y_pred = predict_function(*final_args)[0]
    return 1 - np.mean((np.argmax(
        y_pred, axis=1).ravel()) == (np.argmax(y, axis=1).ravel()))

epoch_results = fixed_n_epochs_trainer(
    fit_function, error, train_indices, valid_indices,
    checkpoint_dict, [X, y],
    minibatch_size,
    list_of_train_output_names=["train_cost"],
    valid_output_name="valid_error",
    n_epochs=100,
    optimizer_object=opt)
