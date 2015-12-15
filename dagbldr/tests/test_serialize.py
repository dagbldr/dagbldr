from collections import OrderedDict
import numpy as np
import theano

from dagbldr.datasets import load_digits
from dagbldr.optimizers import sgd
from dagbldr.utils import add_datasets_to_graph, get_params_and_grads
from dagbldr.utils import get_weights_from_graph
from dagbldr.utils import convert_to_one_hot
from dagbldr.utils import create_checkpoint_dict, load_last_checkpoint
from dagbldr.utils import TrainingLoop
from dagbldr.nodes import softmax_zeros_layer
from dagbldr.nodes import categorical_crossentropy


digits = load_digits()
X = digits["data"]
y = digits["target"]
train_indices = np.arange(len(X))
valid_indices = np.arange(len(X))
n_targets = 10
y = convert_to_one_hot(y, n_targets)


def test_loop():
    # graph holds information necessary to build layers from parents
    graph = OrderedDict()
    X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph)
    # random state so script is deterministic
    random_state = np.random.RandomState(1999)

    minibatch_size = 10

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

    checkpoint_dict = {"fit_function": fit_function,
                       "cost_function": cost_function,
                       "predict_function": predict_function}

    def error(*args):
        xargs = args[:-1]
        y = args[-1]
        final_args = xargs
        y_pred = predict_function(*final_args)[0]
        return 1 - np.mean((np.argmax(
            y_pred, axis=1).ravel()) == (np.argmax(y, axis=1).ravel()))

    TL1 = TrainingLoop(fit_function, error,
                       train_indices[:10],
                       valid_indices[:10],
                       minibatch_size,
                       checkpoint_dict=checkpoint_dict,
                       list_of_train_output_names=["train_cost"],
                       valid_output_name="valid_error", n_epochs=1,
                       optimizer_object=opt)
    epoch_results1 = TL1.run([X, y])
    TL1.train_indices = train_indices[10:20]
    TL1.valid_indices = valid_indices[10:20]
    epoch_results1 = TL1.run([X, y])

    TL2 = TrainingLoop(fit_function, error,
                       train_indices[:20],
                       valid_indices[:20],
                       minibatch_size,
                       checkpoint_dict=checkpoint_dict,
                       list_of_train_output_names=["train_cost"],
                       valid_output_name="valid_error", n_epochs=1,
                       optimizer_object=opt)
    epoch_results2 = TL2.run([X, y])

    r1 = TL1.__dict__["checkpoint_dict"]["previous_results"]["train_cost"][-1]
    r2 = TL2.__dict__["checkpoint_dict"]["previous_results"]["train_cost"][-1]
    assert r1 == r2
