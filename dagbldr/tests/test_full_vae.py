"""
from collections import OrderedDict
import numpy as np
import theano

from dagbldr.datasets import load_digits, minibatch_iterator
from dagbldr.utils import add_datasets_to_graph, convert_to_one_hot
from dagbldr.utils import get_params_and_grads
from dagbldr.utils import TrainingLoop
from dagbldr.nodes import softplus_layer, linear_layer
from dagbldr.nodes import gaussian_log_sample_layer
from dagbldr.nodes import sigmoid_layer
from dagbldr.nodes import binary_crossentropy, gaussian_log_kl
from dagbldr.optimizers import sgd

# Common between tests
digits = load_digits()
X = digits["data"][:100]
# Binarize digits
X[X > 0.5] = 2
X[X < 0.5] = 1
X -= 1
y = digits["target"][:100]
n_classes = len(set(y))
y = convert_to_one_hot(y, n_classes)


def test_vae():
    minibatch_size = 10
    random_state = np.random.RandomState(1999)
    graph = OrderedDict()

    X_sym = add_datasets_to_graph([X], ["X"], graph)

    l1_enc = softplus_layer([X_sym], graph, 'l1_enc', proj_dim=100,
                            random_state=random_state)
    mu = linear_layer([l1_enc], graph, 'mu', proj_dim=50,
                      random_state=random_state)
    log_sigma = linear_layer([l1_enc], graph, 'log_sigma', proj_dim=50,
                             random_state=random_state)
    samp = gaussian_log_sample_layer([mu], [log_sigma], graph,
                                     'gaussian_log_sample',
                                     random_state=random_state)
    l1_dec = softplus_layer([samp], graph, 'l1_dec', proj_dim=100,
                            random_state=random_state)
    out = sigmoid_layer([l1_dec], graph, 'out', proj_dim=X.shape[1],
                        random_state=random_state)

    kl = gaussian_log_kl([mu], [log_sigma], graph, 'gaussian_kl').mean()
    cost = binary_crossentropy(out, X_sym).mean() + kl
    params, grads = get_params_and_grads(graph, cost)
    learning_rate = 0.00000
    opt = sgd(params, learning_rate)
    updates = opt.updates(params, grads)

    fit_function = theano.function([X_sym], [cost], updates=updates,
                                   mode="FAST_COMPILE")

    cost_function = theano.function([X_sym], [cost],
                                    mode="FAST_COMPILE")

    checkpoint_dict = {}
    train_itr = minibatch_iterator([X], minibatch_size, axis=0)
    valid_itr = minibatch_iterator([X], minibatch_size, axis=0)
    TL = TrainingLoop(fit_function, cost_function,
                      train_itr, valid_itr,
                      checkpoint_dict=checkpoint_dict,
                      list_of_train_output_names=["cost"],
                      valid_output_name="valid_cost",
                      n_epochs=1)
    TL.run()
"""
