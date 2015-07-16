from collections import OrderedDict
import numpy as np
import theano

from dagbldr.datasets import load_digits
from dagbldr.utils import add_datasets_to_graph, convert_to_one_hot
from dagbldr.utils import get_params_and_grads
from dagbldr.utils import iterate_function
from dagbldr.nodes import relu_layer, linear_layer, gaussian_log_sample_layer
from dagbldr.nodes import sigmoid_layer
from dagbldr.nodes import binary_crossentropy, gaussian_log_kl
from dagbldr.optimizers import sgd

# Common between tests
digits = load_digits()
X = digits["data"]
y = digits["target"]
n_classes = len(set(y))
y = convert_to_one_hot(y, n_classes)


def test_vae():
    minibatch_size = 100
    random_state = np.random.RandomState(1999)
    graph = OrderedDict()

    X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph)

    l1_enc = relu_layer([X_sym, y_sym], graph, 'l1_enc', proj_dim=20,
                        random_state=random_state)
    mu = linear_layer([l1_enc], graph, 'mu', proj_dim=10,
                      random_state=random_state)
    log_sigma = linear_layer([l1_enc], graph, 'log_sigma', proj_dim=10,
                             random_state=random_state)
    samp = gaussian_log_sample_layer([mu], [log_sigma], graph,
                                     'gaussian_log_sample',
                                     random_state=random_state)
    l1_dec = relu_layer([samp], graph, 'l1_dec', proj_dim=20,
                        random_state=random_state)
    out = sigmoid_layer([l1_dec], graph, 'out', proj_dim=X.shape[1],
                        random_state=random_state)

    kl = gaussian_log_kl([mu], [log_sigma], graph, 'gaussian_kl').mean()
    cost = binary_crossentropy(out, X_sym).mean() + kl
    params, grads = get_params_and_grads(graph, cost)
    learning_rate = 0.001
    opt = sgd(params)
    updates = opt.updates(params, grads, learning_rate)

    train_function = theano.function([X_sym, y_sym], [cost], updates=updates,
                                     mode="FAST_COMPILE")

    iterate_function(train_function, [X, y], minibatch_size,
                     list_of_output_names=["cost"], n_epochs=1)

if __name__ == "__main__":
    test_vae()
