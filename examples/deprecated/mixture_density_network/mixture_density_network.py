from collections import OrderedDict
import numpy as np
import theano

from dagbldr.optimizers import adadelta
from dagbldr.utils import add_datasets_to_graph, get_params_and_grads
from dagbldr.utils import create_checkpoint_dict
from dagbldr.utils import TrainingLoop
from dagbldr.datasets import minibatch_iterator
from dagbldr.nodes import tanh_layer
from dagbldr.nodes import log_gaussian_mixture_layer, log_gaussian_mixture_cost

# This example based on a great tutorial on Mixture Density Networks in TF
# http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/


def make_noisy_sinusoid(n_samples=1000):
    random_state = np.random.RandomState(1999)
    x = random_state.uniform(-10, 10, size=(n_samples,))
    r = random_state.normal(size=(n_samples,))
    # Sinusoid with frequency ~0.75, amplitude 7, linear trend of .5
    # and additive noise
    y = np.sin(0.75 * x) * 7 + .5 * x + r
    x = x.astype(theano.config.floatX)
    y = y.astype(theano.config.floatX)
    return x, y

sine_x, sine_y = make_noisy_sinusoid(n_samples=10000)
# Swap X and Y to create a one to many relationship

sine_x, sine_y = sine_y, sine_x
# Make 1 minibatch with feature dimension 1
sine_x = sine_x[:, None]
sine_y = sine_y[:, None]

X = sine_x
y = sine_y

# graph holds information necessary to build layers from parents
graph = OrderedDict()
X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph,
                                     list_of_test_values=[sine_x, sine_y])
# random state so script is deterministic
random_state = np.random.RandomState(1999)

minibatch_size = len(sine_y) / 20
n_hid = 20
n_out = 1

l1 = tanh_layer([X_sym], graph, 'l1', proj_dim=n_hid, random_state=random_state)
coeffs, mus, log_sigmas = log_gaussian_mixture_layer(
    [l1], graph, 'mdn', proj_dim=1, n_components=24, random_state=random_state)
cost = log_gaussian_mixture_cost(coeffs, mus, log_sigmas, y_sym).mean()
params, grads = get_params_and_grads(graph, cost)

opt = adadelta(params)
updates = opt.updates(params, grads)

fit_function = theano.function([X_sym, y_sym], [cost], updates=updates)
cost_function = theano.function([X_sym, y_sym], [cost])
predict_function = theano.function([X_sym], [coeffs, mus, log_sigmas])

checkpoint_dict = create_checkpoint_dict(locals())

train_itr = minibatch_iterator([X, y], minibatch_size, axis=0)
valid_itr = minibatch_iterator([X, y], minibatch_size, axis=0)

next(train_itr)

TL = TrainingLoop(fit_function, cost_function,
                  train_itr, valid_itr,
                  checkpoint_dict=checkpoint_dict,
                  list_of_train_output_names=["train_cost"],
                  valid_output_name="valid_cost",
                  n_epochs=1000)
epoch_results = TL.run()


def plot_log_gaussian_mixture_heatmap(xlim=(-10, 10), ylim=(-10, 10)):
    def log_gaussian(x, mu, log_sigma):
        return 1./(2 * np.pi * np.exp(log_sigma)) * np.exp(
            -(x - mu) ** 2 / (2 * np.exp(log_sigma)))
    xpts = np.linspace(xlim[0], xlim[1], 1000)
    coeffs, mus, log_sigmas = predict_function(xpts[:, None])
    mus = mus.squeeze()
    log_sigmas = log_sigmas.squeeze()
    ypts = np.linspace(ylim[0], ylim[1], 1000)
    z = np.zeros((len(xpts), len(ypts)))
    for k in range(coeffs.shape[-1]):
        print("Mixture component %i" % k)
        for i, y in enumerate(ypts):
            coeff = coeffs[i, k]
            mu = mus[i, k]
            log_sigma = log_sigmas[i, k]
            contrib = coeff * log_gaussian(ypts[::-1], mu, log_sigma)
            z[:, i] += contrib
    return z


# import matplotlib.pyplot as plt
# z = plot_log_gaussian_mixture_heatmap()
# plt.imshow(z)
# plt.show()
