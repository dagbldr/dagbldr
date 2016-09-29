#!/usr/bin/env python
import numpy as np
import theano
from theano import tensor

from dagbldr.utils import get_params
from dagbldr.utils import create_checkpoint_dict

from dagbldr.nodes import tanh
from dagbldr.nodes import log_gaussian_mixture
from dagbldr.nodes import log_gaussian_mixture_cost

from dagbldr.datasets import minibatch_iterator
from dagbldr.training import TrainingLoop
from dagbldr.optimizers import adadelta


# This example based on a great tutorial on Mixture Density Networks in TF
# http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/


def make_noisy_sinusoid(n_samples=1000):
    random_state = np.random.RandomState(1999)
    x = random_state.uniform(-10, 10, size=(n_samples,))
    r = random_state.normal(size=(n_samples,))
    # Sinusoid with frequency ~0.75, amplitude 7, linear trend of .5
    # and additive noise
    y = np.sin(0.75 * x) * 7 + .5 * x + r
    x = x.astype("float32")
    y = y.astype("float32")
    return x, y

sine_x, sine_y = make_noisy_sinusoid(n_samples=10000)
# Swap X and Y to create a one to many relationship

sine_x, sine_y = sine_y, sine_x
# Make 1 minibatch with feature dimension 1
sine_x = sine_x[:, None]
sine_y = sine_y[:, None]

X = sine_x
y = sine_y
X_sym = tensor.fmatrix()
y_sym = tensor.fmatrix()

# random state so script is deterministic
random_state = np.random.RandomState(1999)

minibatch_size = len(sine_y) / 20
n_hid = 20
n_out = 1

l1 = tanh([X_sym], [X.shape[1]], proj_dim=n_hid, name='l1', random_state=random_state)
coeffs, mus, log_sigmas = log_gaussian_mixture(
    [l1], [n_hid], proj_dim=1, n_components=24, name='mdn',
    random_state=random_state)
cost = log_gaussian_mixture_cost(coeffs, mus, log_sigmas, y_sym).mean()
params = list(get_params().values())
grads = theano.grad(cost, params)

opt = adadelta(params)
updates = opt.updates(params, grads)

fit_function = theano.function([X_sym, y_sym], [cost], updates=updates)
cost_function = theano.function([X_sym, y_sym], [cost])
predict_function = theano.function([X_sym], [coeffs, mus, log_sigmas])

checkpoint_dict = create_checkpoint_dict(locals())

train_itr = minibatch_iterator([X, y], minibatch_size, axis=0)
valid_itr = minibatch_iterator([X, y], minibatch_size, axis=0)


def train_loop(itr):
    X_mb, y_mb = next(itr)
    return fit_function(X_mb, y_mb)


def valid_loop(itr):
    X_mb, y_mb = next(itr)
    return cost_function(X_mb, y_mb)

TL = TrainingLoop(train_loop, train_itr,
                  valid_loop, valid_itr,
                  n_epochs=1000,
                  checkpoint_every_n_epochs=50,
                  checkpoint_dict=checkpoint_dict)
epoch_results = TL.run()


def plot_log_gaussian_mixture_heatmap(xlim=(-10, 10), ylim=(-10, 10)):
    def log_gaussian(x, mu, log_sigma):
        return 1./(2 * np.pi * np.exp(log_sigma)) * np.exp(
            -(x - mu) ** 2 / (2 * np.exp(log_sigma)))
    xpts = np.linspace(xlim[0], xlim[1], 1000).astype("float32")
    coeffs, mus, log_sigmas = predict_function(xpts[:, None])
    mus = mus.squeeze()
    log_sigmas = log_sigmas.squeeze()
    ypts = np.linspace(ylim[0], ylim[1], 1000)
    z = np.zeros((len(xpts), len(ypts)))
    for k in range(coeffs.shape[-1]):
        print("Calculating mixture component %i" % k)
        for i, y in enumerate(ypts):
            coeff = coeffs[i, k]
            mu = mus[i, k]
            log_sigma = log_sigmas[i, k]
            contrib = coeff * log_gaussian(ypts[::-1], mu, log_sigma)
            z[:, i] += contrib
    return z


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
z = plot_log_gaussian_mixture_heatmap()
plt.imshow(z)
plt.savefig('out.png')
plt.show()
