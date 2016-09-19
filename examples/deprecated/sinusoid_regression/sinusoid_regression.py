from collections import OrderedDict
import numpy as np
import theano

from dagbldr.optimizers import rmsprop
from dagbldr.utils import add_datasets_to_graph, get_params_and_grads
from dagbldr.utils import create_checkpoint_dict
from dagbldr.utils import fixed_n_epochs_trainer
from dagbldr.nodes import tanh_layer, linear_layer
from dagbldr.nodes import squared_error

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

sine_x, sine_y = make_noisy_sinusoid()
# Make 1 minibatch with feature dimension 1
sine_x = sine_x[:, None]
sine_y = sine_y[:, None]

# No real validation here
train_indices = np.arange(len(sine_y))
valid_indices = np.arange(len(sine_y))
X = sine_x
y = sine_y

# graph holds information necessary to build layers from parents
graph = OrderedDict()
X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph)
# random state so script is deterministic
random_state = np.random.RandomState(1999)

minibatch_size = len(sine_y)
n_hid = 20
n_out = 1

l1 = tanh_layer([X_sym], graph, 'l1', proj_dim=n_hid, random_state=random_state)
y_pred = linear_layer([l1], graph, 'y_pred',  proj_dim=n_out,
                      random_state=random_state)
cost = ((y_pred - y_sym) ** 2).mean()
# Can also define cost this way using dagbldr
# cost = squared_error(y_pred, y_sym).mean()
params, grads = get_params_and_grads(graph, cost)

learning_rate = 1E-3
momentum = 0.8
opt = rmsprop(params, learning_rate, momentum)
updates = opt.updates(params, grads)

fit_function = theano.function([X_sym, y_sym], [cost], updates=updates)
cost_function = theano.function([X_sym, y_sym], [cost])
predict_function = theano.function([X_sym], [y_pred])

checkpoint_dict = create_checkpoint_dict(locals())

epoch_results = fixed_n_epochs_trainer(
    fit_function, cost_function, train_indices, valid_indices,
    checkpoint_dict, [X, y],
    minibatch_size,
    list_of_train_output_names=["train_cost"],
    valid_output_name="valid_cost",
    n_epochs=1000)

# pred_sine_y = predict_function(sine_x)[0]
# plt.plot(sine_x, pred_sine_y, "o", color="red", alpha=0.3)
# plt.plot(sine_x, sine_y, "o", color="steelblue", alpha=0.3)
