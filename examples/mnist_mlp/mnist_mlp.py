import numpy as np
import theano
from theano import tensor

from dagbldr.datasets import fetch_mnist

from dagbldr.utils import get_params
from dagbldr.utils import get_weights
from dagbldr.utils import convert_to_one_hot
from dagbldr.utils import create_checkpoint_dict

from dagbldr.nodes import tanh, softmax_zeros
from dagbldr.nodes import categorical_crossentropy

from dagbldr.datasets import minibatch_iterator
from dagbldr.training import TrainingLoop
from dagbldr.optimizers import sgd

mnist = fetch_mnist()
X = mnist["data"].astype("float32")
y = mnist["target"]
n_targets = 10
y = convert_to_one_hot(y, n_targets).astype("float32")

X_sym = tensor.fmatrix()
y_sym = tensor.fmatrix()

# random state so script is deterministic
random_state = np.random.RandomState(1999)

minibatch_size = 20
n_hid = 500

l1 = tanh([X_sym], [X.shape[1]], proj_dim=n_hid, name='l1',
          random_state=random_state)
y_pred = softmax_zeros([l1], [n_hid], proj_dim=n_targets, name='y_pred')
nll = categorical_crossentropy(y_pred, y_sym).mean()
weights = get_weights(skip_regex=None).values()
L2 = sum([(w ** 2).sum() for w in weights])
cost = nll + .0001 * L2

params = list(get_params().values())
grads = theano.grad(cost, params)

learning_rate = 0.01
opt = sgd(params, learning_rate)
updates = opt.updates(params, grads)

fit_function = theano.function([X_sym, y_sym], [cost], updates=updates)
cost_function = theano.function([X_sym, y_sym], [cost])
predict_function = theano.function([X_sym], [y_pred])

checkpoint_dict = create_checkpoint_dict(locals())


train_itr = minibatch_iterator([X, y], minibatch_size, axis=0,
                               stop_index=60000)
valid_itr = minibatch_iterator([X, y], minibatch_size, axis=0,
                               start_index=60000)


def train_loop(itr):
    X_mb, y_mb = next(itr)
    return fit_function(X_mb, y_mb)


def valid_loop(itr):
    X_mb, y_mb = next(itr)
    y_pred = predict_function(X_mb)[0]
    y_pred_inds = np.argmax(y_pred, axis=1).ravel()
    y_inds = np.argmax(y_mb, axis=1).ravel()
    return [1 - np.mean((y_pred_inds == y_inds).astype("float32"))]


TL = TrainingLoop(train_loop, train_itr,
                  valid_loop, valid_itr,
                  n_epochs=1000,
                  checkpoint_every_n_epochs=50,
                  checkpoint_dict=checkpoint_dict)
epoch_results = TL.run()
