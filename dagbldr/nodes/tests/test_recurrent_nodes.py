import numpy as np
import theano
from theano import tensor

from dagbldr.nodes import simple
from dagbldr.nodes import simple_fork
from dagbldr.nodes import linear
from dagbldr.optimizers import sgd
from dagbldr import get_params

from dagbldr import del_shared


def make_sines(n_timesteps, n_offsets, harmonic=False, square=False):
    # Generate sinewaves offset in phase
    n_full = n_timesteps
    d1 = 3 * np.arange(n_full) / (2 * np.pi)
    d2 = 3 * np.arange(n_offsets) / (2 * np.pi)
    full_sines = np.sin(np.array([d1] * n_offsets).T + d2).astype("float32")
    # Uncomment to add harmonics
    if harmonic:
        full_sines += np.sin(np.array([1.7 * d1] * n_offsets).T + d2)
        full_sines += np.sin(np.array([7.362 * d1] * n_offsets).T + d2)
    if square:
        full_sines[full_sines <= 0] = 0
        full_sines[full_sines > 0] = 1
    full_sines = full_sines[:, :, None]
    return full_sines

n_timesteps = 5
minibatch_size = 3
full_sines = make_sines(n_timesteps, minibatch_size)
X = full_sines[:-1]
y = full_sines[1:]

X_sym = tensor.tensor3()
y_sym = tensor.tensor3()


def run_simple():
    del_shared()
    n_in = X.shape[-1]
    n_hid = 20
    n_out = y.shape[-1]

    random_state = np.random.RandomState(42)
    h_init = np.zeros((minibatch_size, n_hid)).astype("float32")

    h0 = tensor.fmatrix()

    l1 = simple_fork([X_sym], [n_in], n_hid, name="l1",
                     random_state=random_state)

    def step(in_t, h_tm1):
        h_t = simple(in_t, h_tm1, n_hid, name="rec", random_state=random_state)
        return h_t

    h, _ = theano.scan(step, sequences=[l1], outputs_info=[h0])

    pred = linear([h], [n_hid], n_out, name="l2", random_state=random_state)
    cost = ((y_sym - pred) ** 2).sum()
    params = list(get_params().values())

    grads = tensor.grad(cost, params)
    learning_rate = 0.000000000001
    opt = sgd(params, learning_rate)
    updates = opt.updates(params, grads)

    f = theano.function([X_sym, y_sym, h0], [cost, h], updates=updates,
                        mode="FAST_COMPILE")
    f(X, y, h_init)
