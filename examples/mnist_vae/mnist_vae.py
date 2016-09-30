import numpy as np
import theano
from theano import tensor

from dagbldr.datasets import fetch_binarized_mnist, minibatch_iterator
from dagbldr.utils import get_params
from dagbldr.utils import TrainingLoop
from dagbldr.utils import create_checkpoint_dict

from dagbldr.nodes import softplus
from dagbldr.nodes import sigmoid
from dagbldr.nodes import linear
from dagbldr.nodes import gaussian_log_sample
from dagbldr.nodes import gaussian_log_kl
from dagbldr.nodes import binary_crossentropy

from dagbldr.optimizers import adam

mnist = fetch_binarized_mnist()

X = mnist["data"].astype("float32")
X_sym = tensor.fmatrix()

# random state so script is deterministic
random_state = np.random.RandomState(1999)

minibatch_size = 100
n_code = 100
n_hid = 200
width = 28
height = 28
n_input = width * height

# encode path aka q
l1_enc = softplus([X_sym], [X.shape[1]], proj_dim=n_hid, name='l1_enc',
                  random_state=random_state)
l2_enc = softplus([l1_enc], [n_hid], proj_dim=n_hid, name='l2_enc',
                  random_state=random_state)
code_mu = linear([l2_enc], [n_hid], proj_dim=n_code, name='code_mu',
                 random_state=random_state)
code_log_sigma = linear([l2_enc], [n_hid], proj_dim=n_code,
                        name='code_log_sigma', random_state=random_state)
kl = gaussian_log_kl([code_mu], [code_log_sigma]).mean()
sample_state = np.random.RandomState(2177)
samp = gaussian_log_sample([code_mu], [code_log_sigma], name='samp',
                           random_state=sample_state)

# decode path aka p
l1_dec = softplus([samp], [n_code], proj_dim=n_hid, name='l1_dec',
                  random_state=random_state)
l2_dec = softplus([l1_dec], [n_hid], proj_dim=n_hid, name='l2_dec',
                  random_state=random_state)
out = sigmoid([l2_dec], [n_hid], proj_dim=X.shape[1], name='out',
              random_state=random_state)

nll = binary_crossentropy(out, X_sym).mean()
# See https://arxiv.org/pdf/1406.5298v2.pdf, eq 5
# log p(x | z) = -nll so swap sign
# want to minimize cost in optimization so multiply by -1
# cost = -1 * (-nll - kl)
cost = nll + kl
params = list(get_params().values())
grads = theano.grad(cost, params)

learning_rate = 0.0003
opt = adam(params, learning_rate)
updates = opt.updates(params, grads)

fit_function = theano.function([X_sym], [nll, kl, nll + kl], updates=updates)
cost_function = theano.function([X_sym], [nll + kl])
encode_function = theano.function([X_sym], [code_mu, code_log_sigma])
decode_function = theano.function([samp], [out])


checkpoint_dict = create_checkpoint_dict(locals())

train_itr = minibatch_iterator([X], minibatch_size,
                               stop_index=60000, axis=0)
valid_itr = minibatch_iterator([X], minibatch_size,
                               start_index=60000, stop_index=70000,
                               axis=0)


def train_loop(itr):
    X_mb = next(itr)
    return [fit_function(X_mb)[2]]


def valid_loop(itr):
    X_mb = next(itr)
    return cost_function(X_mb)


TL = TrainingLoop(train_loop, train_itr,
                  valid_loop, valid_itr,
                  n_epochs=5000,
                  checkpoint_every_n_epochs=50,
                  checkpoint_dict=checkpoint_dict)
epoch_results = TL.run()
