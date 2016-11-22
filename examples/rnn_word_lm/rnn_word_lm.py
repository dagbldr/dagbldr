#!/usr/bin/env python
import numpy as np
import theano
from theano import tensor

from dagbldr.nodes import embed
from dagbldr.nodes import softmax
from dagbldr.nodes import lstm_fork
from dagbldr.nodes import lstm
from dagbldr.nodes import slice_state
from dagbldr.nodes import categorical_crossentropy

from dagbldr import get_params
from dagbldr.utils import create_checkpoint_dict

from dagbldr.optimizers import adam

from dagbldr.training import TrainingLoop
from dagbldr.datasets import fetch_lovecraft
from dagbldr.datasets import word_sequence_iterator

lovecraft = fetch_lovecraft()
n_timesteps = 50
minibatch_size = 20
vocabulary_size = 5000
train_itr = word_sequence_iterator(lovecraft["data"], minibatch_size,
                                   n_timesteps, max_vocabulary_size=vocabulary_size)
valid_itr = word_sequence_iterator(lovecraft["data"], minibatch_size,
                                   n_timesteps, max_vocabulary_size=vocabulary_size)

mb = train_itr.next()
X_mb = mb[:-1]
y_mb = mb[1:]
train_itr.reset()
n_classes = train_itr.n_classes

n_emb = 128
n_hid = 256
n_out = n_classes

train_h_init = np.zeros((minibatch_size, 2 * n_hid)).astype("float32")
valid_h_init = np.zeros((minibatch_size, 2 * n_hid)).astype("float32")

X_sym = tensor.fmatrix()
y_sym = tensor.fmatrix()
h0 = tensor.fmatrix()

X_sym.tag.test_value = X_mb
y_sym.tag.test_value = y_mb
h0.tag.test_value = train_h_init

random_state = np.random.RandomState(1999)

l1 = embed([X_sym], n_classes, n_emb, name="emb",
           random_state=random_state)
in_fork = lstm_fork([l1], [n_emb], n_hid, name="h1",
                    random_state=random_state)
def step(in_t, h_tm1):
    h_t = lstm(in_t, h_tm1, [n_hid], n_hid, name="lstm_l1", random_state=random_state)
    return h_t

h, _ = theano.scan(step,
                   sequences=[in_fork],
                   outputs_info=[h0])

h_o = slice_state(h, n_hid)

y_pred = softmax([h_o], [n_hid], n_classes, name="h2", random_state=random_state)
loss = categorical_crossentropy(y_pred, y_sym)
cost = loss.mean(axis=1).sum(axis=0)

params = list(get_params().values())
params = params
grads = tensor.grad(cost, params)

learning_rate = 0.0001
opt = adam(params, learning_rate)
updates = opt.updates(params, grads)

fit_function = theano.function([X_sym, y_sym, h0], [cost, h], updates=updates)
cost_function = theano.function([X_sym, y_sym, h0], [cost, h])
predict_function = theano.function([X_sym, h0], [y_pred, h])

def train_loop(itr):
    mb = next(itr)
    X_mb, y_mb = mb[:-1], mb[1:]
    cost, h = fit_function(X_mb, y_mb, train_h_init)
    train_h_init[:] = h[-1, :]
    return [cost]


def valid_loop(itr):
    mb = next(itr)
    X_mb, y_mb = mb[:-1], mb[1:]
    cost, h = cost_function(X_mb, y_mb, valid_h_init)
    valid_h_init[:] = h[-1, :]
    return [cost]


checkpoint_dict = create_checkpoint_dict(locals())

TL = TrainingLoop(train_loop, train_itr,
                  valid_loop, valid_itr,
                  n_epochs=100,
                  checkpoint_every_n_epochs=1,
                  checkpoint_dict=checkpoint_dict,
                  skip_minimums=True)
epoch_results = TL.run()
