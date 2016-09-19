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

from dagbldr.training import TrainingLoop
from dagbldr.optimizers import sgd

mnist = fetch_mnist()
train_indices = mnist["train_indices"]
valid_indices = mnist["valid_indices"]
X = mnist["data"].astype("float32")
y = mnist["target"]
n_targets = 10
y = convert_to_one_hot(y, n_targets).astype("float32")

X_sym = tensor.fmatrix()
y_sym = tensor.fmatrix()

# random state so script is deterministic
random_state = np.random.RandomState(1999)

minibatch_size = 20
n_hid = 1000

l1 = tanh([X_sym], [X.shape[1]], proj_dim=n_hid, name='l1',
          random_state=random_state)
y_pred = softmax_zeros([l1], [n_hid], proj_dim=n_targets, name='y_pred')
nll = categorical_crossentropy(y_pred, y_sym).mean()
weights = get_weights(skip_regex=None).values()
L2 = sum([(w ** 2).sum() for w in weights])
cost = nll + .0001 * L2

params = get_params().values()
grads = theano.grad(cost, params)

learning_rate = 0.01
opt = sgd(params, learning_rate)
updates = opt.updates(params, grads)

fit_function = theano.function([X_sym, y_sym], [cost], updates=updates)
cost_function = theano.function([X_sym, y_sym], [cost])
predict_function = theano.function([X_sym], [y_pred])

checkpoint_dict = create_checkpoint_dict(locals())


def error(*args):
    xargs = args[:-1]
    y = args[-1]
    final_args = xargs
    y_pred = predict_function(*final_args)[0]
    return 1 - np.mean((np.argmax(
        y_pred, axis=1).ravel()) == (np.argmax(y, axis=1).ravel()))


TL = TrainingLoop(fit_function, error, train_indices, valid_indices,
                  checkpoint_dict=checkpoint_dict,
                  minibatch_size=minibatch_size,
                  list_of_train_output_names=["train_cost"],
                  valid_output_name="valid_error",
                  n_epochs=1000,
                  optimizer_object=opt)
epoch_results = TL.run([X, y])
