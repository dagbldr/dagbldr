# Author: Kyle Kastner
# License: BSD 3-clause
from dagbldr.datasets import make_ocr
from dagbldr.datasets import minibatch_iterator
from dagbldr.utils import convert_to_one_hot
from dagbldr.utils import add_datasets_to_graph
from dagbldr.utils import get_params_and_grads
from dagbldr.utils import TrainingLoop
from dagbldr.optimizers import adadelta
from dagbldr.nodes import location_attention_tanh_recurrent_layer
from dagbldr.nodes import sigmoid_layer
from dagbldr.nodes import binary_crossentropy
from dagbldr.nodes import masked_cost
import theano
import itertools
from collections import OrderedDict
import numpy as np


random_state = np.random.RandomState(1999)
graph = OrderedDict()
base_string = "cat"
true_strings = sorted(list(set(["".join(i) for i in [
    s for s in itertools.permutations(base_string)]])))
ocr = make_ocr(true_strings)
X = ocr["data"]
vocab = ocr["vocabulary"]
y = convert_to_one_hot(ocr["target"], n_classes=len(vocab)).astype(
    theano.config.floatX)
minibatch_size = mbs = 2
train_itr = minibatch_iterator([X, y], minibatch_size, make_mask=True, axis=1)
X_mb, X_mb_mask, y_mb, y_mb_mask = next(train_itr)
train_itr.reset()
valid_itr = minibatch_iterator([X, y], minibatch_size, make_mask=True, axis=1)
datasets_list = [X_mb, X_mb_mask, y_mb, y_mb_mask]
names_list = ["X", "X_mask", "y", "y_mask"]
X_sym, X_mask_sym, y_sym, y_mask_sym = add_datasets_to_graph(
    datasets_list, names_list, graph, list_of_test_values=datasets_list)

n_hid = 256
n_out = 8

h = location_attention_tanh_recurrent_layer(
    [X_sym], [y_sym], X_mask_sym, y_mask_sym, n_hid, graph, 'l1_att_rec',
    random_state=random_state)

X_hat = sigmoid_layer([h], graph, 'output', proj_dim=n_out,
                      random_state=random_state)
cost = binary_crossentropy(X_hat, X_sym).mean()
cost = masked_cost(cost, X_mask_sym).mean()
params, grads = get_params_and_grads(graph, cost)
opt = adadelta(params)
updates = opt.updates(params, grads)
fit_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym],
                               [cost], updates=updates)
valid_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym], [cost])

checkpoint_dict = {}
checkpoint_dict["fit_function"] = fit_function
checkpoint_dict["valid_function"] = valid_function
TL = TrainingLoop(fit_function, valid_function, train_itr, valid_itr,
                  checkpoint_dict=checkpoint_dict,
                  list_of_train_output_names=["train_cost"],
                  valid_output_name="valid_cost",
                  n_epochs=500,
                  optimizer_object=opt)
epoch_results = TL.run()
