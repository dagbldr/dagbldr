from dagbldr.datasets import fetch_fruitspeech
from dagbldr.nodes import log_ctc_cost
from dagbldr.nodes import maxout_layer
from dagbldr.nodes import bidirectional_gru_recurrent_layer, softmax_layer
from dagbldr.optimizers import adadelta
from dagbldr.utils import fixed_n_epochs_trainer
from dagbldr.utils import add_datasets_to_graph, make_masked_minibatch
from dagbldr.utils import get_params_and_grads

import numpy as np
import theano
from collections import OrderedDict

random_state = np.random.RandomState(1999)
graph = OrderedDict()

data = fetch_fruitspeech()
X = data["specgrams"]
y = data["target"]
vocab_size = data["vocabulary_size"]
vocab = data["vocabulary"]
train_indices = data["train_indices"]
valid_indices = data["valid_indices"]

minibatch_size = 10

X_mb, X_mb_mask = make_masked_minibatch(X, slice(0, minibatch_size))
y_mb, y_mb_mask = make_masked_minibatch(y, slice(0, minibatch_size))

n_hid = 500
n_out = vocab_size + 1

datasets_list = [X_mb, X_mb_mask, y_mb, y_mb_mask]
names_list = ["X", "X_mask", "y", "y_mask"]
X_sym, X_mask_sym, y_sym, y_mask_sym = add_datasets_to_graph(
    datasets_list, names_list, graph)

l1 = maxout_layer([X_sym], graph, 'l1', n_hid, random_state=random_state)
h = bidirectional_gru_recurrent_layer([l1], X_mask_sym, n_hid, graph,
                                      'l1_rec', random_state=random_state)
l2 = maxout_layer([h], graph, 'l2', n_hid, random_state=random_state)
y_pred = softmax_layer([l2], graph, 'softmax', n_out, random_state=random_state)

cost = log_ctc_cost(y_sym, y_mask_sym, y_pred, X_mask_sym).mean()
params, grads = get_params_and_grads(graph, cost)

opt = adadelta(params)
updates = opt.updates(params, grads)

checkpoint_dict = {}

fit_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym], [cost],
                               updates=updates)
cost_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym], [cost])
predict_function = theano.function([X_sym, X_mask_sym], [y_pred])


def prediction_strings(y_pred):
    indices = y_pred.argmax(axis=1)
    # remove blanks
    indices = indices[indices != vocab_size]
    non_ctc_string = "".join([vocab[i] for i in indices])
    # remove repeats
    not_same = np.where((indices[1:] != indices[:-1]))[0]
    last_char = ""
    if len(not_same) > 0:
        last_char = vocab[indices[-1]]
        indices = indices[not_same]
    s = "".join([vocab[i] for i in indices])
    ctc_string = s + last_char
    return ctc_string, non_ctc_string


def print_ctc_prediction(X_sym, X_mask_sym, y_sym, y_mask_sym):
    all_y_pred = predict_function(X_sym, X_mask_sym)[0]
    for n in range(all_y_pred.shape[1]):
        y_pred = all_y_pred[:, n]
        ctc_string, non_ctc_string = prediction_strings(y_pred)
        print(ctc_string)
        print(non_ctc_string)

fixed_n_epochs_trainer(fit_function, cost_function,
                       train_indices, valid_indices,
                       checkpoint_dict, [X, y], minibatch_size,
                       monitor_function=print_ctc_prediction,
                       list_of_minibatch_functions=[
                           make_masked_minibatch,
                           make_masked_minibatch],
                       list_of_train_output_names=["cost"],
                       valid_output_name="valid_cost",
                       valid_frequency=10,
                       n_epochs=1000)
