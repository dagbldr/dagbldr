"""
Simple GRU RNNs for solving the QA tasks from:
"Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
J. Weston, A. Bordes, S. Chopra, T. Mikolov, A. Rush
http://arxiv.org/abs/1502.05698

Inspired by (and approximately replicating) the blog post by Stephen Merity:
http://smerity.com/articles/2015/keras_qa.html

This blog post was turned into a Keras example:
https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py
"""

from collections import OrderedDict
import theano
import numpy as np

from dagbldr.datasets import fetch_babi
from dagbldr.utils import make_embedding_minibatch, make_minibatch
from dagbldr.utils import add_embedding_datasets_to_graph, add_datasets_to_graph
from dagbldr.utils import fixed_n_epochs_trainer
from dagbldr.utils import get_params_and_grads
from dagbldr.nodes import gru_recurrent_layer, softmax_layer
from dagbldr.nodes import embedding_layer, categorical_crossentropy
from dagbldr.optimizers import adadelta


babi = fetch_babi(task_number=2)
X_story = babi["stories"]
X_query = babi["queries"]
y_answer = babi["target"]
train_indices = babi["train_indices"]
valid_indices = babi["valid_indices"]
vocab_size = babi["vocabulary_size"]

random_state = np.random.RandomState(1999)
graph = OrderedDict()

minibatch_size = 32
n_emb = 50
n_hid = 100
X_story_mb, X_story_mask = make_embedding_minibatch(
    X_story, slice(0, minibatch_size))
X_query_mb, X_query_mask = make_embedding_minibatch(
    X_query, slice(0, minibatch_size))

embedding_datasets = [X_story_mb, X_query_mb]
masks = [X_story_mask, X_query_mask]
r = add_embedding_datasets_to_graph(embedding_datasets, masks, "babi_data",
                                    graph)
(X_story_syms, X_query_syms), (X_story_mask_sym, X_query_mask_sym) = r

y_sym = add_datasets_to_graph([y_answer], ["y"], graph)


l1_story = embedding_layer(X_story_syms, vocab_size, n_emb, graph, 'l1_story',
                           random_state=random_state)
masked_story = X_story_mask_sym.dimshuffle(0, 1, 'x') * l1_story
h_story = gru_recurrent_layer([masked_story], X_story_mask_sym, n_hid, graph,
                              'story_rec', random_state)

l1_query = embedding_layer(X_query_syms, vocab_size, n_emb, graph, 'l1_query',
                           random_state)
h_query = gru_recurrent_layer([l1_query], X_query_mask_sym, n_hid, graph,
                              'query_rec', random_state)
y_pred = softmax_layer([h_query[-1], h_story[-1]], graph, 'y_pred',
                       y_answer.shape[1], random_state=random_state)
cost = categorical_crossentropy(y_pred, y_sym).mean()
params, grads = get_params_and_grads(graph, cost)

opt = adadelta(params)
updates = opt.updates(params, grads)
print("Compiling fit...")
fit_function = theano.function(X_story_syms + [X_story_mask_sym] + X_query_syms
                               + [X_query_mask_sym, y_sym], [cost],
                               updates=updates)
print("Compiling cost...")
cost_function = theano.function(X_story_syms + [X_story_mask_sym] + X_query_syms
                                + [X_query_mask_sym, y_sym], [cost])
print("Compiling predict...")
predict_function = theano.function(X_story_syms + [X_story_mask_sym] +
                                   X_query_syms + [X_query_mask_sym], [y_pred])


def error(*args):
    xargs = args[:-1]
    y = args[-1]
    final_args = xargs
    y_pred = predict_function(*final_args)[0]
    return 1 - np.mean((np.argmax(
        y_pred, axis=1).ravel()) == (np.argmax(y, axis=1).ravel()))

checkpoint_dict = {}
epoch_results = fixed_n_epochs_trainer(
    fit_function, error, train_indices, valid_indices, checkpoint_dict,
    [X_story, X_query, y_answer],
    minibatch_size,
    list_of_minibatch_functions=[make_embedding_minibatch,
                                 make_embedding_minibatch,
                                 make_minibatch],
    list_of_train_output_names=["cost"],
    valid_output_name="valid_error", n_epochs=100)
