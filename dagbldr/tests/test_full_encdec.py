import numpy as np
import theano

from theano.compat.python2x import OrderedDict
from dagbldr.datasets import load_mountains
from dagbldr.optimizers import sgd
from dagbldr.utils import add_datasets_to_graph, get_params_and_grads
from dagbldr.utils import iterate_function, make_character_level_from_text
from dagbldr.utils import gen_text_minibatch_func
from dagbldr.nodes import masked_cost, categorical_crossentropy
from dagbldr.nodes import softmax_layer, shift_layer
from dagbldr.nodes import gru_recurrent_layer, conditional_gru_recurrent_layer


# minibatch size
minibatch_size = 50

# Get data for lovecraft experiments
mountains = load_mountains()
text = mountains["data"]
# Get a tiny subset
text = text[:100]
cleaned, mfunc, inv_mfunc, mapper = make_character_level_from_text(text)
n_chars = len(mapper.keys())

# Necessary setup since text is done on per minibatch basis
text_minibatch_func = gen_text_minibatch_func(n_chars)
X = [l[:-10] for l in cleaned]
y = [l[-10:] for l in cleaned]
X_mb, X_mask = text_minibatch_func(X, 0, minibatch_size)
y_mb, y_mask = text_minibatch_func(y, 0, minibatch_size)


def test_gru_cond_recurrent():
    random_state = np.random.RandomState(1999)
    graph = OrderedDict()
    n_hid = 10
    n_out = n_chars

    # input (where first dimension is time)
    datasets_list = [X_mb, X_mask, y_mb, y_mask]
    names_list = ["X", "X_mask", "y", "y_mask"]
    test_values_list = [X, X_mask, y, y_mask]
    X_sym, X_mask_sym, y_sym, y_mask_sym = add_datasets_to_graph(
        datasets_list, names_list, graph, list_of_test_values=test_values_list)

    h = gru_recurrent_layer([X_sym], X_mask_sym, n_hid, graph, 'l1_end',
                            random_state)

    shifted_y_sym = shift_layer([y_sym], graph, 'shift')

    h_dec, context = conditional_gru_recurrent_layer([y_sym], [h], y_mask_sym,
                                                     n_hid, graph, 'l2_dec',
                                                     random_state)

    # linear output activation
    y_hat = softmax_layer([h_dec, context, shifted_y_sym], graph, 'l2_proj',
                          n_out, random_state)

    # error between output and target
    cost = categorical_crossentropy(y_hat, y_sym)
    cost = masked_cost(cost, y_mask_sym).mean()
    # Parameters of the model
    params, grads = get_params_and_grads(graph, cost)

    # Use stochastic gradient descent to optimize
    opt = sgd(params)
    learning_rate = 0.01
    updates = opt.updates(params, grads, learning_rate)

    fit_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym],
                                   [cost], updates=updates,
                                   mode="FAST_COMPILE")

    iterate_function(fit_function, [X, y], minibatch_size,
                     list_of_minibatch_functions=[text_minibatch_func],
                     list_of_output_names=["cost"], n_epochs=1)
