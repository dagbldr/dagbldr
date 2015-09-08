from collections import OrderedDict
import numpy as np
import theano
from theano import tensor
from nose.tools import assert_raises
from numpy.testing import assert_almost_equal

from dagbldr.datasets import load_digits
from dagbldr.optimizers import sgd
from dagbldr.utils import add_embedding_datasets_to_graph, convert_to_one_hot
from dagbldr.utils import add_datasets_to_graph
from dagbldr.utils import get_params_and_grads, make_embedding_minibatch
from dagbldr.nodes import fixed_projection_layer, embedding_layer
from dagbldr.nodes import projection_layer, linear_layer, softmax_layer
from dagbldr.nodes import softmax_zeros_layer
from dagbldr.nodes import sigmoid_layer, tanh_layer, softplus_layer
from dagbldr.nodes import exp_layer, relu_layer, dropout_layer
from dagbldr.nodes import softmax_sample_layer, gaussian_sample_layer
from dagbldr.nodes import gaussian_log_sample_layer

# Common between tests
digits = load_digits()
X = digits["data"]
y = digits["target"]
n_classes = len(set(y))
y = convert_to_one_hot(y, n_classes)


def run_common_layer(layer):
    random_state = np.random.RandomState(42)
    graph = OrderedDict()
    X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph)
    single_o = layer([X_sym], graph, 'single', proj_dim=5,
                     random_state=random_state)
    concat_o = layer([X_sym, y_sym], graph, 'concat', proj_dim=5,
                     random_state=random_state)
    # Check that things can be reused
    repeated_o = layer([X_sym], graph, 'single', strict=False)

    # Check that strict mode raises an error if repeated
    assert_raises(AttributeError, layer, [X_sym], graph, 'concat')

    f = theano.function([X_sym, y_sym], [single_o, concat_o, repeated_o],
                        mode="FAST_COMPILE")
    single, concat, repeat = f(X, y)
    assert_almost_equal(single, repeat)


def test_dropout_layer():
    random_state = np.random.RandomState(42)
    graph = OrderedDict()
    X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph)
    on_off = tensor.iscalar()
    dropped = dropout_layer([X_sym], graph, 'dropout', on_off,
                            random_state=random_state)

    f = theano.function([X_sym, on_off], [dropped], mode="FAST_COMPILE")
    drop = f(np.ones_like(X), 1)[0]
    full = f(np.ones_like(X), 0)[0]
    # Make sure drop switch works
    assert_almost_equal((full.sum() / 2) / drop.sum(), 1., decimal=2)


def test_batch_normalization():
    random_state = np.random.RandomState(1999)
    graph = OrderedDict()
    X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph,
                                         list_of_test_values=[X, y])
    on_off = tensor.iscalar()
    on_off.tag.test_value = 1
    l1 = relu_layer([X_sym], graph, "proj", proj_dim=5,
                    batch_normalize=True, mode_switch=on_off,
                    random_state=random_state)
    l2 = relu_layer([l1], graph, "proj2", proj_dim=5,
                    batch_normalize=True, mode_switch=on_off,
                    random_state=random_state)
    f = theano.function([X_sym, on_off], [l2], mode="FAST_COMPILE")
    params, grads = get_params_and_grads(graph, l2.mean())
    opt = sgd(params)
    updates = opt.updates(params, grads, .1)
    train_f = theano.function([X_sym, on_off], [l2], mode="FAST_COMPILE",
                              updates=updates)
    valid_f = theano.function([X_sym, on_off], [l2], mode="FAST_COMPILE")
    X1 = random_state.rand(*X.shape)
    X2 = np.vstack([X1, .5 * X1])
    t1 = train_f(X1, 0)[0]
    t2 = valid_f(X1, 1)[0]
    t3 = train_f(X2, 0)[0]
    t4 = valid_f(X1, 1)[0]
    t5 = valid_f(X1, 1)[0]
    assert_almost_equal(t4, t5)
    assert_raises(AssertionError, assert_almost_equal, t2, t4)


def test_embedding_layer():
    random_state = np.random.RandomState(1999)
    graph = OrderedDict()
    max_index = 100
    proj_dim = 12
    fake_str_int = [[1, 5, 7, 1, 6, 2], [2, 3, 6, 2], [3, 3, 3, 3, 3, 3, 3]]
    minibatch, mask = make_embedding_minibatch(
        fake_str_int, slice(0, 3))
    (emb_slices,), (emb_mask,) = add_embedding_datasets_to_graph(
        [minibatch], [mask], "emb", graph)
    emb = embedding_layer(emb_slices, max_index, proj_dim, graph,
                          'emb', random_state)
    followup_dim = 17
    proj = linear_layer([emb], graph, 'proj', followup_dim,
                        random_state=random_state)
    f = theano.function(emb_slices, [proj], mode="FAST_COMPILE")
    out, = f(*minibatch)
    assert(out.shape[-1] == 17)
    assert(out.shape[-2] == len(fake_str_int))


def test_fixed_projection_layer():
    random_state = np.random.RandomState(1999)
    rand_projection = random_state.randn(64, 12)

    graph = OrderedDict()
    X_sym = add_datasets_to_graph([X], ["X"], graph)
    out = fixed_projection_layer([X_sym], rand_projection,
                                 graph, 'proj')
    out2 = fixed_projection_layer([X_sym], rand_projection,
                                  graph, 'proj',
                                  pre=rand_projection[:, 0])
    out3 = fixed_projection_layer([X_sym], rand_projection,
                                  graph, 'proj',
                                  post=rand_projection[0])
    final = linear_layer([out2], graph, 'linear', 17,
                         random_state=random_state)
    # Test that it compiles with and without bias
    f = theano.function([X_sym], [out, out2, out3, final],
                        mode="FAST_COMPILE")

    # Test updates
    params, grads = get_params_and_grads(
        graph, final.mean())
    opt = sgd(params)
    updates = opt.updates(params, grads, .1)
    f2 = theano.function([X_sym], [out2, final],
                         updates=updates)
    ret = f(np.ones_like(X))[0]
    assert ret.shape[1] != X.shape[1]
    ret2 = f(np.ones_like(X))[1]
    assert ret.shape[1] != X.shape[1]
    out1, final1 = f2(X)
    out2, final2 = f2(X)

    # Make sure fixed basis is unchanged
    assert_almost_equal(out1, out2)

    # Make sure linear layer is updated
    assert_raises(AssertionError, assert_almost_equal, final1, final2)


def test_projection_layer():
    run_common_layer(projection_layer)


def test_linear_layer():
    run_common_layer(linear_layer)


def test_sigmoid_layer():
    run_common_layer(sigmoid_layer)


def test_tanh_layer():
    run_common_layer(tanh_layer)


def test_softplus_layer():
    run_common_layer(softplus_layer)


def test_exp_layer():
    run_common_layer(exp_layer)


def test_relu_layer():
    run_common_layer(relu_layer)


def test_softmax_layer():
    run_common_layer(softmax_layer)


def test_softmax_zeros_layer():
    graph = OrderedDict()
    X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph)
    single_o = softmax_zeros_layer([X_sym], graph, 'single', proj_dim=5)
    concat_o = softmax_zeros_layer([X_sym, y_sym], graph, 'concat', proj_dim=5)
    # Check that things can be reused
    repeated_o = softmax_layer([X_sym], graph, 'single', strict=False)

    # Check that strict mode raises an error if repeated
    assert_raises(AttributeError, softmax_layer, [X_sym], graph, 'concat')

    f = theano.function([X_sym, y_sym], [single_o, concat_o, repeated_o],
                        mode="FAST_COMPILE")
    single, concat, repeat = f(X, y)
    assert_almost_equal(single, repeat)


def test_softmax_sample_layer():
    random_state = np.random.RandomState(42)
    graph = OrderedDict()
    X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph)
    softmax = softmax_layer([X_sym], graph, 'softmax', proj_dim=20,
                            random_state=random_state)
    samp = softmax_sample_layer([softmax], graph, 'softmax_sample',
                                random_state=random_state)
    out = linear_layer([samp], graph, 'out', proj_dim=10,
                       random_state=random_state)
    f = theano.function([X_sym], [out], mode="FAST_COMPILE")


def test_gaussian_sample_layer():
    random_state = np.random.RandomState(42)
    graph = OrderedDict()
    X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph)
    mu = linear_layer([X_sym], graph, 'mu', proj_dim=20,
                      random_state=random_state)
    sigma = softplus_layer([X_sym], graph, 'sigma', proj_dim=20,
                           random_state=random_state)
    samp = gaussian_sample_layer([mu], [sigma], graph, 'gaussian_sample',
                                 random_state=random_state)
    out = linear_layer([samp], graph, 'out', proj_dim=10,
                       random_state=random_state)
    f = theano.function([X_sym], [out], mode="FAST_COMPILE")


def test_gaussian_log_sample_layer():
    random_state = np.random.RandomState(42)
    graph = OrderedDict()
    X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph)
    mu = linear_layer([X_sym], graph, 'mu', proj_dim=20,
                      random_state=random_state)
    log_sigma = linear_layer([X_sym], graph, 'log_sigma', proj_dim=20,
                             random_state=random_state)
    samp = gaussian_log_sample_layer([mu], [log_sigma], graph,
                                     'gaussian_sample',
                                     random_state=random_state)
    out = linear_layer([samp], graph, 'out', proj_dim=10,
                       random_state=random_state)
    f = theano.function([X_sym], [out], mode="FAST_COMPILE")
