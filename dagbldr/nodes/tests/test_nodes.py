import numpy as np
import theano
from theano import tensor
from nose.tools import assert_raises
from numpy.testing import assert_almost_equal

from dagbldr.datasets import load_digits
from dagbldr import get_params, del_shared
from dagbldr.optimizers import sgd
from dagbldr.utils import convert_to_one_hot
from dagbldr.nodes import fixed_projection
from dagbldr.nodes import linear
from dagbldr.nodes import projection
from dagbldr.nodes import softmax
from dagbldr.nodes import sigmoid
from dagbldr.nodes import tanh
from dagbldr.nodes import relu
from dagbldr.nodes import softplus
from dagbldr.nodes import exponential
from dagbldr.nodes import maxout
from dagbldr.nodes import softmax_zeros
from dagbldr.nodes import log_gaussian_mixture
from dagbldr.nodes import bernoulli_and_correlated_log_gaussian_mixture
from dagbldr.nodes import conv2d
from dagbldr.nodes import pool2d

# Common between tests
digits = load_digits()
X = digits["data"].astype("float32")
y = digits["target"]
n_classes = len(set(y))
y = convert_to_one_hot(y, n_classes).astype("float32")
X_sym = tensor.fmatrix()
y_sym = tensor.fmatrix()


def run_common(layer):
    random_state = np.random.RandomState(42)

    single_o = layer([X_sym], [X.shape[1]], 5, 'single',
                     random_state=random_state)
    concat_o = layer([X_sym, y_sym], [X.shape[1], y.shape[1]], 5, 'concat',
                     random_state=random_state)
    # Check that things can be reused
    repeated_o = layer([X_sym], [X.shape[1]], 5, 'single', strict=False)

    # Check that strict mode raises an error if repeated
    assert_raises(AttributeError, layer, [X_sym], [X.shape[1]], 5, 'concat')

    f = theano.function([X_sym, y_sym], [single_o, concat_o, repeated_o],
                        mode="FAST_COMPILE")
    single, concat, repeat = f(X, y)
    assert_almost_equal(single, repeat)

'''
def test_dropout():
    random_state = np.random.RandomState(42)
    graph = OrderedDict()
    X_sym, y_sym = add_datasets_to_graph([X, y], ["X", "y"], graph)
    on_off = tensor.iscalar()
    dropped = dropout([X_sym], graph, 'dropout', on_off,
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
    l1 = relu([X_sym], graph, "proj", proj_dim=5,
              batch_normalize=True, mode_switch=on_off,
              random_state=random_state)
    l2 = relu([l1], graph, "proj2", proj_dim=5,
              batch_normalize=True, mode_switch=on_off,
              random_state=random_state)
    f = theano.function([X_sym, on_off], [l2], mode="FAST_COMPILE")
    params, grads = get_params_and_grads(graph, l2.mean())
    opt = sgd(params, .1)
    updates = opt.updates(params, grads)
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


def test_embedding():
    random_state = np.random.RandomState(1999)
    graph = OrderedDict()
    max_index = 100
    proj_dim = 12
    fake_str_int = [[1, 5, 7, 1, 6, 2], [2, 3, 6, 2], [3, 3, 3, 3, 3, 3, 3]]
    minibatch, mask = make_embedding_minibatch(
        fake_str_int, slice(0, 3))
    (emb_slices,), (emb_mask,) = add_embedding_datasets_to_graph(
        [minibatch], [mask], "emb", graph)
    emb = embedding(emb_slices, max_index, proj_dim, graph,
                    'emb', random_state)
    followup_dim = 17
    proj = linear([emb], graph, 'proj', followup_dim,
                  random_state=random_state)
    f = theano.function(emb_slices, [proj], mode="FAST_COMPILE")
    out, = f(*minibatch)
    assert(out.shape[-1] == 17)
    assert(out.shape[-2] == len(fake_str_int))
'''


def test_fixed_projection():
    random_state = np.random.RandomState(1999)
    rand_projection = random_state.randn(64, 12)
    rand_dim = rand_projection.shape[1]

    out = fixed_projection([X_sym], [X.shape[1]], rand_projection, 'proj1')
    out2 = fixed_projection([X_sym], [X.shape[1]], rand_projection, 'proj2',
                            pre=rand_projection[:, 0])
    out3 = fixed_projection([X_sym], [X.shape[1]], rand_projection,
                            'proj3', post=rand_projection[0])
    final = linear([out2], [rand_dim], 5, 'linear', random_state=random_state)
    # Test that it compiles with and without bias
    f = theano.function([X_sym], [out, out2, out3, final],
                        mode="FAST_COMPILE")

    # Test updates
    params = list(get_params().values())
    grads = tensor.grad(final.mean(), params)
    opt = sgd(params, .1)
    updates = opt.updates(params, grads)
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


def test_projection():
    run_common(projection)


def test_linear():
    run_common(linear)


def test_softmax():
    run_common(softmax)


def test_sigmoid():
    run_common(sigmoid)


def test_tanh():
    run_common(tanh)


def test_softplus():
    run_common(softplus)


def test_relu():
    run_common(relu)


def test_exponential():
    run_common(exponential)


def test_maxout():
    run_common(maxout)


def test_softmax_zeros():
    run_common(softmax_zeros)


def test_log_gaussian_mixture():
    lgm = log_gaussian_mixture
    random_state = np.random.RandomState(42)

    single_o = lgm([X_sym], [X.shape[1]], 5, 'single',
                   random_state=random_state)[0]
    concat_o = lgm([X_sym, y_sym], [X.shape[1], y.shape[1]], 5, 'concat',
                   random_state=random_state)[0]
    # Check that things can be reused
    repeated_o = lgm([X_sym], [X.shape[1]], 5, 'single', strict=False)[0]

    # Check that strict mode raises an error if repeated
    assert_raises(AttributeError, lgm, [X_sym], [X.shape[1]], 5, 'concat')

    f = theano.function([X_sym, y_sym], [single_o, concat_o, repeated_o],
                        mode="FAST_COMPILE")
    single, concat, repeat = f(X, y)
    assert_almost_equal(single, repeat)


def test_bernoulli_and_correlated_log_gaussian_mixture():
    bclgm = bernoulli_and_correlated_log_gaussian_mixture
    random_state = np.random.RandomState(42)

    single_o = bclgm([X_sym], [X.shape[1]], 2, 'single',
                     random_state=random_state)[0]
    concat_o = bclgm([X_sym, y_sym], [X.shape[1], y.shape[1]], 2, 'concat',
                     random_state=random_state)[0]
    # Check that things can be reused
    repeated_o = bclgm([X_sym], [X.shape[1]], 2, 'single', strict=False)[0]

    # Check that strict mode raises an error if repeated
    assert_raises(AttributeError, bclgm, [X_sym], [X.shape[1]], 2, 'concat')

    f = theano.function([X_sym, y_sym], [single_o, concat_o, repeated_o],
                        mode="FAST_COMPILE")
    single, concat, repeat = f(X, y)
    assert_almost_equal(single, repeat)


def test_conv2d():
    random_state = np.random.RandomState(42)
    # 3 channel mnist
    X_r = np.random.randn(10, 3, 28, 28).astype("float32")
    X_sym = tensor.tensor4(dtype="float32")
    l1 = conv2d([X_sym], [(3, 28, 28)], 5, name='l1', random_state=random_state)
    # test that they can stack as well
    l2 = conv2d([l1], [(5, 28, 28)], 6, name='l2', random_state=random_state)
    f = theano.function([X_sym], [l1, l2], mode="FAST_RUN")
    l1, l2 = f(X_r)


def test_pool2d():
    random_state = np.random.RandomState(42)
    # 3 channel mnist
    X_r = np.random.randn(10, 3, 28, 28).astype("float32")
    X_sym = tensor.tensor4(dtype="float32")
    del_shared()
    l1 = conv2d([X_sym], [(3, 28, 28)], 5, name='l1', random_state=random_state)
    l2 = pool2d([l1], name='l2')
    # test that they can stack as well
    l3 = pool2d([l2], name='l3')
    f = theano.function([X_sym], [l1, l2, l3], mode="FAST_RUN")
    l1, l2, l3 = f(X_r)
