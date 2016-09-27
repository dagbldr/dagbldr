import numpy as np
import theano
from theano import tensor
from nose.tools import assert_raises
from numpy.testing import assert_almost_equal

from dagbldr.datasets import load_digits
from dagbldr import del_shared
from dagbldr.utils import convert_to_one_hot
from dagbldr.nodes import linear
from dagbldr.nodes import softplus
from dagbldr.nodes import gaussian_sample
from dagbldr.nodes import gaussian_log_sample

# Common between tests
digits = load_digits()
X = digits["data"].astype("float32")
y = digits["target"]
n_classes = len(set(y))
y = convert_to_one_hot(y, n_classes).astype("float32")
X_sym = tensor.fmatrix()
y_sym = tensor.fmatrix()


def test_gaussian_sample():
    del_shared()
    random_state = np.random.RandomState(1999)
    mu = linear([X_sym], [X.shape[1]], proj_dim=100, name='mu',
                random_state=random_state)
    sigma = softplus([X_sym], [X.shape[1]], proj_dim=100, name='sigma',
                     random_state=random_state)
    random_state = np.random.RandomState(1999)
    r1 = gaussian_sample([mu], [sigma], name="samp1", random_state=random_state)
    random_state = np.random.RandomState(1999)
    r2 = gaussian_sample([mu], [sigma], name="samp2", random_state=random_state)
    random_state = np.random.RandomState(42)
    r3 = gaussian_sample([mu], [sigma], name="samp3", random_state=random_state)
    sample_function = theano.function([X_sym], [r1, r2, r3],
                                      mode="FAST_COMPILE")
    s_r1, s_r2, s_r3 = sample_function(X[:100])
    assert_almost_equal(s_r1, s_r2)
    assert_raises(AssertionError, assert_almost_equal, s_r1, s_r3)


def test_gaussian_log_sample():
    del_shared()
    random_state = np.random.RandomState(1999)
    mu = linear([X_sym], [X.shape[1]], proj_dim=100, name='mu',
                random_state=random_state)
    sigma = linear([X_sym], [X.shape[1]], proj_dim=100, name='sigma',
                   random_state=random_state)
    random_state = np.random.RandomState(1999)
    r1 = gaussian_log_sample([mu], [sigma], name="samp1",
                             random_state=random_state)
    random_state = np.random.RandomState(1999)
    r2 = gaussian_log_sample([mu], [sigma], name="samp2",
                             random_state=random_state)
    random_state = np.random.RandomState(42)
    r3 = gaussian_log_sample([mu], [sigma], name="samp3",
                             random_state=random_state)
    sample_function = theano.function([X_sym], [r1, r2, r3],
                                      mode="FAST_COMPILE")
    s_r1, s_r2, s_r3 = sample_function(X[:100])
    assert_almost_equal(s_r1, s_r2)
    assert_raises(AssertionError, assert_almost_equal, s_r1, s_r3)
