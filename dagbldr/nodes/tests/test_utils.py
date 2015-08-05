from dagbldr.nodes.utils import generate_autoregressive_mask
from numpy.testing import assert_allclose, assert_raises
import numpy as np


def test_generate_autoregressive_mask():
    random_state = np.random.RandomState(1999)
    s1 = 784
    s2 = 256
    first_mask, _ = generate_autoregressive_mask(s1, s2, random_state)
    second_mask, _ = generate_autoregressive_mask(s1, s2)
    assert_raises(AssertionError, assert_allclose, first_mask, second_mask)
    random_state = np.random.RandomState(1999)
    third_mask, ordering = generate_autoregressive_mask(s1, s2, random_state)
    assert_allclose(first_mask, third_mask)
    assert max(ordering) == s1
    assert min(ordering) == 1
