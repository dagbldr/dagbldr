from dagbldr.nodes.utils import generate_autoregressive_masks
from numpy.testing import assert_allclose, assert_raises
import numpy as np


def test_generate_autoregressive_masks():
    def _test_mask_ar_property(masks):
        res = masks[0]
        for m in masks[1:]:
            res = np.dot(res, m)
        joint_mask = res
        assert np.sum(np.trace(joint_mask)) == 0.

    random_state = np.random.RandomState(1999)
    sizes = [3, 4, 4, 3]
    masks, orderings = generate_autoregressive_masks(
        sizes, forced_input_ordering=np.array([3, 1, 2]),
        forced_samplings=[np.array([2, 1, 2, 2]), np.array([1, 2, 2, 1])],
        random_state=random_state)
    # compare to masks from paper
    paper_orderings = [np.array([3, 1, 2]),
                       np.array([2, 1, 2, 2]),
                       np.array([1, 2, 2, 1]),
                       np.array([3, 1, 2])]
    assert len(paper_orderings) == len(orderings)
    for o, p in list(zip(orderings, paper_orderings)):
        assert_allclose(o, p)
    paper_l0_mask = np.array([[0, 0, 0, 0],
                              [1, 1, 1, 1],
                              [1, 0, 1, 1]])
    assert_allclose(masks[0], paper_l0_mask)
    paper_l1_mask = np.array([[0, 1, 1, 0],
                              [1, 1, 1, 1],
                              [0, 1, 1, 0],
                              [0, 1, 1, 0]])
    assert_allclose(masks[1], paper_l1_mask)
    paper_l2_mask = np.array([[1, 0, 1],
                              [1, 0, 0],
                              [1, 0, 0],
                              [1, 0, 1]])
    assert_allclose(masks[2], paper_l2_mask)
    _test_mask_ar_property(masks)

    random_state = np.random.RandomState(1999)
    sizes = [7, 5, 5, 7]
    first_masks, _ = generate_autoregressive_masks(
        sizes, random_state=random_state)
    second_masks, _ = generate_autoregressive_masks(
        sizes, random_state=np.random.RandomState(1111))
    for n in range(len(first_masks)):
        assert_raises(AssertionError, assert_allclose,
                      first_masks[n], second_masks[n])
    random_state = np.random.RandomState(1999)
    third_masks, ordering = generate_autoregressive_masks(
        sizes, random_state=random_state)
    for n in range(len(first_masks)):
        assert_allclose(first_masks[n], third_masks[n])
    _test_mask_ar_property(third_masks)
