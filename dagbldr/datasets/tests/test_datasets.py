from dagbldr.datasets import load_digits
from dagbldr.datasets import load_iris
from nose.tools import assert_equal


def test_digits():
    digits = load_digits()
    assert_equal(len(digits["data"]), len(digits["target"]))


def test_iris():
    iris = load_iris()
    assert_equal(len(iris["data"]), len(iris["target"]))
