from nose.tools import assert_raises

from dagbldr.utils import make_character_level_from_text, convert_to_one_hot
from dagbldr.utils import make_embedding_minibatch
from dagbldr.datasets import load_digits

digits = load_digits()
X = digits["data"]
y = digits["target"]


def test_make_embedding_minibatch():
    fake_str_int = [[1, 5, 7, 1, 6, 2], [2, 3, 6, 2], [3, 3, 3, 3, 3, 3, 3]]
    uniform_minibatch, mask = make_embedding_minibatch(
        fake_str_int, slice(0, 2))


def test_convert_to_one_hot():
    fake_str_int = [[1, 5, 7, 1, 6, 0], [2, 3, 6, 0]]

    n_classes = len(set(y))
    convert_to_one_hot(y, n_classes)
    convert_to_one_hot(fake_str_int, 8)
    assert_raises(ValueError, convert_to_one_hot, X[0], n_classes)
    assert_raises(ValueError, convert_to_one_hot, X, n_classes)


def test_character_level_from_text():
    test_strs = ["All work and no play makes jack a dull boy!@#**-~`",
                 ""]

    # Make sure that it raises if it gets the wrong input
    assert_raises(ValueError, make_character_level_from_text, test_strs[0])
    assert_raises(ValueError, make_character_level_from_text, test_strs[0][0])

    clean, mf, imf, m = make_character_level_from_text(test_strs)
    if len(clean) != len(test_strs) - 1:
        raise AssertionError("Failed to remove empty line")

    new_str = "zzzzzzzzzzzzzz"
    new_clean = mf(new_str)
    # Make sure all the unknown chars get UNK tags
    if sum(new_clean[:-1]) != len(new_clean[:-1]) * m["UNK"]:
        raise AssertionError("Failed to handle unknown char")
    # Make sure last tag is EOS
    if new_clean[-1] != m["EOS"]:
        raise AssertionError("Failed to add EOS tag")

if __name__ == "__main__":
    test_make_embedding_minibatch()
