from dagbldr.utils import make_character_level_from_text
from nose.tools import assert_raises


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
