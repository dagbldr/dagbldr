from dagbldr.datasets.dataset_utils import character_sequence_iterator
from dagbldr.datasets.dataset_utils import word_sequence_iterator

sample_sentences = ["The end of the world was nigh",
                    "My hands feel just like two balloons",
                    "Hello my ragtime gal"]

def test_character_sequence_iterator():
    itr = character_sequence_iterator(sample_sentences, 1, 10)
    r = itr.next()
    ir = itr.inverse_transform(r)
    assert "".join(ir[0]) == "The end of"
    neg = itr.inverse_transform(itr.transform(["harsh times"]))
    assert neg[0] == list("harsh times")


def test_word_sequence_iterator():
    itr = word_sequence_iterator(sample_sentences, 1, 10)
    r = itr.next()
    ir = itr.inverse_transform(r)
    assert " ".join(ir[0]) == "the end of the world was nigh <EOS> my hands"
    # Test unk
    neg = itr.inverse_transform(itr.transform(["harsh times"]))
    assert neg[0] == ["<UNK>", "<UNK>", "<EOS>"]
