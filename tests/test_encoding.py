import pytest

from src.encoding import Vocabulary, NotFittedError


@pytest.fixture
def data():
    return ["A", "B", "C", "D"]


@pytest.fixture
def vocab(data):
    vocab = Vocabulary(handle_unknown=True)
    vocab.fit(data)
    return vocab


def test_fit_transform_returns_int(vocab):
    transformed_value = vocab.transform("A")
    assert type(transformed_value) is int


def test_inverse_transform_returns_original_value_type(vocab):
    original_value = vocab.inverse_transform(2)
    assert type(original_value) is str


def test_transform_inverse_transform_returns_original_value(vocab, data):
    tranformed_data = [vocab.transform(v) for v in data]
    original_data = [vocab.inverse_transform(v) for v in tranformed_data]
    assert original_data == data


def test_unseen_data_returns_default_value(vocab):
    transformed_unk = vocab.transform("X")
    assert transformed_unk == vocab.DEFAULT_UNK_VALUE


def test_unseen_data_raises_error_when_unknown_false(data):
    vocab = Vocabulary(handle_unknown=False)
    vocab.fit(data)
    with pytest.raises(KeyError):
        vocab.transform("X")


def test_transform_not_fit_raises_error():
    vocab = Vocabulary()

    with pytest.raises(NotFittedError):
        vocab.transform("A")


def test_len_fitted_unknown_true_returns_length_data_plus_one(data):
    vocab = Vocabulary(handle_unknown=True).fit(data)
    assert len(vocab) == len(data) + 1


def test_len_fitted_unknown_false_returns_length_data(data):
    vocab = Vocabulary(handle_unknown=False).fit(data)
    assert len(vocab) == len(data)
