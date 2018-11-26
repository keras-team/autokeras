from unittest.mock import patch

from autokeras.text.text_preprocessor import *
from tests.common import TEST_TEMP_DIR

word_index = {"foo": 0, "bar": 1}
embedding_index = {"foo": np.random.uniform(low=0.0, high=1.0, size=(100,)),
                   "bar": np.random.uniform(low=0.0, high=1.0, size=(100,))}

embedding_matrix = np.random.rand(3, 100)


def mock_clean_str(x_train):
    return x_train


def mock_tokenlize_text(max_num_words, max_seq_length, x_train):
    return x_train, word_index


def mock_processing(path, word_index, input_length, x_train):
    return x_train


def mock_download_pre_train(file_path, extract_path):
    pass


def mock_read_embedding_index(extract_path):
    return embedding_index


def mock_load_pretrain(path, word_index):
    return embedding_matrix


@patch('autokeras.text.text_preprocessor.processing', side_effect=mock_processing)
@patch('autokeras.text.text_preprocessor.tokenlize_text', side_effect=mock_tokenlize_text)
@patch('autokeras.text.text_preprocessor.clean_str', side_effect=mock_clean_str)
def test_text_preprocess_class(_, _1, _2):
    train_x = np.random.rand(100, 25, 25)
    train_x = text_preprocess(train_x)


def test_clean_str():
    test_string = "Dummy   1@\'s\'ven\'t\'re\'d\'ll,!()?"
    assert clean_str(test_string) == "dummy 1 's 've n't 're 'd 'll , ! \( \) \?"


@patch('autokeras.text.text_preprocessor.read_embedding_index', side_effect=mock_read_embedding_index)
@patch('autokeras.text.text_preprocessor.download_pre_train', side_effect=mock_download_pre_train)
def test_load_pretrain(_, _1):
    embedding_matrix = load_pretrain(TEST_TEMP_DIR, word_index)
    assert (embedding_matrix[0] == embedding_index.get("foo")).all()
    assert (embedding_matrix[1] == embedding_index.get("bar")).all()


@patch('autokeras.text.text_preprocessor.GPUtil.getFirstAvailable', return_value=[0])
@patch('autokeras.text.text_preprocessor.load_pretrain', side_effect=mock_load_pretrain)
def test_processing(_, _1):
    train_x = np.full((1, 2), 1)
    train_x = processing(TEST_TEMP_DIR, word_index, 2, train_x)
    assert np.allclose(train_x[0][0], embedding_matrix[1])


def test_tokenlize_text():
    dummy_train = ['foo bar']
    train_x, dummy_index = tokenlize_text(max_num_words=2, max_seq_length=3, x_train=dummy_train)
    assert (train_x == [[0, 0, 1]]).all()
    assert (dummy_index.get("foo") == 1)
    assert (dummy_index.get("bar") == 2)
