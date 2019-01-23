from unittest.mock import patch

from autokeras.predefined_model import *
from tests.common import clean_dir, mock_train, TEST_TEMP_DIR, MockProcess


@patch('torch.multiprocessing.get_context', side_effect=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_fit_predict_save(_, _1):
    train_x = np.random.rand(100, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    for Model in [PredefinedResnet, PredefinedDensenet]:
        clf = Model(verbose=True)
        clf.fit(train_x, train_y)
        results = clf.predict(train_x)
        assert all(map(lambda result: result in train_y, results))
        score = clf.evaluate(train_x, train_y)
        assert score <= 1.0

    clean_dir(TEST_TEMP_DIR)
