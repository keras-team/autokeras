import os
from unittest.mock import patch

import pytest

from autokeras.predefined_model import *
from tests.common import clean_dir, mock_train, TEST_TEMP_DIR
from autokeras.preprocessor import OneHotEncoder
from autokeras.preprocessor import ImageDataTransformer


@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_fit_predict_save(_):
    train_x = np.random.rand(100, 25, 25, 1)
    train_y = np.random.randint(0, 5, 100)
    y_encoder = OneHotEncoder()
    y_encoder.fit(train_y)
    train_y_encoded = y_encoder.transform(train_y)
    clf = PredefinedResnet(n_output_node=5, input_shape=train_x.shape[1:],
                           inverse_transform_y_method=y_encoder.inverse_transform,
                           verbose=True,
                           path=rand_temp_folder_generator())
    data_transformer = ImageDataTransformer(train_x)
    train_loader = data_transformer.transform_train(train_x, train_y_encoded)
    valid_loader = data_transformer.transform_test(train_x, train_y_encoded)
    clf.fit(train_loader, valid_loader)
    test_loader = data_transformer.transform_test(train_x)
    results = clf.predict(test_loader)
    assert all(map(lambda result: result in train_y, results))
    model_path = os.path.join(TEST_TEMP_DIR, 'torchmodel')
    clf.save(model_path)
    assert os.path.isfile(model_path)

    clean_dir(TEST_TEMP_DIR)