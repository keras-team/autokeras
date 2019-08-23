import autokeras as ak
import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_lgbm')


def test_lgbm(tmp_dir):
    x_train = np.random.rand(2, 32)
    y_train = np.array([[1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0]])

    input_node = ak.Input()
    output_node = input_node
    output_node = ak.LgbmModule()(output_node)
    output_node = ak.IdentityBlock()(output_node)
    output_node = ak.ClassificationHead()(output_node)

    auto_model = ak.GraphAutoModel(input_node,
                                   output_node,
                                   directory=tmp_dir,
                                   max_trials=1)
    auto_model.fit(x_train, y_train, epochs=1,
                   validation_data=(x_train, y_train))
    result = auto_model.predict(x_train)

    assert result.shape == (100, 10)
