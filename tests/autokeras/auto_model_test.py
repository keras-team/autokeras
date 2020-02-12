from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import autokeras as ak


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_auto_model')


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_evaluate(tuner_fn, tmp_dir):
    tmp_dir = Path(tmp_dir)
    x_train = np.random.rand(100, 32)
    y_train = np.random.rand(100, 1)

    input_node = ak.Input()
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.RegressionHead()(output_node)

    auto_model = ak.AutoModel(input_node,
                              output_node,
                              directory=tmp_dir,
                              max_trials=1)
    auto_model.fit(x_train, y_train, epochs=1, validation_data=(x_train, y_train))
    auto_model.evaluate(x_train, y_train)
    assert tuner_fn.called


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_auto_model_predict(tuner_fn, tmp_dir):
    tmp_dir = Path(tmp_dir)
    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = ak.AutoModel(ak.ImageInput(),
                              ak.RegressionHead(),
                              directory=tmp_dir,
                              max_trials=2)
    auto_model.fit(x_train, y_train, epochs=2, validation_split=0.2)
    auto_model.predict(x_train)
    assert tuner_fn.called


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_final_fit_concat(tuner_fn, tmp_dir):
    tmp_dir = Path(tmp_dir)
    tuner = tuner_fn.return_value.return_value

    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = ak.AutoModel(ak.ImageInput(),
                              ak.RegressionHead(),
                              directory=tmp_dir,
                              max_trials=2)
    auto_model.fit(x_train, y_train, epochs=2, validation_split=0.2)
    assert auto_model._split_dataset
    assert tuner.search.call_args_list[0][1]['fit_on_val_data']


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_final_fit_not_concat(tuner_fn, tmp_dir):
    tmp_dir = Path(tmp_dir)
    tuner = tuner_fn.return_value.return_value

    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = ak.AutoModel(ak.ImageInput(),
                              ak.RegressionHead(),
                              directory=tmp_dir,
                              max_trials=2)
    auto_model.fit(x_train, y_train, epochs=2, validation_data=(x_train, y_train))
    assert not auto_model._split_dataset
    assert not tuner.search.call_args_list[0][1]['fit_on_val_data']


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_overwrite(tuner_fn, tmp_dir):
    tmp_dir = Path(tmp_dir)
    tuner_class = tuner_fn.return_value

    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = ak.AutoModel(ak.ImageInput(),
                              ak.RegressionHead(),
                              directory=tmp_dir,
                              max_trials=2,
                              overwrite=False)
    auto_model.fit(x_train, y_train, epochs=2, validation_data=(x_train, y_train))
    assert not tuner_class.call_args_list[0][1]['overwrite']


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_export_model(tuner_fn, tmp_dir):
    tmp_dir = Path(tmp_dir)
    tuner_class = tuner_fn.return_value
    tuner = tuner_class.return_value

    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = ak.AutoModel(ak.ImageInput(),
                              ak.RegressionHead(),
                              directory=tmp_dir,
                              max_trials=2,
                              overwrite=False)
    auto_model.fit(x_train, y_train, epochs=2, validation_data=(x_train, y_train))
    auto_model.export_model()
    assert tuner.get_best_model.called
