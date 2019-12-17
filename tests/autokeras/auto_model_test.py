from unittest import mock

import numpy as np
import pytest

import autokeras as ak


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_auto_model')


@mock.patch('autokeras.auto_model.tuner_module.get_tuner_class')
def test_evaluate(tuner_fn, tmp_dir):
    pg = mock.Mock()
    pg.preprocess.return_value = (mock.Mock(), mock.Mock())
    tuner_class = tuner_fn.return_value
    tuner = tuner_class.return_value
    tuner.get_best_model.return_value = (pg, mock.Mock())

    x_train = np.random.rand(100, 32)
    y_train = np.random.rand(100, 1)

    input_node = ak.Input()
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.RegressionHead()(output_node)

    auto_model = ak.GraphAutoModel(input_node,
                                   output_node,
                                   directory=tmp_dir,
                                   max_trials=1)
    auto_model.fit(x_train, y_train, epochs=1, validation_data=(x_train, y_train))
    auto_model.evaluate(x_train, y_train)
    assert tuner_fn.called
    assert tuner_class.called
    assert tuner.get_best_model.called


@mock.patch('autokeras.auto_model.tuner_module.get_tuner_class')
def test_auto_model_predict(tuner_fn, tmp_dir):
    pg = mock.Mock()
    pg.preprocess.return_value = (mock.Mock(), mock.Mock())
    tuner_class = tuner_fn.return_value
    tuner = tuner_class.return_value
    tuner.get_best_model.return_value = (pg, mock.Mock())

    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = ak.AutoModel(ak.ImageInput(),
                              ak.RegressionHead(),
                              directory=tmp_dir,
                              max_trials=2)
    auto_model.fit(x_train, y_train, epochs=2, validation_split=0.2)
    auto_model.predict(x_train)
    assert tuner_fn.called
    assert tuner_class.called
    assert tuner.get_best_model.called


@mock.patch('autokeras.auto_model.tuner_module.get_tuner_class')
def test_final_fit_concat(tuner_fn, tmp_dir):
    tuner_class = tuner_fn.return_value

    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = ak.AutoModel(ak.ImageInput(),
                              ak.RegressionHead(),
                              directory=tmp_dir,
                              max_trials=2)
    auto_model.fit(x_train, y_train, epochs=2, validation_split=0.2)
    assert auto_model._split_dataset
    assert tuner_class.call_args_list[0][1]['fit_on_val_data']


@mock.patch('autokeras.auto_model.tuner_module.get_tuner_class')
def test_final_fit_not_concat(tuner_fn, tmp_dir):
    tuner_class = tuner_fn.return_value

    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = ak.AutoModel(ak.ImageInput(),
                              ak.RegressionHead(),
                              directory=tmp_dir,
                              max_trials=2)
    auto_model.fit(x_train, y_train, epochs=2, validation_data=(x_train, y_train))
    assert not auto_model._split_dataset
    assert not tuner_class.call_args_list[0][1]['fit_on_val_data']


@mock.patch('autokeras.auto_model.tuner_module.get_tuner_class')
def test_overwrite(tuner_fn, tmp_dir):
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
