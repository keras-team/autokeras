from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

import autokeras as ak
from tests import utils


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_evaluate(tuner_fn, tmp_path):
    x_train = np.random.rand(100, 32)
    y_train = np.random.rand(100, 1)

    input_node = ak.Input()
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.RegressionHead()(output_node)

    auto_model = ak.AutoModel(input_node,
                              output_node,
                              directory=tmp_path,
                              max_trials=1)
    auto_model.fit(x_train, y_train, epochs=1, validation_data=(x_train, y_train))
    auto_model.evaluate(x_train, y_train)
    assert tuner_fn.called


def get_single_io_auto_model(tmp_path):
    return ak.AutoModel(ak.ImageInput(),
                        ak.RegressionHead(),
                        directory=tmp_path,
                        max_trials=2)


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_auto_model_predict(tuner_fn, tmp_path):
    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = get_single_io_auto_model(tmp_path)
    auto_model.fit(x_train, y_train, epochs=2, validation_split=0.2)
    auto_model.predict(x_train)
    assert tuner_fn.called


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_final_fit_concat(tuner_fn, tmp_path):
    tuner = tuner_fn.return_value.return_value

    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = get_single_io_auto_model(tmp_path)
    auto_model.fit(x_train, y_train, epochs=2, validation_split=0.2)
    assert auto_model._split_dataset
    assert tuner.search.call_args_list[0][1]['fit_on_val_data']


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_final_fit_not_concat(tuner_fn, tmp_path):
    tuner = tuner_fn.return_value.return_value

    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = get_single_io_auto_model(tmp_path)
    auto_model.fit(x_train, y_train, epochs=2, validation_data=(x_train, y_train))
    assert not auto_model._split_dataset
    assert not tuner.search.call_args_list[0][1]['fit_on_val_data']


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_overwrite(tuner_fn, tmp_path):
    tuner_class = tuner_fn.return_value

    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = get_single_io_auto_model(tmp_path)
    auto_model.fit(x_train, y_train, epochs=2, validation_data=(x_train, y_train))
    assert not tuner_class.call_args_list[0][1]['overwrite']


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_export_model(tuner_fn, tmp_path):
    tuner_class = tuner_fn.return_value
    tuner = tuner_class.return_value

    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)

    auto_model = get_single_io_auto_model(tmp_path)
    auto_model.fit(x_train, y_train, epochs=2, validation_data=(x_train, y_train))
    auto_model.export_model()
    assert tuner.get_best_model.called


def get_multi_io_auto_model(tmp_path):
    return ak.AutoModel([ak.ImageInput(), ak.ImageInput()],
                        [ak.RegressionHead(), ak.RegressionHead()],
                        directory=tmp_path,
                        max_trials=2,
                        overwrite=False)


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_multi_io_with_tf_dataset(tuner_fn, tmp_path):
    auto_model = get_multi_io_auto_model(tmp_path)
    x1 = utils.generate_data()
    y1 = utils.generate_data(shape=(1,))
    dataset = tf.data.Dataset.from_tensor_slices(((x1, x1), (y1, y1)))
    auto_model.fit(dataset, epochs=2)

    for adapter in auto_model._input_adapters + auto_model._output_adapters:
        assert adapter.shape is not None


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_single_nested_dataset(tuner_fn, tmp_path):
    auto_model = ak.AutoModel(ak.ImageInput(),
                              ak.RegressionHead(),
                              directory=tmp_path,
                              max_trials=2,
                              overwrite=False)
    x1 = utils.generate_data()
    y1 = utils.generate_data(shape=(1,))
    dataset = tf.data.Dataset.from_tensor_slices(((x1,), y1))
    auto_model.fit(dataset, epochs=2)

    for adapter in auto_model._input_adapters + auto_model._output_adapters:
        assert adapter.shape is not None


def dataset_error(x, y, validation_data, message, tmp_path):
    auto_model = get_multi_io_auto_model(tmp_path)
    with pytest.raises(ValueError) as info:
        auto_model.fit(x, y, epochs=2, validation_data=validation_data)
    assert message in str(info.value)


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_data_io_consistency_input(tuner_fn, tmp_path):
    x1 = utils.generate_data()
    y1 = utils.generate_data(shape=(1,))
    dataset = tf.data.Dataset.from_tensor_slices(((x1,), (y1, y1)))
    dataset_error(dataset, None, dataset, 'Expect x to have', tmp_path)


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_data_io_consistency_output(tuner_fn, tmp_path):
    x1 = utils.generate_data()
    y1 = utils.generate_data(shape=(1,))
    dataset = tf.data.Dataset.from_tensor_slices(((x1, x1), (y1,)))
    dataset_error(dataset, None, dataset, 'Expect y to have', tmp_path)


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_data_io_consistency_validation(tuner_fn, tmp_path):
    x1 = utils.generate_data()
    y1 = utils.generate_data(shape=(1,))
    dataset = tf.data.Dataset.from_tensor_slices(((x1, x1), (y1, y1)))
    val_dataset = tf.data.Dataset.from_tensor_slices(((x1,), (y1, y1)))
    dataset_error(dataset, None, val_dataset,
                  'Expect x in validation_data to have', tmp_path)


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_dataset_and_y(tuner_fn, tmp_path):
    x1 = utils.generate_data()
    y1 = utils.generate_data(shape=(1,))
    x = tf.data.Dataset.from_tensor_slices((x1, x1))
    y = tf.data.Dataset.from_tensor_slices((y1, y1))
    val_dataset = tf.data.Dataset.from_tensor_slices(((x1,), (y1, y1)))
    dataset_error(x, y, val_dataset,
                  'Expect y is None', tmp_path)


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_multi_input_predict(tuner_fn, tmp_path):
    auto_model = get_multi_io_auto_model(tmp_path)
    x1 = utils.generate_data()
    y1 = utils.generate_data(shape=(1,))
    dataset = tf.data.Dataset.from_tensor_slices(((x1, x1), (y1, y1)))
    auto_model.fit(dataset, None, epochs=2, validation_data=dataset)

    dataset2 = tf.data.Dataset.from_tensor_slices(((x1, x1),))
    auto_model.predict(dataset2)


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_multi_input_predict2(tuner_fn, tmp_path):
    auto_model = get_multi_io_auto_model(tmp_path)
    x1 = utils.generate_data()
    y1 = utils.generate_data(shape=(1,))
    dataset = tf.data.Dataset.from_tensor_slices(((x1, x1), (y1, y1)))
    auto_model.fit(dataset, None, epochs=2, validation_data=dataset)

    dataset2 = tf.data.Dataset.from_tensor_slices((x1, x1))
    auto_model.predict(dataset2)


@mock.patch('autokeras.auto_model.get_tuner_class')
def test_single_input_predict(tuner_fn, tmp_path):
    auto_model = get_single_io_auto_model(tmp_path)
    x1 = utils.generate_data()
    y1 = utils.generate_data(shape=(1,))
    dataset = tf.data.Dataset.from_tensor_slices((x1, y1))
    auto_model.fit(dataset, None, epochs=2, validation_data=dataset)

    dataset2 = tf.data.Dataset.from_tensor_slices((x1, y1))
    auto_model.predict(dataset2)
