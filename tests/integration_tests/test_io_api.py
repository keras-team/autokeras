import numpy as np
import autokeras as ak
import tensorflow as tf
import pytest
from tensorflow.python.keras.datasets import mnist


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('test_io_api')


def test_io_api(tmp_dir):
    (x_train, y_classification), (x_test, y_test) = mnist.load_data()
    data_slice = 200
    x_train = x_train[:data_slice]
    print(x_train.dtype)
    y_classification = y_classification[:data_slice]
    x_test = x_test[:data_slice]
    y_test = y_test[:data_slice]
    x_train = x_train.astype(np.float64)
    x_test = x_test.astype(np.float64)
    print(x_train.dtype)
    # x_image = np.reshape(x_train, (200, 28, 28, 1))
    # x_test = np.reshape(x_test, (200, 28, 28, 1))
    x_image = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    '''x_structured = np.random.rand(x_train.shape[0], 100)
    y_regression = np.random.rand(x_train.shape[0], 1)'''
    x_structured = np.random.rand(x_train.shape[0], 100)
    y_regression = np.random.rand(x_train.shape[0], 1)
    y_classification = y_classification.reshape(-1, 1)
    # y_classification = np.reshape(y_classification, (-1, 1))
    # Build model and train.
    automodel = ak.AutoModel(
        inputs=[ak.ImageInput(),
                ak.StructuredDataInput()],
        outputs=[ak.RegressionHead(metrics=['mae']),
                 ak.ClassificationHead(loss='categorical_crossentropy',
                                       metrics=['accuracy'])])
    automodel.fit([x_image, x_structured],
                  [y_regression, y_classification],
                  validation_split=0.2)
