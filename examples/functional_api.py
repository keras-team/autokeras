import autokeras as ak
import numpy as np
import tensorflow as tf
from keras.datasets import mnist

# Prepare the data.
import autokeras.hypermodel.block
import autokeras.hypermodel.head

(x_train, y_classification), (x_test, y_test) = mnist.load_data()
data_slice = 5
x_train = x_train[:data_slice]
y_classification = y_classification[:data_slice]
x_test = x_test[:data_slice]
y_test = y_test[:data_slice]
x_image = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

x_structured = np.random.rand(x_train.shape[0], 100)
y_regression = np.random.rand(x_train.shape[0], 1)
y_classification = y_classification.reshape(-1, 1)

# Build model and train.
inputs = ak.ImageInput(shape=(28, 28, 1))
outputs1 = ak.ResNetBlock(version='next')(inputs)
outputs2 = ak.XceptionBlock()(inputs)
image_outputs = ak.Merge()((outputs1, outputs2))

structured_inputs = ak.StructuredDataInput()
structured_outputs = ak.DenseBlock()(structured_inputs)
merged_outputs = ak.Merge()((structured_outputs, image_outputs))

classification_outputs = autokeras.hypermodel.head.ClassificationHead()(merged_outputs)
regression_outputs = autokeras.hypermodel.head.RegressionHead()(merged_outputs)
automodel = ak.GraphAutoModel(inputs=[inputs, structured_inputs],
                              outputs=[regression_outputs,
                                       classification_outputs])

automodel.fit((x_image, x_structured),
              (y_regression, y_classification),
              # trials=100,
              validation_split=0.2,
              epochs=200,
              callbacks=[tf.keras.callbacks.EarlyStopping()])
