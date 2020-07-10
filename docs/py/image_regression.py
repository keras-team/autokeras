"""shell
pip install autokeras
pip install git+https://github.com/keras-team/keras-tuner.git@1.0.2rc0
"""

"""
To make this tutorial easy to follow, we just treat MNIST dataset as a regression
dataset. It means we will treat prediction targets of MNIST dataset, which are
integers ranging from 0 to 9 as numerical values, so that they can be directly 
used as the regression targets.

## A Simple Example
The first step is to prepare your data. Here we use the MNIST dataset as an example
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.data_utils import Sequence
import autokeras as ak
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:100]
y_train = y_train[:100]
print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)
print(y_train[:3])  # array([7, 2, 1], dtype=uint8)

"""
The second step is to run the ImageRegressor.
It is recommended have more trials for more complicated datasets.
This is just a quick demo of MNIST, so we set max_trials to 1.
"""

# Initialize the image regressor.
reg = ak.ImageRegressor(
    overwrite=True,
    max_trials=1)
# Feed the image regressor with training data.
reg.fit(x_train, y_train, epochs=2)


# Predict with the best model.
predicted_y = reg.predict(x_test)
print(predicted_y)


# Evaluate the best model with testing data.
print(reg.evaluate(x_test, y_test))

"""
## Validation Data
By default, AutoKeras use the last 20% of training data as validation data. As shown in
the example below, you can use validation_split to specify the percentage.
"""

reg.fit(
    x_train,
    y_train,
    # Split the training data and use the last 15% as validation data.
    validation_split=0.15,
    epochs=2,
)

"""
You can also use your own validation set instead of splitting it from the training data
with validation_data.
"""

split = 50000
x_val = x_train[split:]
y_val = y_train[split:]
x_train = x_train[:split]
y_train = y_train[:split]
reg.fit(
    x_train,
    y_train,
    # Use your own validation set.
    validation_data=(x_val, y_val),
    epochs=2,
)

"""
## Customized Search Space
For advanced users, you may customize your search space by using AutoModel instead of
ImageRegressor. You can configure the ImageBlock for some high-level configurations,
e.g., block_type for the type of neural network to search, normalize for whether to do
data normalization, augment for whether to do data augmentation. You can also do not
specify these arguments, which would leave the different choices to be tuned
automatically. See the following example for detail.
"""

input_node = ak.ImageInput()
output_node = ak.ImageBlock(
    # Only search ResNet architectures.
    block_type="resnet",
    # Normalize the dataset.
    normalize=False,
    # Do not do data augmentation.
    augment=False,
)(input_node)
output_node = ak.RegressionHead()(output_node)
reg = ak.AutoModel(
    inputs=input_node,
    outputs=output_node,
    overwrite=True,
    max_trials=1)
reg.fit(x_train, y_train, epochs=2)

"""
The usage of AutoModel is similar to the functional API of Keras. Basically, you are
building a graph, whose edges are blocks and the nodes are intermediate outputs of
blocks. To add an edge from input_node to output_node with output_node =
ak.[some_block]([block_args])(input_node).

You can even also use more fine grained blocks to customize the search space even
further. See the following example.
"""

input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node = ak.ImageAugmentation(horizontal_flip=False)(output_node)
output_node = ak.ResNetBlock()(output_node)
output_node = ak.RegressionHead()(output_node)
reg = ak.AutoModel(
    inputs=input_node,
    outputs=output_node,
    overwrite=True,
    max_trials=1)
reg.fit(x_train, y_train, epochs=2)

"""
## Data Format
The AutoKeras ImageRegressor is quite flexible for the data format.

For the image, it accepts data formats both with and without the channel dimension. The
images in the MNIST dataset do not have the channel dimension. Each image is a matrix
with shape (28, 28). AutoKeras also accepts images of three dimensions with the channel
dimension at last, e.g., (32, 32, 3), (28, 28, 1).

For the regression targets, it should be a vector of numerical values.
AutoKeras accepts numpy.ndarray.

We also support using tf.data.Dataset format for the training data. In this case, the
images would have to be 3-dimentional.
"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the images to have the channel dimension.
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))
y_train = y_train.reshape(y_train.shape + (1,))
y_test = y_test.reshape(y_test.shape + (1,))

print(x_train.shape)  # (60000, 28, 28, 1)
print(y_train.shape)  # (60000, 10)

train_set = tf.data.Dataset.from_tensor_slices(((x_train,), (y_train,)))
test_set = tf.data.Dataset.from_tensor_slices(((x_test,), (y_test,)))

reg = ak.ImageRegressor(
    overwrite=True,
    max_trials=1)
# Feed the tensorflow Dataset to the regressor.
reg.fit(train_set, epochs=2)
# Predict with the best model.
predicted_y = reg.predict(test_set)
# Evaluate the best model with testing data.
print(reg.evaluate(test_set))

"""
## Reference
[ImageRegressor](/image_regressor),
[AutoModel](/auto_model/#automodel-class),
[ImageBlock](/block/#imageblock-class),
[Normalization](/block/#normalization-class),
[ImageAugmentation](/block/#image-augmentation-class),
[ResNetBlock](/block/#resnetblock-class),
[ImageInput](/node/#imageinput-class),
[RegressionHead](/block/#regressionhead-class).
"""
