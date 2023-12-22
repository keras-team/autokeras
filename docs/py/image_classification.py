"""shell
pip install autokeras
"""
import numpy as np
import tensorflow as tf
from keras.datasets import mnist

import autokeras as ak

"""
## A Simple Example
The first step is to prepare your data. Here we use the MNIST dataset as an
example
"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)
print(y_train[:3])  # array([7, 2, 1], dtype=uint8)

"""
The second step is to run the ImageClassifier.
It is recommended have more trials for more complicated datasets.
This is just a quick demo of MNIST, so we set max_trials to 1.
For the same reason, we set epochs to 10.
You can also leave the epochs unspecified for an adaptive number of epochs.
"""

# Initialize the image classifier.
clf = ak.ImageClassifier(overwrite=True, max_trials=1)
# Feed the image classifier with training data.
clf.fit(x_train, y_train, epochs=10)


# Predict with the best model.
predicted_y = clf.predict(x_test)
print(predicted_y)


# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))

"""
## Validation Data
By default, AutoKeras use the last 20% of training data as validation data. As
shown in the example below, you can use validation_split to specify the
percentage.
"""

clf.fit(
    x_train,
    y_train,
    # Split the training data and use the last 15% as validation data.
    validation_split=0.15,
    epochs=10,
)

"""
You can also use your own validation set instead of splitting it from the
training data with validation_data.
"""

split = 50000
x_val = x_train[split:]
y_val = y_train[split:]
x_train = x_train[:split]
y_train = y_train[:split]
clf.fit(
    x_train,
    y_train,
    # Use your own validation set.
    validation_data=(x_val, y_val),
    epochs=10,
)

"""
## Customized Search Space
For advanced users, you may customize your search space by using AutoModel
instead of ImageClassifier. You can configure the ImageBlock for some
high-level configurations, e.g., block_type for the type of neural network to
search, normalize for whether to do data normalization, augment for whether to
do data augmentation. You can also do not specify these arguments, which would
leave the different choices to be tuned automatically. See the following
example for detail.
"""

input_node = ak.ImageInput()
output_node = ak.ImageBlock(
    # Only search ResNet architectures.
    block_type="resnet",
    # Normalize the dataset.
    normalize=True,
    # Do not do data augmentation.
    augment=False,
)(input_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(
    inputs=input_node, outputs=output_node, overwrite=True, max_trials=1
)
clf.fit(x_train, y_train, epochs=10)

"""
The usage of AutoModel is similar to the functional API of Keras. Basically, you
are building a graph, whose edges are blocks and the nodes are intermediate
outputs of blocks. To add an edge from input_node to output_node with
output_node = ak.[some_block]([block_args])(input_node).

You can even also use more fine grained blocks to customize the search space
even further. See the following example.
"""

input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node = ak.ImageAugmentation(horizontal_flip=False)(output_node)
output_node = ak.ResNetBlock(version="v2")(output_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(
    inputs=input_node, outputs=output_node, overwrite=True, max_trials=1
)
clf.fit(x_train, y_train, epochs=10)

"""
## Data Format
The AutoKeras ImageClassifier is quite flexible for the data format.

For the image, it accepts data formats both with and without the channel
dimension. The images in the MNIST dataset do not have the channel dimension.
Each image is a matrix with shape (28, 28). AutoKeras also accepts images of
three dimensions with the channel dimension at last, e.g., (32, 32, 3), (28,
28, 1).

For the classification labels, AutoKeras accepts both plain labels, i.e.
strings or integers, and one-hot encoded encoded labels, i.e. vectors of 0s and
1s.

So if you prepare your data in the following way, the ImageClassifier should
still work.
"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the images to have the channel dimension.
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

# One-hot encode the labels.
eye = np.eye(10)
y_train = eye[y_train]
y_test = eye[y_test]

print(x_train.shape)  # (60000, 28, 28, 1)
print(y_train.shape)  # (60000, 10)
print(y_train[:3])
# array([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
#        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])

"""
We also support using tf.data.Dataset format for the training data.
"""

train_set = tf.data.Dataset.from_tensor_slices(((x_train,), (y_train,)))
test_set = tf.data.Dataset.from_tensor_slices(((x_test,), (y_test,)))

clf = ak.ImageClassifier(overwrite=True, max_trials=1)
# Feed the tensorflow Dataset to the classifier.
clf.fit(train_set, epochs=10)
# Predict with the best model.
predicted_y = clf.predict(test_set)
# Evaluate the best model with testing data.
print(clf.evaluate(test_set))

"""
## Reference
[ImageClassifier](/image_classifier),
[AutoModel](/auto_model/#automodel-class),
[ImageBlock](/block/#imageblock-class),
[Normalization](/block/#normalization-class),
[ImageAugmentation](/block/#image-augmentation-class),
[ResNetBlock](/block/#resnetblock-class),
[ImageInput](/node/#imageinput-class),
[ClassificationHead](/block/#classificationhead-class).
"""
