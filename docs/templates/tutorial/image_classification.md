# Image Classification
## A Simple Example
The first step is to prepare your data. Here we use the [MNIST
dataset](https://keras.io/datasets/#mnist-database-of-handwritten-digits) as an
example.

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,)
print(y_train[:3]) # array([7, 2, 1], dtype=uint8)
```

The second step is to run the [ImageClassifier](/image_classifier).

```python
import autokeras as ak

# Initialize the image classifier.
clf = ak.ImageClassifier(max_trials=10) # It tries 10 different models.
# Feed the image classifier with training data.
clf.fit(x_train, y_train)
# Predict with the best model.
predicted_y = clf.predict(x_test)
# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))
```


## Validation Data
By default, AutoKeras use the last 20% of training data as validation data.
As shown in the example below, you can use `validation_split` to specify the percentage.

```python
clf.fit(x_train,
        y_train,
        # Split the training data and use the last 15% as validation data.
        validation_split=0.15)
```

You can also use your own validation set
instead of splitting it from the training data with `validation_data`.

```python
split = 50000
x_val = x_train[split:]
y_val = y_train[split:]
x_train = x_train[:split]
y_train = y_train[:split]
clf.fit(x_train,
        y_train,
        # Use your own validation set.
        validation_data=(x_val, y_val))
```

## Customized Search Space
For advanced users, you may customize your search space by using
[AutoModel](/auto_model/#automodel-class) instead of
[ImageClassifier](/image_classifier). You can configure the
[ImageBlock](/block/#imageblock-class) for some high-level configurations,
`block_type` for the type of neural network to search, `normalize` for whether to do
data normalization, `augment` for whether to do data augmentation. You can also
do not specify these arguments, which would leave the different choices to be
tuned automatically.
See the following example for detail.

```python
import autokeras as ak

input_node = ak.ImageInput()
output_node = ak.ImageBlock(
    # Only search ResNet architectures.
    block_type='resnet',
    # Normalize the dataset.
    normalize=True,
    # Do not do data augmentation.
    augment=False)(input_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=10)
clf.fit(x_train, y_train)
```
The usage of [AutoModel](/auto_model/#automodel-class) is similar to the
[functional API](https://www.tensorflow.org/guide/keras/functional) of Keras.
Basically, you are building a graph, whose edges are blocks and the nodes are intermediate outputs of blocks.
To add an edge from `input_node` to `output_node` with
`output_node = ak.[some_block]([block_args])(input_node)`.

You can even also use more fine grained blocks to customize the search space even
further. See the following example.

```python
import autokeras as ak

input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node = ak.ImageAugmentation(percentage=0.3)(output_node)
output_node = ak.ResNetBlock(version='v2')(output_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=10)
clf.fit(x_train, y_train)
```

## Data Format
The AutoKeras ImageClassifier is quite flexible for the data format.

For the image, it accepts data formats both with and without the
channel dimension. The images in the MNIST dataset do not have the channel dimension.
Each image is a matrix with shape (28, 28).  AutoKeras also accepts images of three
dimensions with the channel dimension at last, e.g., (32, 32, 3), (28, 28, 1).

For the classification labels, AutoKeras accepts both plain labels, i.e. strings or
integers, and one-hot encoded encoded labels, i.e. vectors of 0s and 1s.

So if you prepare your data in the following way, the ImageClassifier should still
work.

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the images to have the channel dimension.
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

# One-hot encode the labels.
import numpy as np
eye = np.eye(10)
y_train = eye[y_train]
y_test = eye[y_test]

print(x_train.shape) # (60000, 28, 28, 1)
print(y_train.shape) # (60000, 10)
print(y_train[:3])
# array([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
#        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])
```

We also support using [tf.data.Dataset](
https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=stable) format for
the training data. In this case, the images would have to be 3-dimentional. The
labels have to be one-hot encoded.  So you can wrap the prepared reshaped and one-hot
encoded data above into tensorflow Dataset as follows.

```python
import tensorflow as tf
train_set = tf.data.Dataset.from_tensor_slices(((x_train, ), (y_train, )))
test_set = tf.data.Dataset.from_tensor_slices(((x_test, ), (y_test, )))

clf = ak.ImageClassifier(max_trials=10)
# Feed the tensorflow Dataset to the classifier.
clf.fit(train_set)
# Predict with the best model.
predicted_y = clf.predict(test_set)
# Evaluate the best model with testing data.
print(clf.evaluate(test_set))
```

## Reference
[ImageClassifier](/image_classifier),
[AutoModel](/auto_model/#automodel-class),
[ImageBlock](/block/#imageblock-class),
[Normalization](/preprocessor/#normalization-class),
[ImageAugmentation](/preprocessor/#image-augmentation-class),
[ResNetBlock](/block/#resnetblock-class),
[ClassificationHead](/head/#classification-head-class)
