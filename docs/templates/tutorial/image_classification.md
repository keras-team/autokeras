# Image Classification
## A Simple Example
The first step is to prepare your data. Here we use MNIST dataset as an example.

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,)
print(y_train[:3]) # array([7, 2, 1], dtype=uint8)
```

The second step is to run the image classifier.

```python
import autokeras as ak

# Initialize the image classifier.
clf = ak.ImageClassifier()
# Feed the image classifier with training data.
clf.fit(x_train,
        y_train,
        # It tries 10 different models.
        max_trials=10)
# Predict with the best model.
predicted_y = clf.predict(x_test)
# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))
```

## Data Format
The AutoKeras ImageClassifier is quite flexible for the data format.

For the image, it accepts data formates both with channel dimension and without
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

We also support using tensorflow Dataset format for the training data. In this case,
the images would have to be 3-dimentional. The labels have to be one-hot encoded.  So
you can wrap the prepared reshaped and one-hot encoded data above into tensorflow
Dataset as follows.

```python
import tensorflow as tf
train_set = tf.data.Dataset.from_tensor_slices(((x_train, ), (y_train, )))
test_set = tf.data.Dataset.from_tensor_slices(((x_test, ), (y_test, )))

clf = ak.ImageClassifier()
# Feed the tensorflow Dataset to the classifier.
clf.fit(train_set,
        # It tries 10 different models.
        max_trials=10)
# Predict with the best model.
predicted_y = clf.predict(test_set)
# Evaluate the best model with testing data.
print(clf.evaluate(test_set))
```


## Configuration
loss
metrics
max_trials
directory
objective
overwrite
multi_label

```python
```

validation_split & validation_data
other training args

```python
```
