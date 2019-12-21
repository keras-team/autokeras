# Image Classification
## A Simple Example
The first step is to prepare your data.

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,)
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
# Evaluate the best model with testing data.
print(clf.evaluate(x_test))
```

## Data Format
The 

Each image in the MNIST dataset is a matrix with shape (28, 28). AutoKeras accepts
images either 2-dimensional or 3-dimensional, with the channel dimension as the last
axis, e.g., (32, 32, 3).

The classification labels of MNIST dataset are integers. Each label is represented by
one integer, e.g., 0 represents the .

