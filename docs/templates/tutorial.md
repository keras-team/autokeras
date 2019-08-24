# AutoKeras 1.0 Tutorial

**This is only a draft tutorial. 1.0 version has not been released yet.**

In AutoKeras, there are 3 levels of APIs: task API, IO API, and functional API.

## Task API
We have designed an extremely simple interface for a series of tasks.
The following code example shows how to do image classification with the task API.

```python
import autokeras as ak
from keras.datasets import mnist

# Prepare the data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

# Search and train the classifier.
clf = ak.ImageClassifier(max_trials=100)
clf.fit(x_train, y_train)
y = clf.predict(x_test, y_test)
```

See the [documentation of Task API](/task) for more details.



## IO API

The following code example shows how to use IO API for multi-modal and multi-task scenarios.

```python
```
## Functional API
