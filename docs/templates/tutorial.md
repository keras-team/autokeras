# AutoKeras 1.0 Tutorial

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

See the documentation of Task APIs for more details.



## IO API

The following code example shows how to use IO API for multi-modal and multi-task scenarios using [AutoModel](/auto_model)

```python
import numpy as np
import autokeras as ak
from keras.datasets import mnist

# Prepare the data.
(x_train, y_classification), (x_test, y_test) = mnist.load_data()
x_image = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

x_structured = np.random.rand(x_train.shape[0], 100)
y_regression = np.random.rand(x_train.shape[0], 1)

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

```

Now we support `ImageInput`, `TextInput`, and `StructuredDataInput`.

## Functional API

You can also define your own neural architecture with the predefined blocks and [GraphAutoModel](/graph_auto_model).

```python
import autokeras as ak
import numpy as np
import tensorflow as tf
from keras.datasets import mnist

# Prepare the data.
(x_train, y_classification), (x_test, y_test) = mnist.load_data()
x_image = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

x_structured = np.random.rand(x_train.shape[0], 100)
y_regression = np.random.rand(x_train.shape[0], 1)

# Build model and train.
inputs = ak.ImageInput(shape=(28, 28, 1))
outputs1 = ak.ResNetBlock(version='next')(inputs)
outputs2 = ak.XceptionBlock()(inputs)
image_outputs = ak.Merge()((outputs1, outputs2))

structured_inputs = ak.StructuredInput()
structured_outputs = ak.DenseBlock()(structured_inputs)
merged_outputs = ak.Merge()((image_outputs, structured_outputs))

classification_outputs = ak.ClassificationHead()(merged_outputs)
regression_outputs = ak.RegressionHead()(merged_outputs)
automodel = ak.GraphAutoModel(inputs=inputs,
                              outputs=[regression_outputs,
                                       classification_outputs])

automodel.fit((x_image, x_structured),
              (y_regression, y_classification),
              trials=100,
              epochs=200,
              callbacks=[tf.keras.callbacks.EarlyStopping(),
                         tf.keras.callbacks.LearningRateScheduler()])

```

For complete list of blocks, please checkout the documentation [here](/block).
