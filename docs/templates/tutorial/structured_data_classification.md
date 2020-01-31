# Structured Data Classification
## A Simple Example
The first step is to prepare your data. Here we use the [Titanic
dataset](https://www.kaggle.com/c/titanic) as an example. You can download the CSV
files [here](https://github.com/keras-team/autokeras/tree/master/tests/fixtures/titanic).

The second step is to run the
[StructuredDataClassifier](/structured_data_classifier).
Replace all the `/path/to` with the path to the csv files.

```python
import autokeras as ak

# Initialize the structured data classifier.
clf = ak.StructuredDataClassifier(max_trials=10) # It tries 10 different models.
# Feed the structured data classifier with training data.
clf.fit(
    # The path to the train.csv file.
    '/path/to/train.csv',
    # The name of the label column.
    'survived')
# Predict with the best model.
predicted_y = clf.predict('/path/to/eval.csv')
# Evaluate the best model with testing data.
print(clf.evaluate('/path/to/eval.csv', 'survived'))
```

## Data Format
The AutoKeras StructuredDataClassifier is quite flexible for the data format.

The example above shows how to use the CSV files directly. Besides CSV files, it also
supports numpy.ndarray, pandas.DataFrame or [tf.data.Dataset](
https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=stable). The data should be
two-dimensional with numerical or categorical values.

For the classification labels,
AutoKeras accepts both plain labels, i.e.  strings or integers, and one-hot encoded
encoded labels, i.e. vectors of 0s and 1s.
The labels can be numpy.ndarray, pandas.DataFrame, or pandas.Series.

The following examples show how the data can be prepared with numpy.ndarray,
pandas.DataFrame, and tensorflow.data.Dataset.

```python
import pandas as pd
# x_train as pandas.DataFrame, y_train as pandas.Series
x_train = pd.read_csv('train.csv')
print(type(x_train)) # pandas.DataFrame
y_train = x_train.pop('survived')
print(type(y_train)) # pandas.Series

# You can also use pandas.DataFrame for y_train.
y_train = pd.DataFrame(y_train)
print(type(y_train)) # pandas.DataFrame

# You can also use numpy.ndarray for x_train and y_train.
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
print(type(x_train)) # numpy.ndarray
print(type(y_train)) # numpy.ndarray

# Preparing testing data.
x_test = pd.read_csv('eval.csv')
y_test = x_test.pop('survived')

# It tries 10 different models.
clf = ak.StructuredDataClassifier(max_trials=10)
# Feed the structured data classifier with training data.
clf.fit(x_train, y_train)
# Predict with the best model.
predicted_y = clf.predict(x_test)
# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))
```

The following code shows how to convert numpy.ndarray to tf.data.Dataset.
Notably, the labels have to be one-hot encoded for multi-class
classification to be wrapped into tensorflow Dataset.
Since the Titanic dataset is binary
classification, it should not be one-hot encoded.

```python
import tensorflow as tf
train_set = tf.data.Dataset.from_tensor_slices(((x_train, ), (y_train, )))
test_set = tf.data.Dataset.from_tensor_slices(((x_test, ), (y_test, )))

clf = ak.StructuredDataClassifier(max_trials=10)
# Feed the tensorflow Dataset to the classifier.
clf.fit(train_set)
# Predict with the best model.
predicted_y = clf.predict(test_set)
# Evaluate the best model with testing data.
print(clf.evaluate(test_set))
```

You can also specify the column names and types for the data as follows.
The `column_names` is optional if the training data already have the column names, e.g.
pandas.DataFrame, CSV file.
Any column, whose type is not specified will be inferred from the training data.

```python
# Initialize the structured data classifier.
clf = ak.StructuredDataClassifier(
    column_names=[
        'sex',
        'age',
        'n_siblings_spouses',
        'parch',
        'fare',
        'class',
        'deck',
        'embark_town',
        'alone'],
    column_types={'sex': 'categorical', 'fare': 'numerical'},
    max_trials=10, # It tries 10 different models.
)
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
split = 500
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
[StructuredDataClassifier](/structured_data_classifier). You can configure the
[StructuredDataBlock](/block/#structureddatablock-class) for some high-level
configurations, e.g., `categorical_encoding` for whether to use the
[CategoricalToNumerical](/preprocessor/#categoricaltonumerical-class). You can also do not specify these
arguments, which would leave the different choices to be tuned automatically. See
the following example for detail.

```python
import autokeras as ak

input_node = ak.StructuredDataInput()
output_node = ak.StructuredDataBlock(
    categorical_encoding=True,
    block_type='dense')(input_node)
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

input_node = ak.StructuredDataInput()
output_node = ak.CategoricalToNumerical()(input_node)
output_node = ak.DenseBlock()(output_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=10)
clf.fit(x_train, y_train)
```


## Reference
[StructuredDataClassifier](/structured_data_classifier),
[AutoModel](/auto_model/#automodel-class),
[StructuredDataClassifier](/structured_data_classifier),
[StructuredDataBlock](/block/#structureddatablock-class),
[DenseBlock](/block/#denseblock-class),
[StructuredDataInput](/node/#structureddatainput-class),
[ClassificationHead](/head/#classificationhead-class),
[CategoricalToNumerical](/preprocessor/#categoricaltonumerical-class).
