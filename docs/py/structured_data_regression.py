"""shell
pip install autokeras
pip install git+https://github.com/keras-team/keras-tuner.git
"""

"""
## A Simple Example
The first step is to prepare your data. Here we use the [California housing
dataset](https://scikit-learn.org/stable/datasets/index.html#california-housing-dataset) as an example.
"""

from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak

house_dataset = fetch_california_housing()
df = pd.DataFrame(
    np.concatenate((
        house_dataset.data, 
        house_dataset.target.reshape(-1,1)),
        axis=1),
    columns=house_dataset.feature_names + ['Price'])
train_size = int(df.shape[0] * 0.9)
df[:train_size].to_csv('train.csv', index=False)
df[train_size:].to_csv('eval.csv', index=False)
train_file_path = 'train.csv'
test_file_path = 'eval.csv'

"""
The second step is to run the
[StructuredDataRegressor](/structured_data_regressor).
As a quick demo, we set epochs to 10.
You can also leave the epochs unspecified for an adaptive number of epochs.
"""

# Initialize the structured data regressor.
reg = ak.StructuredDataRegressor(
    overwrite=True,
    max_trials=3) # It tries 3 different models.
# Feed the structured data regressor with training data.
reg.fit(
    # The path to the train.csv file.
    train_file_path,
    # The name of the label column.
    'Price',
    epochs=10)
# Predict with the best model.
predicted_y = reg.predict(test_file_path)
# Evaluate the best model with testing data.
print(reg.evaluate(test_file_path, 'Price'))

"""
## Data Format
The AutoKeras StructuredDataRegressor is quite flexible for the data format.

The example above shows how to use the CSV files directly. Besides CSV files, it also
supports numpy.ndarray, pandas.DataFrame or [tf.data.Dataset](
https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=stable). The data should be
two-dimensional with numerical or categorical values.

For the regression targets, it should be a vector of numerical values.
AutoKeras accepts numpy.ndarray, pandas.DataFrame, or pandas.Series.

The following examples show how the data can be prepared with numpy.ndarray,
pandas.DataFrame, and tensorflow.data.Dataset.
"""

import pandas as pd
import numpy as np
# x_train as pandas.DataFrame, y_train as pandas.Series
x_train = pd.read_csv(train_file_path)
print(type(x_train)) # pandas.DataFrame
y_train = x_train.pop('Price')
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
x_test = pd.read_csv(test_file_path)
y_test = x_test.pop('Price')

# It tries 10 different models.
reg = ak.StructuredDataRegressor(max_trials=3, overwrite=True)
# Feed the structured data regressor with training data.
reg.fit(x_train, y_train, epochs=10)
# Predict with the best model.
predicted_y = reg.predict(x_test)
# Evaluate the best model with testing data.
print(reg.evaluate(x_test, y_test))

"""
The following code shows how to convert numpy.ndarray to tf.data.Dataset.
"""

train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))

reg = ak.StructuredDataRegressor(max_trials=3, overwrite=True)
# Feed the tensorflow Dataset to the regressor.
reg.fit(train_set, epochs=10)
# Predict with the best model.
predicted_y = reg.predict(test_set)
# Evaluate the best model with testing data.
print(reg.evaluate(test_set))

"""
You can also specify the column names and types for the data as follows.
The `column_names` is optional if the training data already have the column names, e.g.
pandas.DataFrame, CSV file.
Any column, whose type is not specified will be inferred from the training data.
"""

# Initialize the structured data regressor.
reg = ak.StructuredDataRegressor(
    column_names=[
        'MedInc', 'HouseAge', 'AveRooms', 
        'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'],
    column_types={'MedInc': 'numerical', 'Latitude': 'numerical'},
    max_trials=10, # It tries 10 different models.
    overwrite=True,
)


"""
## Validation Data
By default, AutoKeras use the last 20% of training data as validation data.
As shown in the example below, you can use `validation_split` to specify the percentage.
"""

reg.fit(x_train,
        y_train,
        # Split the training data and use the last 15% as validation data.
        validation_split=0.15,
        epochs=10)

"""
You can also use your own validation set
instead of splitting it from the training data with `validation_data`.
"""

split = 500
x_val = x_train[split:]
y_val = y_train[split:]
x_train = x_train[:split]
y_train = y_train[:split]
reg.fit(x_train,
        y_train,
        # Use your own validation set.
        validation_data=(x_val, y_val),
        epochs=10)

"""
## Customized Search Space
For advanced users, you may customize your search space by using
[AutoModel](/auto_model/#automodel-class) instead of
[StructuredDataRegressor](/structured_data_regressor). You can configure the
[StructuredDataBlock](/block/#structureddatablock-class) for some high-level
configurations, e.g., `categorical_encoding` for whether to use the
[CategoricalToNumerical](/block/#categoricaltonumerical-class). You can also do not specify these
arguments, which would leave the different choices to be tuned automatically. See
the following example for detail.
"""

import autokeras as ak

input_node = ak.StructuredDataInput()
output_node = ak.StructuredDataBlock(categorical_encoding=True)(input_node)
output_node = ak.RegressionHead()(output_node)
reg = ak.AutoModel(
    inputs=input_node, 
    outputs=output_node, 
    overwrite=True,
    max_trials=3)
reg.fit(x_train, y_train, epochs=10)

"""
The usage of [AutoModel](/auto_model/#automodel-class) is similar to the
[functional API](https://www.tensorflow.org/guide/keras/functional) of Keras.
Basically, you are building a graph, whose edges are blocks and the nodes are intermediate outputs of blocks.
To add an edge from `input_node` to `output_node` with
`output_node = ak.[some_block]([block_args])(input_node)`.

You can even also use more fine grained blocks to customize the search space even
further. See the following example.
"""

import autokeras as ak

input_node = ak.StructuredDataInput()
output_node = ak.CategoricalToNumerical()(input_node)
output_node = ak.DenseBlock()(output_node)
output_node = ak.RegressionHead()(output_node)
reg = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=3,
                   overwrite=True)
reg.fit(x_train, y_train, epochs=10)

"""
You can also export the best model found by AutoKeras as a Keras Model.
"""

model = reg.export_model()
model.summary()
# numpy array in object (mixed type) is not supported.
# you need convert it to unicode or float first.
model.predict(x_train)


"""
## Reference
[StructuredDataRegressor](/structured_data_regressor),
[AutoModel](/auto_model/#automodel-class),
[StructuredDataBlock](/block/#structureddatablock-class),
[DenseBlock](/block/#denseblock-class),
[StructuredDataInput](/node/#structureddatainput-class),
[RegressionHead](/block/#regressionhead-class),
[CategoricalToNumerical](/block/#categoricaltonumerical-class).
"""
