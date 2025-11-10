"""shell
pip install autokeras
"""

from sklearn.datasets import fetch_california_housing

import autokeras as ak

"""
## A Simple Example
The first step is to prepare your data. Here we use the [California housing
dataset](
https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
as an example.
"""


house_dataset = fetch_california_housing()
train_size = int(house_dataset.data.shape[0] * 0.9)

x_train = house_dataset.data[:train_size]
y_train = house_dataset.target[:train_size]
x_test = house_dataset.data[train_size:]
y_test = house_dataset.target[train_size:]

"""
The second step is to run the
[StructuredDataRegressor](/structured_data_regressor).
As a quick demo, we set epochs to 10.
You can also leave the epochs unspecified for an adaptive number of epochs.
"""

# Initialize the structured data regressor.
reg = ak.StructuredDataRegressor(
    overwrite=True, max_trials=3
)  # It tries 3 different models.
# Feed the structured data regressor with training data.
reg.fit(
    x_train,
    y_train,
    epochs=10,
)
# Predict with the best model.
predicted_y = reg.predict(x_test)
# Evaluate the best model with testing data.
print(reg.evaluate(x_test, y_test))

"""
You can also specify the column names and types for the data as follows.  The
`column_names` is optional if the training data already have the column names,
e.g.  pandas.DataFrame, CSV file.  Any column, whose type is not specified will
be inferred from the training data.
"""

# Initialize the structured data regressor.
reg = ak.StructuredDataRegressor(
    column_names=[
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ],
    column_types={"MedInc": "numerical", "Latitude": "numerical"},
    max_trials=10,  # It tries 10 different models.
    overwrite=True,
)


"""
## Validation Data
By default, AutoKeras use the last 20% of training data as validation data.  As
shown in the example below, you can use `validation_split` to specify the
percentage.
"""

reg.fit(
    x_train,
    y_train,
    # Split the training data and use the last 15% as validation data.
    validation_split=0.15,
    epochs=10,
)

"""
You can also use your own validation set
instead of splitting it from the training data with `validation_data`.
"""

split = 500
x_val = x_train[split:]
y_val = y_train[split:]
x_train = x_train[:split]
y_train = y_train[:split]
reg.fit(
    x_train,
    y_train,
    # Use your own validation set.
    validation_data=(x_val, y_val),
    epochs=10,
)

"""
## Customized Search Space
For advanced users, you may customize your search space by using
[AutoModel](/auto_model/#automodel-class) instead of
[StructuredDataRegressor](/structured_data_regressor). You can configure the
[StructuredDataBlock](/block/#structureddatablock-class) for some high-level
configurations, e.g., `categorical_encoding` for whether to use the
[CategoricalToNumerical](/block/#categoricaltonumerical-class). You can also do
not specify these arguments, which would leave the different choices to be
tuned automatically. See the following example for detail.
"""


input_node = ak.StructuredDataInput()
output_node = ak.StructuredDataBlock(categorical_encoding=True)(input_node)
output_node = ak.RegressionHead()(output_node)
reg = ak.AutoModel(
    inputs=input_node, outputs=output_node, overwrite=True, max_trials=3
)
reg.fit(x_train, y_train, epochs=10)

"""
The usage of [AutoModel](/auto_model/#automodel-class) is similar to the
[functional API](https://keras.io/api/models/model/#with-the-functional-api) of
Keras.
Basically, you are building a graph, whose edges are blocks and the nodes are
intermediate outputs of blocks.  To add an edge from `input_node` to
`output_node` with `output_node = ak.[some_block]([block_args])(input_node)`.

You can even also use more fine grained blocks to customize the search space
even further. See the following example.
"""


input_node = ak.StructuredDataInput()
output_node = ak.CategoricalToNumerical()(input_node)
output_node = ak.DenseBlock()(output_node)
output_node = ak.RegressionHead()(output_node)
reg = ak.AutoModel(
    inputs=input_node, outputs=output_node, max_trials=3, overwrite=True
)
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
