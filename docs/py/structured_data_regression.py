"""shell
export KERAS_BACKEND="torch"
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
## Reference
[StructuredDataRegressor](/structured_data_regressor),
[AutoModel](/auto_model/#automodel-class),
[StructuredDataBlock](/block/#structureddatablock-class),
[DenseBlock](/block/#denseblock-class),
[StructuredDataInput](/node/#structureddatainput-class),
[RegressionHead](/block/#regressionhead-class),
"""
