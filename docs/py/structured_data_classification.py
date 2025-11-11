"""shell
pip install autokeras
"""

import keras
import pandas as pd

import autokeras as ak

"""
## A Simple Example
The first step is to prepare your data. Here we use the [Titanic
dataset](https://www.kaggle.com/c/titanic) as an example.
"""


TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = keras.utils.get_file("eval.csv", TEST_DATA_URL)

# Load data into numpy arrays
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

y_train = train_df["survived"].values
x_train = train_df.drop("survived", axis=1).values

y_test = test_df["survived"].values
x_test = test_df.drop("survived", axis=1).values


"""
The second step is to run the
[StructuredDataClassifier](/structured_data_classifier).
As a quick demo, we set epochs to 10.
You can also leave the epochs unspecified for an adaptive number of epochs.
"""

# Initialize the structured data classifier.
clf = ak.StructuredDataClassifier(
    overwrite=True, max_trials=3
)  # It tries 3 different models.
# Feed the structured data classifier with training data.
clf.fit(
    x_train,
    y_train,
    epochs=10,
)
# Predict with the best model.
predicted_y = clf.predict(x_test)
# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))


"""
You can also specify the column names and types for the data as follows.  The
`column_names` is optional if the training data already have the column names,
e.g.  pandas.DataFrame, CSV file.  Any column, whose type is not specified will
be inferred from the training data.
"""

# Initialize the structured data classifier.
clf = ak.StructuredDataClassifier(
    column_names=[
        "sex",
        "age",
        "n_siblings_spouses",
        "parch",
        "fare",
        "class",
        "deck",
        "embark_town",
        "alone",
    ],
    column_types={"sex": "categorical", "fare": "numerical"},
    max_trials=10,  # It tries 10 different models.
    overwrite=True,
)


"""
## Validation Data
By default, AutoKeras use the last 20% of training data as validation data.  As
shown in the example below, you can use `validation_split` to specify the
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
You can also use your own validation set
instead of splitting it from the training data with `validation_data`.
"""

split = 500
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
## Reference
[StructuredDataClassifier](/structured_data_classifier),
[AutoModel](/auto_model/#automodel-class),
[StructuredDataBlock](/block/#structureddatablock-class),
[DenseBlock](/block/#denseblock-class),
[StructuredDataInput](/node/#structureddatainput-class),
[ClassificationHead](/block/#classificationhead-class),
"""
