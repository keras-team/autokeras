"""shell
pip install autokeras
"""

import pandas as pd

import autokeras as ak

"""
## Social Media Articles Example

Regression tasks estimate a numeric variable, such as the price of a house
or a person's age.

This example estimates the view counts for an article on social media platforms,
trained on a
[News Popularity](
https://archive.ics.uci.edu/ml/datasets/
News+Popularity+in+Multiple+Social+Media+Platforms)
dataset collected from 2015-2016.

First, prepare your text data in a `numpy.ndarray` format.
"""


# converting from other formats (such as pandas) to numpy
df = pd.read_csv("./News_Final.csv")

text_inputs = df.Title.to_numpy(dtype="str")
media_success_outputs = df.Facebook.to_numpy(dtype="int")

"""
Next, initialize and train the [TextRegressor](/text_regressor).
"""


# Initialize the text regressor
reg = ak.TextRegressor(max_trials=15)  # AutoKeras tries 15 different models.

# Find the best model for the given training data
reg.fit(text_inputs, media_success_outputs)

# Predict with the chosen model:
predict_y = reg.predict(text_inputs)
