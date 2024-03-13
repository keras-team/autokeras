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

First, prepare your text data in a `numpy.ndarray` or `tensorflow.Dataset`
format.
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

"""
If your text source has a larger vocabulary (number of distinct words), you may
need to create a custom pipeline in AutoKeras to increase the `max_tokens`
parameter.
"""

text_input = (df.Title + " " + df.Headline).to_numpy(dtype="str")

# text input and tokenization
input_node = ak.TextInput()
output_node = ak.TextToIntSequence(max_tokens=20000)(input_node)

# regression output
output_node = ak.RegressionHead()(output_node)

# initialize AutoKeras and find the best model
reg = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=15)
reg.fit(text_input, media_success_outputs)

"""
Measure the accuracy of the regressor on an independent test set:
"""

print(reg.evaluate(text_input, media_success_outputs))
