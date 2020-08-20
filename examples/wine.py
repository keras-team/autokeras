"""
This Script searches for a model for the wine dataset
Source and Description of data:
(https://archive.ics.uci.edu/ml/datasets/wine)

1. Download the train and test from in the following links:
[train.csv]
(https://raw.githubusercontent.com/keras-team/autokeras/master/tests/
fixtures/wine/train.csv)
and
[eval.csv](
https://raw.githubusercontent.com/keras-team/autokeras/master/tests/
fixtures/wine/eval.csv
).
2. Replace `PATH_TO/train.csv` and `PATH_TO/eval.csv` in the following example
with the real path to those two files.
Then, you can run the code.
"""
import autokeras as ak

# Initialize the classifier.
clf = ak.StructuredDataClassifier(max_trials=5)
# x is the path to the csv file. y is the column name of the column to predict
clf.fit(x='PATH_TO/train.csv', y='Wine')
# Evaluate the accuracy of the found model.
print('Accuracy: {accuracy}'.format(
    accuracy=clf.evaluate(x='PATH_TO/eval.csv', y='Wine')))
