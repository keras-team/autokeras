"""
Search for a good model for the [Titanic](https://www.kaggle.com/c/titanic) dataset.
First, you need to download the titanic dataset file
[train.csv](
https://raw.githubusercontent.com/keras-team/autokeras/master/tests/
fixtures/titanic/train.csv
)
and
[eval.csv](
https://raw.githubusercontent.com/keras-team/autokeras/master/tests/
fixtures/titanic/eval.csv
).
Second, replace `PATH_TO/train.csv` and `PATH_TO/eval.csv` in the following example
with the real path to those two files.
Then, you can run the code.
"""

import autokeras as ak

# Initialize the classifier.
clf = ak.StructuredDataClassifier(max_trials=30)
# x is the path to the csv file. y is the column name of the column to predict.
clf.fit(x='PATH_TO/train.csv', y='survived')
# Evaluate the accuracy of the found model.
print('Accuracy: {accuracy}'.format(
    accuracy=clf.evaluate(x='PATH_TO/eval.csv', y='survived')))
