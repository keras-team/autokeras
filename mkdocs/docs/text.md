# Automated text classifier tutorial

### Introduction
Class `TextClassifier` and `TextRegressor` is designed for automated generate best performance cnn neural architecture
for a given text dataset. 

### Example
```python
    clf = TextClassifier(verbose=True)
    clf.fit(x=x_train, y=y_train, batch_size=10, time_limit=12 * 60 * 60)
```
After searching the best model, one can call `clf.final_fit` to test the best model found in searching.

### Arguments

* x_train: string format text data
* y_train: int format text label


### Notes:

Preprocessing of the text data:
* Class `TextClassifier` and `TextRegressor` contains a pre-process of the text data. Which means the input data
should be in string format. 
* The default pre-process model uses the [glove6B model](https://nlp.stanford.edu/projects/glove/) from Stanford NLP. 
* To change the default setting of the pre-process model, one need to change the corresponding variable:
`EMBEDDING_DIM`, `PRE_TRAIN_FILE_LINK`, `PRE_TRAIN_FILE_LINK`, `PRE_TRAIN_FILE_NAME` in `constant.py`.
