##ClassifierBase
Base class of Classifier.
ClassifierBase is the base class of all classifier classes, classifier is used to train and predict data.
####Attributes
**y_encoder**: An instance of OneHotEncoder for y_train (array of categorical labels).

**verbose**: A boolean value indicating the verbosity mode.

**searcher**: An instance of one of the subclasses of Searcher. It search different
    neural architecture to find the best model.

**searcher_type**: The type of searcher to use. It must be 'climb' or 'random'.

**path**: A path to the directory to save the classifier.

**model_id**: Identifier for the best model.

###__init__
Initialize the instance.
The classifier will be loaded from file if the directory in 'path' has a saved classifier. Otherwise it would create a new one.
###_validate
Check x_train's type and the shape of x_train, y_train.

###fit
Find the best model.
Format the input, and split the dataset into training and testing set, save the classifier and find the best model.
####Args
**x_train**: An numpy.ndarray instance contains the training data.

**y_train**: An numpy.ndarray instance contains the label of the training data.

###predict
Return predict result for the testing data.

####Args
**x_test**: An instance of numpy.ndarray contains the testing data.

###summary
Print the summary of the best model.

###_get_searcher_class
Return searcher class based on the 'searcher_type'.

###evaluate
Return the accuracy score between predict value and test_y.

###cross_validate
Do the n_splits cross-validation for the input.

##ImageClassifier
Image classifier class inherited from ClassifierBase class.
It is used for image classification. It searches convolutional neural network architectures for the best configuration for the dataset.
###__init__
