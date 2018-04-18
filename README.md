# Welcome to Auto-Keras

[![Build Status](https://travis-ci.org/jhfjhfj1/autokeras.svg?branch=master)](https://travis-ci.org/jhfjhfj1/autokeras)
[![Coverage Status](https://coveralls.io/repos/github/jhfjhfj1/autokeras/badge.svg?branch=master)](https://coveralls.io/github/jhfjhfj1/autokeras?branch=master)

This is a automated machine learning (AutoML) package based on Keras. 
It aims at automatically search for the architecture and hyperparameters for deep learning models.
The ultimate goal for this project is for domain experts in fields other than computer science or machine learning
to use deep learning models conveniently.

Here is a short example for using the package.

    
    import autokeras as ak
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    clf = ak.ImageClassifier()
    clf.fit(x_train, y_train)
    results = clf.predict(x_test)

For the repository on GitHub visit [Auto-Keras on GitHub](https://github.com/jhfjhfj1/autokeras).


### About

This package is developed by [DATA LAB](http://faculty.cs.tamu.edu/xiahu/) at Texas A&M University.
