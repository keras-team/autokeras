# Auto-Keras

[![Build Status](https://travis-ci.org/jhfjhfj1/autokeras.svg?branch=master)](https://travis-ci.org/jhfjhfj1/autokeras)

This is a automated machine learning (AutoML) package based on Keras. 
It aims at automatically search for the architecture and hyperparameters for deep learning models.
The ultimate goal for this project is for domain experts in fields other than computer science or machine learning
to use deep learning models conveniently.

Currently, only a default architecture is selected. More complicated searching algorithms will be added soon.

Here is a short example for using the package.

    
    import autokeras as ak
    
    train_x = np.random.rand(100, 25)
    test_x = np.random.rand(100, 25)
    train_y = np.random.randint(0, 5, 100)
    
    clf = ak.AutoKerasClassifier()
    clf.fit(train_x, train_y)
    results = clf.predict(test_x)
