<img src="https://github.com/jhfjhfj1/autokeras/blob/docs/logo.png?raw=true" alt="drawing" width="400px"/>

[![Build Status](https://travis-ci.org/jhfjhfj1/autokeras.svg?branch=master)](https://travis-ci.org/jhfjhfj1/autokeras)
[![Coverage Status](https://coveralls.io/repos/github/jhfjhfj1/autokeras/badge.svg?branch=master)](https://coveralls.io/github/jhfjhfj1/autokeras?branch=master)

This is a automated machine learning (AutoML) package based on Keras.
It aims at automatically search for the architecture and hyperparameters for deep learning models.
The ultimate goal for this project is for domain experts in fields other than computer science or machine learning
to use deep learning models conveniently.

To install the package please use the commend as follows:

    pip install autokeras

Here is a short example for using the package.


    import autokeras as ak

    clf = ak.ImageClassifier()
    clf.fit(x_train, y_train)
    results = clf.predict(x_test)

For the repository on GitHub visit [Auto-Keras on GitHub](https://github.com/jhfjhfj1/autokeras).

If you use Auto-Keras in a scientific publication, we would appreciate references to the following paper:

Efficient Neural Architecture Search with Network Morphism.
Haifeng Jin, Qingquan Song, Xia Hu.
[arXiv:1806.10282](https://arxiv.org/abs/1806.10282).

Biblatex entry:

    @online{jin2018efficient,
      author       = {Haifeng Jin and Qingquan Song and Xia Hu},
      title        = {Efficient Neural Architecture Search with Network Morphism},
      date         = {2018-06-27},
      year         = {2018},
      eprintclass  = {cs.LG},
      eprinttype   = {arXiv},
      eprint       = {cs.LG/1806.10282},
    }
