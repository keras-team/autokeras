<img src="https://github.com/jhfjhfj1/autokeras/blob/master/logo.png?raw=true" alt="drawing" width="400px"/>

<p><a href="https://travis-ci.org/jhfjhfj1/autokeras"><img alt="Build Status" src="https://travis-ci.org/jhfjhfj1/autokeras.svg?branch=master" style="width: 90px"/></a>
<a href="https://coveralls.io/github/jhfjhfj1/autokeras?branch=master"><img alt="Coverage Status" src="https://coveralls.io/repos/github/jhfjhfj1/autokeras/badge.svg?branch=master" style="width: 99px"/></a></p>

Auto-Keras is an open source software library for automated machine learning (AutoML) based on Keras. The ultimate goal of AutoML is to allow domain experts with limited data science or machine learning background easily accessible to deep learning models.
Auto-Keras provides functions to automatically search for architecture and hyperparameters of deep learning models.

To install the package please use the commend as follows:

    pip install autokeras

Here is a short example of using the package.


    import autokeras as ak

    clf = ak.ImageClassifier()
    clf.fit(x_train, y_train)
    results = clf.predict(x_test)

For the documentation, please visit the [Auto-Keras](https://autokeras.com/) official website.

If you use Auto-Keras in a scientific publication, we would appreciate references to the following paper:

Efficient Neural Architecture Search with Network Morphism.
Haifeng Jin, Qingquan Song, and Xia Hu.
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
