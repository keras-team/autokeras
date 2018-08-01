<img src="https://github.com/jhfjhfj1/autokeras/blob/master/logo.png?raw=true" alt="drawing" width="400px"/>

<p><a href="https://travis-ci.org/jhfjhfj1/autokeras"><img alt="Build Status" src="https://travis-ci.org/jhfjhfj1/autokeras.svg?branch=master" style="width: 90px"/></a>
<a href="https://coveralls.io/github/jhfjhfj1/autokeras?branch=master"><img alt="Coverage Status" src="https://coveralls.io/repos/github/jhfjhfj1/autokeras/badge.svg?branch=master" style="width: 99px"/></a></p>

Auto-Keras is an automated machine learning (AutoML) package based on Keras.
It aims at automatically searching for the architecture and hyperparameters for deep learning models.
The ultimate goal of this project is to enable domain experts in fields other than computer science or machine learning
accessible to deep learning models conveniently.

To install the package please use the commend as follows:

    pip install autokeras

Here is a short example of using the package.


    import autokeras as ak

    clf = ak.ImageClassifier()
    clf.fit(x_train, y_train)
    results = clf.predict(x_test)

For the repository on GitHub please visit [Auto-Keras on GitHub](https://github.com/jhfjhfj1/autokeras).

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
