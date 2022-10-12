<p align="center">
  <img width="500" alt="logo" src="https://autokeras.com/img/row_red.svg"/>
</p>

[![](https://github.com/keras-team/autokeras/workflows/Tests/badge.svg?branch=master)](https://github.com/keras-team/autokeras/actions?query=workflow%3ATests+branch%3Amaster)
[![codecov](https://codecov.io/gh/keras-team/autokeras/branch/master/graph/badge.svg)](https://codecov.io/gh/keras-team/autokeras)
[![PyPI version](https://badge.fury.io/py/autokeras.svg)](https://badge.fury.io/py/autokeras)
[![Python](https://img.shields.io/badge/python-v3.7.0+-success.svg)](https://www.python.org/downloads/)
[![Tensorflow](https://img.shields.io/badge/tensorflow-v2.8.0+-success.svg)](https://www.tensorflow.org/versions)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/keras-team/autokeras/issues)

Official Website: [autokeras.com](https://autokeras.com)

##
AutoKeras: An AutoML system based on Keras.
It is developed by <a href="http://faculty.cs.tamu.edu/xiahu/index.html" target="_blank" rel="nofollow">DATA Lab</a> at Texas A&M University.
The goal of AutoKeras is to make machine learning accessible to everyone.

## Learning resources

* A short example.

```python
import autokeras as ak

clf = ak.ImageClassifier()
clf.fit(x_train, y_train)
results = clf.predict(x_test)
```

* [Official website tutorials](https://autokeras.com/tutorial/overview/).
* The book of [*Automated Machine Learning in Action*](https://www.manning.com/books/automated-machine-learning-in-action?query=automated&utm_source=jin&utm_medium=affiliate&utm_campaign=affiliate&a_aid=jin).
* The LiveProjects of [*Image Classification with AutoKeras*](https://www.manning.com/liveprojectseries/autokeras-ser).
<p align="center">
<a href="https://www.manning.com/books/automated-machine-learning-in-action?query=automated&utm_source=jin&utm_medium=affiliate&utm_campaign=affiliate&a_aid=jin"><img src="https://images.manning.com/360/480/resize/book/0/fc56aaf-b2ba-4ef4-85b3-4a31edbe8ecc/Song-AML-HI.png" alt="drawing" width="266"/></a>
&nbsp
&nbsp
<a href="https://www.manning.com/liveprojectseries/autokeras-ser"><img src="https://images.manning.com/360/480/resize/liveProjectSeries/9/38c715a-0c8c-4f66-b440-83d29993877a/ImageClassificationwithAutoKeras.jpg" alt="drawing" width="250"/></a>
</p>


## Installation

To install the package, please use the `pip` installation as follows:

```shell
pip3 install autokeras
```

Please follow the [installation guide](https://autokeras.com/install) for more details.

**Note:** Currently, AutoKeras is only compatible with **Python >= 3.7** and **TensorFlow >= 2.8.0**.

## Community

Ask your questions on our [GitHub Discussions](https://github.com/keras-team/autokeras/discussions).

## Contributing Code

Here is how we manage our project.

We pick the critical issues to work on from [GitHub issues](https://github.com/keras-team/autokeras/issues).
They will be added to this [Project](https://github.com/keras-team/autokeras/projects/3).
Some of the issues will then be added to the [milestones](https://github.com/keras-team/autokeras/milestones),
which are used to plan for the releases.

Refer to our [Contributing Guide](https://autokeras.com/contributing/) to learn the best practices.

Thank all the contributors!

<a href="https://github.com/keras-team/autokeras/graphs/contributors"><img src="https://autokeras.com/img/contributors.svg" /></a>


## Donation

Thank all the donors for supporting us!

<a href="https://opencollective.com/autokeras#backers" target="_blank"><img src="https://opencollective.com/autokeras/sponsor.svg?avatarHeight=36&width=890&button=false"></a>
<a href="https://opencollective.com/autokeras#backers" target="_blank"><img src="https://opencollective.com/autokeras/backer.svg?avatarHeight=36&width=890&button=false"></a>

## Cite this work

Haifeng Jin, Qingquan Song, and Xia Hu. "Auto-keras: An efficient neural architecture search system." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2019. ([Download](https://dl.acm.org/doi/pdf/10.1145/3292500.3330648))

Biblatex entry:

```bibtex
@inproceedings{jin2019auto,
  title={Auto-Keras: An Efficient Neural Architecture Search System},
  author={Jin, Haifeng and Song, Qingquan and Hu, Xia},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1946--1956},
  year={2019},
  organization={ACM}
}
```

## Acknowledgements

The authors gratefully acknowledge the D3M program of the Defense Advanced Research Projects Agency (DARPA) administered through AFRL contract FA8750-17-2-0116; the Texas A&M College of Engineering, and Texas A&M University.
