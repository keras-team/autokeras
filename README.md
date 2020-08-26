<img src="https://autokeras.com/img/row_red.svg" alt="drawing" width="400px" style="display: block; margin-left: auto; margin-right: auto"/>

[![](https://github.com/keras-team/autokeras/workflows/Tests/badge.svg?branch=master)](https://github.com/keras-team/autokeras/actions?query=workflow%3ATests+branch%3Amaster)
[![codecov](https://codecov.io/gh/keras-team/autokeras/branch/master/graph/badge.svg)](https://codecov.io/gh/keras-team/autokeras)
[![PyPI version](https://badge.fury.io/py/autokeras.svg)](https://badge.fury.io/py/autokeras)

Official Website: [autokeras.com](https://autokeras.com)

##
AutoKeras: An AutoML system based on Keras.
It is developed by <a href="http://faculty.cs.tamu.edu/xiahu/index.html" target="_blank" rel="nofollow">DATA Lab</a> at Texas A&M University.
The goal of AutoKeras is to make machine learning accessible for everyone.

## Example

Here is a short example of using the package.

```python
import autokeras as ak

clf = ak.ImageClassifier()
clf.fit(x_train, y_train)
results = clf.predict(x_test)
```

For detailed tutorial, please check [here](https://autokeras.com/tutorial/overview/).

## Installation

To install the package, please use the `pip` installation as follows:

```shell
pip3 install git+https://github.com/keras-team/keras-tuner.git@1.0.2rc1
pip3 install autokeras
```

Please follow the [installation guide](https://autokeras.com/install) for more details.

**Note:** Currently, AutoKeras is only compatible with **Python >= 3.5** and **TensorFlow >= 2.3.0**.

## Community
### Stay Up-to-Date

**Twitter**:
You can also follow us on Twitter [@autokeras](https://twitter.com/autokeras) for the latest news.

**Emails**:
Subscribe our [email list](https://groups.google.com/forum/#!forum/autokeras-announce/join) to receive announcements.

### Questions and Discussions

**GitHub Discussions**:
Ask your questions on our [GitHub Discussions](https://github.com/keras-team/autokeras/discussions).
It is a forum hosted on GitHub. We will monitor and answer the questions there.

### Instant Communications

**Slack**:
[Request an invitation](https://keras-slack-autojoin.herokuapp.com/).
Use the [#autokeras](https://app.slack.com/client/T0QKJHQRE/CSZ5MKZFU) channel for communication.

**QQ Group**:
Join our QQ group 1150366085. Password: akqqgroup

**Online Meetings**:
Join the [online meeting Google group](https://groups.google.com/forum/#!forum/autokeras/join).
The calendar event will appear on your Google Calendar.


## Contributing Code

We engage in keeping everything about AutoKeras open to the public.
Everyone can easily join as a developer.
Here is how we manage our project.

* **Triage the issues**: 
We pick the important issues to work on from [GitHub issues](https://github.com/keras-team/autokeras/issues).
They will be added to this [Project](https://github.com/keras-team/autokeras/projects/3).
Some of the issues will then be added to the [milestones](https://github.com/keras-team/autokeras/milestones),
which are used to plan for the releases.
* **Assign the tasks**: We assign the tasks to people during the online meetings.
* **Discuss**: We can have discussions in multiple places. The code reviews are on GitHub.
Questions can be asked in Slack or during the meetings.

Please join our [Slack](https://autokeras.com/#community) and send Haifeng Jin a message.
Or drop by our [online meetings](https://autokeras.com/#community) and talk to us.
We will help you get started!

Refer to our [Contributing Guide](https://autokeras.com/contributing/) to learn the best practices.

Thank all the contributors!

<a href="https://github.com/keras-team/autokeras/graphs/contributors"><img src="https://notes.haifengjin.com/img/contributors.svg" /></a>


## Donation

We accept financial support on [Open Collective](https://opencollective.com/autokeras).
Thank every backer for supporting us!

<a href="https://opencollective.com/autokeras#backers" target="_blank"><img src="https://opencollective.com/autokeras/sponsor.svg?avatarHeight=36&width=890&button=false"></a>
<a href="https://opencollective.com/autokeras#backers" target="_blank"><img src="https://opencollective.com/autokeras/backer.svg?avatarHeight=36&width=890&button=false"></a>

## Cite this work

Haifeng Jin, Qingquan Song, and Xia Hu. "Auto-keras: An efficient neural architecture search system." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2019. ([Download](https://www.kdd.org/kdd2019/accepted-papers/view/auto-keras-an-efficient-neural-architecture-search-system))

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
