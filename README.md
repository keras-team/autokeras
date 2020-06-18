<img src="https://autokeras.com/img/row_red.svg" alt="drawing" width="400px" style="display: block; margin-left: auto; margin-right: auto"/>

![](https://github.com/keras-team/autokeras/workflows/Tests/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/keras-team/autokeras/branch/master/graph/badge.svg)](https://codecov.io/gh/keras-team/autokeras)
[![PyPI version](https://badge.fury.io/py/autokeras.svg)](https://badge.fury.io/py/autokeras)

Official Website: [autokeras.com](https://autokeras.com)

# Let's Chat!
To make AutoKeras better, I would like to hear your thoughts.
I am happy to answer any questions you have about our project.
[Join our Slack](#community) and send me (Haifeng Jin) a message.
I will schedule a meeting with you.

##
AutoKeras: An AutoML system based on Keras.
It is developed by <a href="http://faculty.cs.tamu.edu/xiahu/index.html" target="_blank" rel="nofollow">DATA Lab</a> at Texas A&M University.
The goal of AutoKeras is to make machine learning accessible for everyone.

## Example

Here is a short example of using the package.

```
import autokeras as ak

clf = ak.ImageClassifier()
clf.fit(x_train, y_train)
results = clf.predict(x_test)
```

For detailed tutorial, please check [here](https://autokeras.com/tutorial/overview/).

## Installation

To install the package, please use the `pip` installation as follows:

```shell
pip3 install autokeras
```

Please follow the [installation guide](https://autokeras.com/install) for more details.

**Note:** Currently, AutoKeras is only compatible with **Python >= 3.5** and **TensorFlow >= 2.1.0**.

## Community
<p float="left">
<a href="https://keras-slack-autojoin.herokuapp.com/"><img src="https://raw.githubusercontent.com/keras-team/autokeras/master/docs/templates/img/slack.svg" width="40px" hspace="10"></a>
<a href="https://twitter.com/autokeras"><img src="https://raw.githubusercontent.com/keras-team/autokeras/master/docs/templates/img/twitter.svg" width="40px" hspace="10"></a>
<a href="https://groups.google.com/forum/#!forum/autokeras-announce/join"><img src="https://raw.githubusercontent.com/keras-team/autokeras/master/docs/templates/img/gmail.svg" width="40px" hspace="10"></a>
<a href="https://groups.google.com/forum/#!forum/autokeras/join"><img src="https://raw.githubusercontent.com/keras-team/autokeras/master/docs/templates/img/calendar.svg" width="40px" hspace="10"></a>
</p>

**Slack**:
[Request an invitation](https://keras-slack-autojoin.herokuapp.com/).
Use the [#autokeras](https://app.slack.com/client/T0QKJHQRE/CSZ5MKZFU) channel for communication.

**Twitter**:
You can also follow us on Twitter [@autokeras](https://twitter.com/autokeras) for the latest news.

**Emails**:
Subscribe our [email list](https://groups.google.com/forum/#!forum/autokeras-announce/join) to receive announcements.

**Online Meetings**:
Join the [Google group](https://groups.google.com/forum/#!forum/autokeras/join) and our online meetings will appear on your Google Calendar.

## Contributing

You can follow the [Contributing Guide](https://autokeras.com/contributing/) to become a contributor.

If you don't know where to start, please join our community on [Slack](https://autokeras.com/#community) and ask us.
We will help you get started!

Thank all the contributors!

<a href="https://github.com/keras-team/autokeras/graphs/contributors"><img src="https://opencollective.com/autokeras/contributors.svg?avatarHeight=24&width=890&button=false" /></a>


## Backers

We accept financial support on [Open Collective](https://opencollective.com/autokeras).
Thank every backer for supporting us!

Organizations:
<a href="https://opencollective.com/autokeras#backers" target="_blank"><img src="https://opencollective.com/autokeras/sponsor.svg?avatarHeight=24&width=890&button=false"></a>

Individuals:
<a href="https://opencollective.com/autokeras#backers" target="_blank"><img src="https://opencollective.com/autokeras/backer.svg?avatarHeight=24&width=890&button=false"></a>

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

## DISCLAIMER

Please note that this is a **pre-release** version of the AutoKeras which is still undergoing final testing before its official release. The website, its software and all content found on it are provided on an
"as is" and "as available" basis. AutoKeras does **not** give any warranties, whether express or implied, as to the suitability or usability of the website, its software or any of its content. AutoKeras will **not** be liable for any loss, whether such loss is direct, indirect, special or consequential, suffered by any party as a result of their use of the libraries or content. Any usage of the libraries is done at the user's own risk and the user will be solely responsible for any damage to any computer system or loss of data that results from such activities. Should you encounter any bugs, glitches, lack of functionality or
other problems on the website, please let us know immediately so we
can rectify these accordingly. Your help in this regard is greatly
appreciated.

## Acknowledgements

The authors gratefully acknowledge the D3M program of the Defense Advanced Research Projects Agency (DARPA) administered through AFRL contract FA8750-17-2-0116; the Texas A&M College of Engineering, and Texas A&M.
