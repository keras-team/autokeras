<img src="https://autokeras.com/img/row_red.svg" alt="drawing" width="400px" style="display: block; margin-left: auto; margin-right: auto"/>

[![Build Status](https://travis-ci.org/keras-team/autokeras.svg?branch=master)](https://travis-ci.org/keras-team/autokeras)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/620bd322918c476aa33230ec911a4301)](https://www.codacy.com/app/jhfjhfj1/autokeras?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=keras-team/autokeras&amp;utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/haifeng-jin/autokeras/branch/master/graph/badge.svg)](https://codecov.io/gh/haifeng-jin/autokeras)
[![PyPI version](https://badge.fury.io/py/autokeras.svg)](https://badge.fury.io/py/autokeras")


Official Website: [autokeras.com](https://autokeras.com)

##

AutoKeras is an open source software library for automated machine learning (AutoML).
It is developed by <a href="http://faculty.cs.tamu.edu/xiahu/index.html" target="_blank" rel="nofollow">DATA Lab</a> at Texas A&M University and community contributors.
The ultimate goal of AutoML is to provide easily accessible deep learning tools to domain experts with limited data science or machine learning background.
AutoKeras provides functions to automatically search for architecture and hyperparameters of deep learning models.

# AutoKeras 1.0 is coming soon!

## Installation

To install the package, please use the `pip` installation as follows:

```shell
pip3 install autokeras # for 0.4 version
pip3 install autokeras==1.0.0b0 # for 1.0 version
```

**Note:** currently, AutoKeras is only compatible with: **Python 3**.

## Example

Here is a short example of using the package.

```
import autokeras as ak

clf = ak.ImageClassifier()
clf.fit(x_train, y_train)
results = clf.predict(x_test)
```

For detailed tutorial, please check [here](https://autokeras.com/tutorial/overview/).

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

## Community

You can use Gitter to communicate with people who are also interested in AutoKeras.
<a href="https://gitter.im/autokeras/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge"><img src="https://badges.gitter.im/autokeras/Lobby.svg" alt="Join the chat at https://gitter.im/autokeras/Lobby" style="width: 92px"></a>

You can also follow us on Twitter [@autokeras](https://twitter.com/autokeras) for the latest news.

## Contributing Code

You can follow the [Contributing Guide](https://autokeras.com/contributing/) for details.
The easist way to contribute is to resolve the issues with the "[call for contributors](https://github.com/keras-team/autokeras/labels/call%20for%20contributors)" tag.
They are friendly to beginners.

## Support AutoKeras

We accept donations on [Open Collective](https://opencollective.com/autokeras).
Thank every backer for supporting us!

<a href="https://opencollective.com/autokeras/donate" target="_blank">
  <img src="https://opencollective.com/autokeras/donate/button@2x.png?color=blue" width=200 />
</a>


## DISCLAIMER

Please note that this is a **pre-release** version of the AutoKeras which is still undergoing final testing before its official release. The website, its software and all content found on it are provided on an
"as is" and "as available" basis. AutoKeras does **not** give any warranties, whether express or implied, as to the suitability or usability of the website, its software or any of its content. AutoKeras will **not** be liable for any loss, whether such loss is direct, indirect, special or consequential, suffered by any party as a result of their use of the libraries or content. Any usage of the libraries is done at the user's own risk and the user will be solely responsible for any damage to any computer system or loss of data that results from such activities. Should you encounter any bugs, glitches, lack of functionality or
other problems on the website, please let us know immediately so we
can rectify these accordingly. Your help in this regard is greatly
appreciated.

## Acknowledgements

The authors gratefully acknowledge the D3M program of the Defense Advanced Research Projects Agency (DARPA) administered through AFRL contract FA8750-17-2-0116; the Texas A&M College of Engineering, and Texas A&M.
