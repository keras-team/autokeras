<img src="https://github.com/jhfjhfj1/autokeras/blob/master/logo.png?raw=true" alt="drawing" width="400px" style="display: block; margin-left: auto; margin-right: auto"/>

[![Build Status](https://travis-ci.org/jhfjhfj1/autokeras.svg?branch=master)](https://travis-ci.org/jhfjhfj1/autokeras)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/620bd322918c476aa33230ec911a4301)](https://www.codacy.com/app/jhfjhfj1/autokeras?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jhfjhfj1/autokeras&amp;utm_campaign=Badge_Grade)
[![Coverage Status](https://coveralls.io/repos/github/jhfjhfj1/autokeras/badge.svg)](https://coveralls.io/github/jhfjhfj1/autokeras)
<a href="https://badge.fury.io/py/autokeras"><img src="https://badge.fury.io/py/autokeras.svg" alt="PyPI version" style="width: 125px"></a>

Official Website: [autokeras.com](https://autokeras.com)

Auto-Keras is an open source software library for automated machine learning (AutoML).
It is developed by <a href="http://faculty.cs.tamu.edu/xiahu/index.html" target="_blank" rel="nofollow">DATA Lab</a> at Texas A&M University and community contributors.
The ultimate goal of AutoML is to provide easily accessible deep learning tools to domain experts with limited data science or machine learning background. 
Auto-Keras provides functions to automatically search for architecture and hyperparameters of deep learning models.

## Example

Here is a short example of using the package.


    import autokeras as ak

    clf = ak.ImageClassifier()
    clf.fit(x_train, y_train)
    results = clf.predict(x_test)

## Cite this work

Auto-Keras: Efficient Neural Architecture Search with Network Morphism.
Haifeng Jin, Qingquan Song, and Xia Hu.
[arXiv:1806.10282](https://arxiv.org/abs/1806.10282).

Biblatex entry:

    @online{jin2018efficient,
      author       = {Haifeng Jin and Qingquan Song and Xia Hu},
      title        = {Auto-Keras: Efficient Neural Architecture Search with Network Morphism},
      date         = {2018-06-27},
      year         = {2018},
      eprintclass  = {cs.LG},
      eprinttype   = {arXiv},
      eprint       = {cs.LG/1806.10282},
    }

## Community

You can use Gitter to communicate with people who also interested in Auto-Keras.
<a href="https://gitter.im/autokeras/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge"><img src="https://badges.gitter.im/autokeras/Lobby.svg" alt="Join the chat at https://gitter.im/autokeras/Lobby" style="width: 92px"></a>

You can also follow us on Twitter [@autokeras](https://twitter.com/autokeras) for the latest news.

## Contributing Code

You can follow the [Contributing Guide](https://autokeras.com/temp/contribute/) for details.
The easist way to contribute is to resolve the issues with the "[call for contributors](https://github.com/jhfjhfj1/autokeras/labels/call%20for%20contributors)" tag.
They are friendly to beginners.
 
## Support Auto-Keras

We accept donations on [Open Collective](https://opencollective.com/autokeras).
Thank every backer for supporting us!

<a href="https://opencollective.com/autokeras/donate" target="_blank">
  <img src="https://opencollective.com/autokeras/donate/button@2x.png?color=blue" width=200 />
</a>


## DISCLAIMER

Please note that this is a **pre-release** version of the Auto-Keras which is still undergoing final testing before its official release. The website, its software and all content found on it are provided on an
“as is” and “as available” basis. Auto-Keras does **not** give any warranties, whether express or implied, as to the suitability or usability of the website, its software or any of its content. Auto-Keras will **not** be liable for any loss, whether such loss is direct, indirect, special or consequential, suffered by any party as a result of their use of the libraries or content. Any usage of the libraries is done at the user’s own risk and the user will be solely responsible for any damage to any computer system or loss of data that results from such activities. Should you encounter any bugs, glitches, lack of functionality or
other problems on the website, please let us know immediately so we
can rectify these accordingly. Your help in this regard is greatly
appreciated.

## Acknowledgements

The authors gratefully acknowledge the D3M program of the Defense Advanced Research Projects Agency (DARPA) administered through AFRL contract FA8750-17-2-0116; the Texas A&M College of Engineering, and Texas A&M. 
