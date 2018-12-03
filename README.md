<img src="https://github.com/jhfjhfj1/autokeras/blob/master/logo.png?raw=true" alt="drawing" width="400px" style="display: block; margin-left: auto; margin-right: auto"/>

<div style="text-align: center">
<p>
<a href="https://travis-ci.org/jhfjhfj1/autokeras"><img alt="Build Status" src="https://travis-ci.org/jhfjhfj1/autokeras.svg?branch=master" style="width: 90px"/></a>
<a href="https://coveralls.io/github/jhfjhfj1/autokeras?branch=master"><img alt="Coverage Status" src="https://coveralls.io/repos/github/jhfjhfj1/autokeras/badge.svg?branch=master" style="width: 99px"/></a>
<a href="https://badge.fury.io/py/autokeras"><img src="https://badge.fury.io/py/autokeras.svg" alt="PyPI version" style="width: 125px"></a>
<a href="https://autokeras.com"><img src="https://readthedocs.org/projects/pip/badge/?version=latest&style=flat" alt="AutoKeras Official Website" style="width: 86px"></a>
<a href="#backers"><img src="https://opencollective.com/autokeras/backers/badge.svg" alt="Backers on Open Collective"></a>
<a href="#sponsors"><img src="https://opencollective.com/autokeras/sponsors/badge.svg" alt="Sponsors on Open Collective"></a>
</p>
</div>

Auto-Keras is an open source software library for automated machine learning (AutoML).
It is developed by <a href="http://faculty.cs.tamu.edu/xiahu/index.html" target="_blank" rel="nofollow">DATA Lab</a> at Texas A&M University and community contributors.
The ultimate goal of AutoML is to provide easily accessible deep learning tools to domain experts with limited data science or machine learning background. 
Auto-Keras provides functions to automatically search for architecture and hyperparameters of deep learning models.

## Installation


To install the package, please use the `pip` installation as follows:

    pip install autokeras
    
**Note:** currently, Auto-Keras is only compatible with: **Python 3.6**.

## Example

Here is a short example of using the package.


    import autokeras as ak

    clf = ak.ImageClassifier()
    clf.fit(x_train, y_train)
    results = clf.predict(x_test)

## Community

You can use Gitter to communicate with people who also interested in Auto-Keras.

<a href="https://gitter.im/autokeras/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge"><img src="https://badges.gitter.im/autokeras/Lobby.svg" alt="Join the chat at https://gitter.im/autokeras/Lobby" style="width: 92px"></a>

## Citing this work

If you use Auto-Keras in a scientific publication, you are highly encouraged (though not required) to cite the following paper:

Efficient Neural Architecture Search with Network Morphism.
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
 
## Support Auto-Keras

We accept donations on [Open Collective](https://opencollective.com/autokeras).
The money will be used to motivate the developers in the open-source community to contribute code to Auto-Keras.
Thank every backer for supporting us!

<a href="https://opencollective.com/autokeras/donate" target="_blank">
  <img src="https://opencollective.com/autokeras/donate/button@2x.png?color=blue" width=300 />
</a>


## DISCLAIMER

Please note that this is a **pre-release** version of the Auto-Keras which is still undergoing final testing before its official release. The website, its software and all content found on it are provided on an
“as is” and “as available” basis. Auto-Keras does **not** give any warranties, whether express or implied, as to the suitability or usability of the website, its software or any of its content. Auto-Keras will **not** be liable for any loss, whether such loss is direct, indirect, special or consequential, suffered by any party as a result of their use of the libraries or content. Any usage of the libraries is done at the user’s own risk and the user will be solely responsible for any damage to any computer system or loss of data that results from such activities. Should you encounter any bugs, glitches, lack of functionality or
other problems on the website, please let us know immediately so we
can rectify these accordingly. Your help in this regard is greatly
appreciated.

## Acknowledgements

The authors gratefully acknowledge the D3M program of the Defense Advanced Research Projects Agency (DARPA) administered through AFRL contract FA8750-17-2-0116; the Texas A&M College of Engineering, and Texas A&M. 

## Contributors

This project exists thanks to all the people who contribute. [[Contribute](CONTRIBUTING.md)].
<a href="https://github.com/jhfjhfj1/autokeras/graphs/contributors"><img src="https://opencollective.com/autokeras/contributors.svg?width=890&button=false" /></a>


## Backers

Thank you to all our backers! 🙏 [[Become a backer](https://opencollective.com/autokeras#backer)]

<a href="https://opencollective.com/autokeras#backers" target="_blank"><img src="https://opencollective.com/autokeras/backers.svg?width=890"></a>


## Sponsors

Support this project by becoming a sponsor. Your logo will show up here with a link to your website. [[Become a sponsor](https://opencollective.com/autokeras#sponsor)]

<a href="https://opencollective.com/autokeras/sponsor/0/website" target="_blank"><img src="https://opencollective.com/autokeras/sponsor/0/avatar.svg"></a>
<a href="https://opencollective.com/autokeras/sponsor/1/website" target="_blank"><img src="https://opencollective.com/autokeras/sponsor/1/avatar.svg"></a>
<a href="https://opencollective.com/autokeras/sponsor/2/website" target="_blank"><img src="https://opencollective.com/autokeras/sponsor/2/avatar.svg"></a>
<a href="https://opencollective.com/autokeras/sponsor/3/website" target="_blank"><img src="https://opencollective.com/autokeras/sponsor/3/avatar.svg"></a>
<a href="https://opencollective.com/autokeras/sponsor/4/website" target="_blank"><img src="https://opencollective.com/autokeras/sponsor/4/avatar.svg"></a>
<a href="https://opencollective.com/autokeras/sponsor/5/website" target="_blank"><img src="https://opencollective.com/autokeras/sponsor/5/avatar.svg"></a>
<a href="https://opencollective.com/autokeras/sponsor/6/website" target="_blank"><img src="https://opencollective.com/autokeras/sponsor/6/avatar.svg"></a>
<a href="https://opencollective.com/autokeras/sponsor/7/website" target="_blank"><img src="https://opencollective.com/autokeras/sponsor/7/avatar.svg"></a>
<a href="https://opencollective.com/autokeras/sponsor/8/website" target="_blank"><img src="https://opencollective.com/autokeras/sponsor/8/avatar.svg"></a>
<a href="https://opencollective.com/autokeras/sponsor/9/website" target="_blank"><img src="https://opencollective.com/autokeras/sponsor/9/avatar.svg"></a>


