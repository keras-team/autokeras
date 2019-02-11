# nasbot

A Python implementation of NASBOT (Neural Architecture Search with Bayesian Optimisation
and Optimal Transport).
This repo also provides OTMANN (Optimal Transport Metric for Architectures of Neural
Networks), which is an optimal transport based distance for neural network architectures.
For more details, please see our paper below.

For questions and bug reports please email kandasamy@cs.cmu.edu.

### Installation

* Download the package.
```bash
$ git clone https://github.com/kirthevasank/nasbot.git
```

* Install the following packages packages via pip: cython, POT (Python Optimal Transport),
graphviz and pygraphviz. graphviz and pygraphviz are only needed to visualise the networks
and are not necessary to run nasbot. However, some unit tests may fail.
```bash
$ pip install cython POT graphviz pygraphviz
```
  In addition to the above, you will need numpy and scipy which can also be pip installed.

* Now set `HOME_PATH` in the set_up file to the parent directory of nasbot, i.e.
`HOME_PATH=<path/to/parent/directory>/nasbot`. Then source the set up file.
```bash
$ source set_up
```

* Next, you need to build the direct fortran library. For this `cd` into
[`utils/direct_fortran`](https://github.com/kirthevasank/nasbot/blob/master/utils/direct_fortran)
and run `bash make_direct.sh`. You will need a fortran compiler such as gnu95.
Once this is done, you can run `python simple_direct_test.py` to make sure that it was
installed correctly.
The default version of NASBOT can be run without direct, but some unit tests might fail.

* Finally, you need to install tensorflow to execute the MLP/CNN demos on GPUs.
```bash
$ pip install tensorflow-gpu
```

**Testing the Installation**:
To test the installation, run ```bash run_all_tests.sh```. Some of the tests are
probabilistic and could fail at times. If this happens, run the same test several times
and make sure it is not consistently failing. Running all tests will take a while.
You can run each unit test individually simpy via `python unittest_xxx.py`.

### Getting started

To help get started, we have some demos in the `demos` directory.
They demonstrate how to specify a) a search space, 
b) the function to be optimised, and c) the number of parallel workers for NASBOT - these
are the bare minimum that need to specified.
Other parameters can be tuned via the APIs available in
[`opt/nasbot.py`](https://github.com/kirthevasank/nasbot/blob/master/opt/nasbot.py).

[`demos/demo_synthetic.py`](https://github.com/kirthevasank/nasbot/blob/master/demos/demo_synthetic.py)
demonstrates NASBOT on a synthetic function while
[`demos/demo_mlp.py`](https://github.com/kirthevasank/nasbot/blob/master/demos/demo_mlp.py)
and
[`demos/demo_cnn.py`](https://github.com/kirthevasank/nasbot/blob/master/demos/demo_cnn.py)
demonstrate NASBOT on MLP and CNN hyper-parameter tuning tasks respectively.
The datasets can be downloaded from the
[author's homepage](http://www.cs.cmu.edu/~kkandasa/research.html).
There are some instructions in the demo files on preparing/saving the data files.
To run the MLP/CNN demos, you will need access to one or more GPUs and install
tensorflow, e.g. `pip install tensorflow-gpu`.


### Using NASBOT in your architecture search task

To use NASBOT in your architecture search task, you need to write a FunctionCaller class
which inherits the `NNFunctionCaller` class in 
[`opt/nn_function_caller.py`](https://github.com/kirthevasank/nasbot/blob/master/opt/nn_function_caller.py).
You should implement the method `_eval_validation_score(nn, qinfo)` in this class which
evaluates a network `nn` and returns the validation score.
NASBOT will *maximise* this score - hence, (for example) you can return the accuracy
for classification problems and the negative MSE for regression problems.
The `qinfo` argument is used to pass other ancillary information that may be required to
conduct the evaluation (e.g. GPU ID).

You can follow the examples in 
[`demos/mlp_function_caller.py`](https://github.com/kirthevasank/nasbot/blob/master/demos/mlp_function_caller.py)
and [`demos/cnn_function_caller.py`](https://github.com/kirthevasank/nasbot/blob/master/demos/cnn_function_caller.py)


### OTMANN
- The OTMANN distance is implemented in
[`nn/nn_comparators.py`](https://github.com/kirthevasank/nasbot/blob/master/nn/nn_comparators.py)
in the class `OTMANNDistanceComputer`.
- The function `get_default_otmann_distance` will return an object which can be used to
  evaluate the OTMANN distance with default parameters.
- You can obtain a customised distance via the function `get_otmann_distance_from_args`.


### Some Details
- A neural network is represented as a graph (see paper below) in
[`nn/neural_network.py`](https://github.com/kirthevasank/nasbot/blob/master/nn/neural_network.py).
- The [`cg`](https://github.com/kirthevasank/nasbot/blob/master/nn/cg)
  directory converts this graphical representation into a tensorflow
  implementation. We have not yet (and do not have immediate plans to)
  implemented other frameworks (e.g. PyTorch, Keras). However, if you have implemented it
  and are willing to share it, we would happy to include a reference here and/or
  incorporate it as part of this repo.
- If you want to change any part of the method, the unit tests in each directory provide
  a decent guide on running/modifying various components.
- We have tested this repo on Linux and Mac on Python 2.
  We are in the process of making this Python 3 compatible.
  We have not tested on Windows.


### Citation
If you use any part of this code in your work, please cite our
[Arxiv paper](https://arxiv.org/pdf/1802.07191.pdf):

```bibtex
@article{kandasamy2018neural,
  title={Neural Architecture Search with Bayesian Optimisation and Optimal Transport},
  author={Kandasamy, Kirthevasan and Neiswanger, Willie and Schneider, Jeff and Poczos,
Barnabas and Xing, Eric},
  journal={arXiv preprint arXiv:1802.07191},
  year={2018}
}
```


### License
This software is released under the MIT license. For more details, please refer
[LICENSE.txt](https://github.com/kirthevasank/nasbot/blob/master/LICENSE.txt).

"Copyright 2018 Kirthevasan Kandasamy"

- For questions and bug reports please email kandasamy@cs.cmu.edu

