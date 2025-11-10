## Requirements

**Python 3**: Follow the TensorFlow install steps to install Python 3.

**Pip**: Follow the TensorFlow install steps to install Pip.

**PyTorch >= 2.3.0**: AutoKeras is based on Keras. We recommend using the
*PyTorch backend
Please follow [this page](https://pytorch.org/get-started/locally/) to install
PyTorch.

## Install AutoKeras
AutoKeras only support **Python 3**.
If you followed previous steps to use virtualenv to install tensorflow,
you can just activate the virtualenv and use the following command to install AutoKeras. 
```
pip install git+https://github.com/keras-team/keras-tuner.git
pip install autokeras
```

If you did not use virtualenv, and you use `python3` command to execute your python program,
please use the following command to install AutoKeras.
```
python3 -m pip install git+https://github.com/keras-team/keras-tuner.git
python3 -m pip install autokeras
```

