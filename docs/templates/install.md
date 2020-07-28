## Requirements

**Python 3**: Follow the TensorFlow install steps to install Python 3.

**Pip**: Follow the TensorFlow install steps to install Pip.

**Tensorflow >= 2.3.0**: AutoKeras is based on TensorFlow.
Please follow
[this tutorial](https://www.tensorflow.org/install/pip) to install TensorFlow for python3.

**GPU Setup (Optional)**:
If you have GPUs on your machine and want to use them to accelerate the training,
you can follow [this tutorial](https://www.tensorflow.org/install/gpu) to setup.

## Install AutoKeras
AutoKeras only support **Python 3**.
If you followed previous steps to use virtualenv to install tensorflow,
you can just activate the virtualenv and use the following command to install AutoKeras. 
```
pip install git+https://github.com/keras-team/keras-tuner.git@1.0.2rc1
pip install autokeras==1.0.5
```

If you did not use virtualenv, and you use `python3` command to execute your python program,
please use the following command to install AutoKeras.
```
python3 -m pip install git+https://github.com/keras-team/keras-tuner.git@1.0.2rc1
python3 -m pip install autokeras==1.0.5
```

