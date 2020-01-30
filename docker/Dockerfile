ARG TF_VERSION=2.1.0
FROM tensorflow/tensorflow:${TF_VERSION}-gpu-py3-jupyter

RUN pip install autokeras

CMD jupyter notebook --port=8888 --ip=0.0.0.0
