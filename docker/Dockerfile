FROM tensorflow/tensorflow:2.2.0

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN pip install git+https://github.com/keras-team/keras-tuner.git@1.0.2rc0
RUN pip install autokeras
