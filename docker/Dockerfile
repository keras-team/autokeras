FROM tensorflow/tensorflow:2.3.0
RUN python -m pip install --no-cache-dir https://github.com/keras-team/keras-tuner/tarball/1.0.2rc4
WORKDIR /opt/autokeras
COPY . .
RUN python -m pip install --no-cache-dir --editable .
WORKDIR /work
