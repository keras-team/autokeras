FROM tensorflow/tensorflow:2.12.0rc1
WORKDIR /opt/autokeras
COPY . .
RUN python -m pip install --no-cache-dir --editable .
WORKDIR /work
