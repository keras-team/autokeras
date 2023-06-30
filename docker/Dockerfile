FROM tensorflow/tensorflow:2.3.0
WORKDIR /opt/autokeras
COPY . .
RUN python -m pip install --no-cache-dir --editable .
WORKDIR /work
