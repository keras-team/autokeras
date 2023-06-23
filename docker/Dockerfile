FROM tensorflow/tensorflow:2.13.0rc2
WORKDIR /opt/autokeras
COPY . .
RUN python -m pip install --no-cache-dir --editable .
WORKDIR /work
