FROM ubuntu:18.04

RUN mkdir /app

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.6 \
        curl \
        ca-certificates \
        build-essential \
        python3.6-dev \
        python3-distutils \
        git \
        gcc \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && \
    rm get-pip.py

RUN git clone https://github.com/keras-team/autokeras.git

RUN python3.6 -m pip install -U keras

RUN cd autokeras && python3.6 -m pip install -U . && cd ..

RUN rm -rf autokeras

CMD /bin/bash
