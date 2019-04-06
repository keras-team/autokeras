FROM python:3.6.7-slim-jessie

RUN mkdir /app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        build-essential \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install keras

RUN python -m pip install numpy==1.15.4

RUN python -m pip install autokeras

