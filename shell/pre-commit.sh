#!/usr/bin/env bash

set -e

export DOCKER_BUILDKIT=1
docker build -t autokeras_formatting -f docker/pre-commit.Dockerfile .
docker run --rm -it -v "$(pwd -P):/autokeras" autokeras_formatting
