help:
	@cat Makefile

DATA?="${HOME}/Data"
GPUS?=all
DOCKER_FILE?=Dockerfile
DOCKER=docker
TF_VERSION=2.3.0
TEST=tests/
SRC?=$(shell dirname `pwd`)

build:
	docker build -t autokeras --build-arg TF_VERSION=$(TF_VERSION) -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run --gpus $(GPUS) -it -v $(SRC):/src/workspace -v $(DATA):/data autokeras bash

ipython: build
	$(DOCKER) run --gpus $(GPUS) -it -v $(SRC):/src/workspace -v $(DATA):/data --env  autokeras ipython

notebook: build
	$(DOCKER) run --gpus $(GPUS) -it -v $(SRC):/src/workspace -v $(DATA):/data --net=host --env autokeras

test: build
	$(DOCKER) run --gpus $(GPUS) -it -v $(SRC):/src/workspace -v $(DATA):/data --env  autokeras py.test $(TEST)

