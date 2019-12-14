help:
	@cat Makefile

DATA?="${HOME}/Data"
GPUS?=all
DOCKER_FILE?=Dockerfile
DOCKER=docker
PYTHON_VERSION?=3.6
CUDA_VERSION?=10.1
CUDNN_VERSION?=7
TEST=tests/
SRC?=$(shell dirname `pwd`)

build:
	docker build -t autokeras --build-arg python_version=$(PYTHON_VERSION) --build-arg cuda_version=$(CUDA_VERSION) --build-arg cudnn_version=$(CUDNN_VERSION) -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run --gpus $(GPUS) -it -v $(SRC):/src/workspace -v $(DATA):/data autokeras bash

ipython: build
	$(DOCKER) run --gpus $(GPUS) -it -v $(SRC):/src/workspace -v $(DATA):/data --env  autokeras ipython

notebook: build
	$(DOCKER) run --gpus $(GPUS) -it -v $(SRC):/src/workspace -v $(DATA):/data --net=host --env autokeras

test: build
	$(DOCKER) run --gpus $(GPUS) -it -v $(SRC):/src/workspace -v $(DATA):/data --env  autokeras py.test $(TEST)

