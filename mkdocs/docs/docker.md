# Auto-Keras Docker

## Download Auto-Keras Docker image


The following command download Auto-Keras docker image to your machine.  

```
docker pull garawalid/autokeras
```

## Start Auto-Keras Docker container

```
docker run -it --shm-size 2G garawalid/autokeras /bin/bash
```

In case you need more memory to run the container, change the value of `shm-size`. ([Docker run reference](https://docs.docker.com/engine/reference/run/#general-form))


## Run application :


To run a local script `file.py` using Auto-Keras within the container, mount the host directory `-v hostDir:/app`.

```
docker run -it -v hostDir:/app --shm-size 2G garawalid/autokeras python file.py
```

## Example :

Let's download the mnist example and run it within the container.  

Download the example :  
```
wget https://raw.githubusercontent.com/jhfjhfj1/autokeras/master/examples/a_simple_example/mnist.py
```

Run the mnist example :
```
docker run -it -v "$(pwd)":/app --shm-size 2G garawalid/autokeras python mnist.py
```
