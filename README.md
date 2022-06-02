# automl-docker

based on rootproject/root:6.22.02-ubuntu20.04

docker and script to test some automl packages for binary classification 

## Installation:

`cd automl-docker`
`docker build -t my-ml -f Dockerfile .`

## Usage 

`docker run --rm -it --shm-size=3.00gb -v $PWD/workdir:/home/foo/workdir --user $(id -u) my-ml` 
`foo:~$ cd workdir/`
`foo:~$ python3 -i etahml.py`

