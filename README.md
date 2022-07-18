# automl-docker

based on rootproject/root:6.22.02-ubuntu20.04

docker and script to test some automl packages for binary classification 

## Installation:

`cd automl-docker`

`ln -s requirements_ag.txt requirements.txt` (or, alternaively, requirements_askl.txt)

`docker build -t my-ml -f Dockerfile .`

## Usage 

`docker run --rm -it --shm-size=3.00gb -v $PWD/workdir:/home/foo/workdir --user $(id -u) my-ml`


### Train the model

`foo:~$ python3 -m pip freeze`

`foo:~$ cd /home/foo/workdir/`

`foo:~$ python3 -i etahml.py`

### Use the model in a C++ program

`foo:~$ mkdir build`

`foo:~$ cd build`

`foo:~$ cmake ..`

`foo:~$ make`

### After placing the input .root file in build folder, copy model and run the program: 

`foo:~$ cp ../etaPi_BDT_clf.joblib . `

`foo:~$ ./example `





