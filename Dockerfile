FROM rootproject/root:6.22.02-ubuntu20.04

USER root

WORKDIR /opt

COPY packages /opt/packages

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get -y install --no-install-recommends -y $(cat /opt/packages) \
    && rm -rf /var/lib/apt/lists/* \
    && echo /opt/root/lib >> /etc/ld.so.conf \
    && ldconfig \
    && apt-get clean

ARG USER=foo
RUN useradd -ms /bin/bash $USER 

COPY requirements.txt /opt/requirements.txt

USER $USER
ENV PATH /usr/local/bin:/home/$USER/.local/bin:$PATH 
WORKDIR /home/$USER

RUN python3 -m pip install --upgrade pip --user \
    && python3 -m pip install --upgrade setuptools wheel --user \
    && python3 -m pip install -r /opt/requirements.txt --user \
    && python3 -m pip cache purge 

CMD ["bash"]
