FROM nvcr.io/nvidia/pytorch:20.09-py3

LABEL IMAGE="qm"
LABEL VERSION="v1.0.0"
LABEL CI_IGNORE="False"

ARG DEBIAN_FRONTEND=noninteractive

# set work directory
WORKDIR /qm

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y git-all
RUN python3 -m pip install -U --force-reinstall pip

COPY files/ .
RUN chmod +x start.sh

RUN pip3 install --upgrade setuptools
RUN pip3 install -r requirements.txt
RUN pip3 install git+https://github.com/amrane99/lungmask

ENV MODELDIR "/models"

ENTRYPOINT ["/bin/bash", "./start.sh"]
