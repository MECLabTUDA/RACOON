FROM ubuntu:20.10

LABEL IMAGE="tuda-dcmtools"
LABEL VERSION="0.1.0"
LABEL CI_IGNORE="False"


# set work directory
WORKDIR /dcmtk

RUN apt-get update
RUN apt-get install -y libicu-dev
RUN apt-get install -y libxml2
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y wget
RUN apt-get install -y dcmtk

COPY files/json2dcm.py .
COPY files/dcm2json.py .
COPY files/convert.sh .
COPY files/requirements.txt .
COPY files/template.xml templates/template.xml
RUN chmod +x convert.sh

RUN pip3 install -r requirements.txt


ENTRYPOINT ["/bin/bash", "./convert.sh"]
