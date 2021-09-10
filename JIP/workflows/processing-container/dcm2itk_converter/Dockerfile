FROM python:3.8

LABEL IMAGE="tuda-dcm-converter"
LABEL VERSION="0.1.0"
LABEL CI_IGNORE="False"


# set work directory
WORKDIR /dcmqi
RUN pip3 install pydicom==2.0.0
RUN pip3 install pydicom-seg
RUN pip3 install dicom2nifti
RUN pip3 install SimpleITK

#Release 1.0.4
RUN curl -L -O https://github.com/QIICR/dcmqi/releases/download/v1.2.2/dcmqi-1.2.2-linux.tar.gz
RUN tar xzvf dcmqi-1.2.2-linux.tar.gz


RUN apt-get update
RUN apt-get install plastimatch -y

COPY files/dcmConvert.py /

CMD ["python3","-u","/dcmConvert.py"]
