FROM nvcr.io/nvidia/pytorch:20.09-py3

LABEL IMAGE="tuda-qm-dice"
LABEL VERSION="0.1.0"
LABEL CI_IGNORE="False"

ARG DEBIAN_FRONTEND=noninteractive

# set work directory
WORKDIR /qm

RUN apt-get update && apt-get install -y python3 python3-pip git
RUN python3 -m pip install -U --force-reinstall pip

COPY files/ .
RUN chmod +x start.sh

RUN pip3 install --upgrade setuptools
RUN pip3 install -r requirements.txt
RUN pip3 install git+https://github.com/amrane99/lungmask
RUN curl https://www.kaapana.ai/kaapana-downloads/secret-7583084/qm-dicePredictor-models.zip -o test-models.zip && unzip test-models.zip && rm test-models.zip && mv JIP/workflows/processing-container/qm-dicePredictor/files/test/test_obj test/test_obj
RUN python3 -c "from lungmask import mask; mask.get_model('unet', 'R231')"

ENV MODELDIR "/models"

ENTRYPOINT ["/bin/bash", "./start.sh"]
