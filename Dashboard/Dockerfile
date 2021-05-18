# base image 
FROM python:3.7

# File Author / Maintainer
MAINTAINER TUDA

ADD ./files /usr/src/app
WORKDIR /usr/src/app

COPY ./files/requirements.txt ./

# Get pip to download and install requirements:
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn==19.6.0

# Expose ports
EXPOSE 8000

# default command to execute    
CMD exec gunicorn racoon.wsgi:application --bind 0.0.0.0:8000 --workers 3 