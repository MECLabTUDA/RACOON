# base image 
FROM python:3.7

LABEL IMAGE="tuda_dashboard"
LABEL VERSION="0.1.0"

ADD ./files /usr/src/app
WORKDIR /usr/src/app

COPY ./files/requirements.txt ./

# Get pip to download and install requirements:
RUN pip install --no-cache-dir -r requirements.txt

COPY files/ ./
RUN chmod +x boot.sh
#RUN python manage.py collectstatic

# Expose ports
EXPOSE 5001

# default command to execute    
# CMD exec gunicorn racoon.wsgi:application --bind 0.0.0.0:8000 --workers 3 

ENTRYPOINT ["./boot.sh"]
