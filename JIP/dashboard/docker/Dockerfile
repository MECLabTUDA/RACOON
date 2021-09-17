# base image 
FROM python:3.7

LABEL IMAGE="tuda-dashboard"
LABEL VERSION="0.1.0"

WORKDIR /usr/src/app

COPY ./files/requirements.txt ./

# Get pip to download and install requirements:
RUN pip install --no-cache-dir -r requirements.txt

# ADD ./files /usr/src/app
COPY files/ ./
RUN chmod +x boot.sh
#RUN python manage.py collectstatic

# Expose ports
EXPOSE 5001

ENV DJANGO_SUPERUSER_USERNAME=kaapana
ENV DJANGO_SUPERUSER_PASSWORD=admin
ENV DJANGO_SUPERUSER_EMAIL=kaapana@gmx.de

ENV ADMIN_USERNAME=kaapana
ENV ADMIN_PASSWORD=admin
ENV ADMIN_EMAIL=kaapana@gmx.de

# default command to execute    
# CMD exec gunicorn racoon.wsgi:application --bind 0.0.0.0:8000 --workers 3 

ENTRYPOINT ["./boot.sh"]
