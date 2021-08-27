#!/bin/sh

#SCRIPT_NAME=$APPLICATION_ROOT gunicorn -b :5000 --access-logfile - --error-logfile - run:app
#python manage.py collectstatic  --noinput
#SCRIPT_NAME=$APPLICATION_ROOT gunicorn -b :5000 --access-logfile - --error-logfile - racoon.wsgi
gunicorn -b :5001 --access-logfile - --error-logfile - racoon.wsgi
