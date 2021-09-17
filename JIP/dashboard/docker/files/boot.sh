#!/bin/bash

#SCRIPT_NAME=$APPLICATION_ROOT gunicorn -b :5000 --access-logfile - --error-logfile - run:app
#python manage.py collectstatic  --noinput
echo "Initializing database"
python manage.py migrate
python manage.py createsuperuser --noinput

echo "Creating admin"
if [[ -n "${ADMIN_USERNAME}" ]] && [[ -n "${ADMIN_PASSWORD}" ]] && [[ -n "${ADMIN_EMAIL}" ]]; then
  # todo create_admin unknown command!
  python manage.py create_admin \
    --username "${ADMIN_USERNAME}" \
    --password "${ADMIN_PASSWORD}" \
    --email "${ADMIN_EMAIL}" \
    --noinput \
  || true
fi

#SCRIPT_NAME=$APPLICATION_ROOT gunicorn -b :5000 --access-logfile - --error-logfile - racoon.wsgi
gunicorn -b :5001 --access-logfile - --error-logfile - racoon.wsgi
