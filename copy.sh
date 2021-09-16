#!/bin/sh

KAAPANA_DIR=$1 # e.g. /home/ksquare/repositories/kaapana

echo Copying to $KAAPANA_DIR
DASHBOARD_DIR=$KAAPANA_DIR/services/applications/dashboard
mkdir -p $DASHBOARD_DIR
cp -r JIP/dashboard/* $DASHBOARD_DIR
cp -r JIP/workflows/* $KAAPANA_DIR/workflows