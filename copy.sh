#!/bin/sh

KAAPANA_DIR=$1 # e.g. /home/ksquare/repositories/kaapana

echo Copying to $KAAPANA_DIR
DASHBOARD_DIR=$KAAPANA_DIR/services/applications/dashboard
mkdir -p $DASHBOARD_DIR

#cp -r JIP/dashboard/* $DASHBOARD_DIR
rsync --exclude dashboard-chart-0.1.0.tgz --exclude venv --exclude docker/files/static -av JIP/dashboard/ $DASHBOARD_DIR
#cp -r JIP/workflows/* $KAAPANA_DIR/workflows
rsync --exclude processing-container/qm-artifacts/files/test/test_obj/ --exclude processing-container/qm-dicePredictor/files/test/test_obj/ -av JIP/workflows/* $KAAPANA_DIR/workflows
