#!/bin/sh

MODEL_DOWNLOAD_DIR=$1 # e.g. $HOME/E230-kaapana-downloads/secret-246949

echo "Creating zips and copyining them to $MODEL_DOWNLOAD_DIR"
mkdir -p $MODEL_DOWNLOAD_DIR

zip -r $MODEL_DOWNLOAD_DIR/qm-dicePredictor-models.zip ./JIP/workflows/processing-container/qm-dicePredictor/files/test/test_obj/
zip -r $MODEL_DOWNLOAD_DIR/qm-artifacts-models.zip ./JIP/workflows/processing-container/qm-artifacts/files/test/test_obj/
