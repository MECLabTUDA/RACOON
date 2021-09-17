#!/bin/bash
case "$DCMTK_COMMAND" in
    'json2dcm')
        echo "COMMAND: $DCMTK_COMMAND"
        echo $(ls)
        python3 -u json2dcm.py
    ;;
    'dcm2json')
        echo "COMMAND: $DCMTK_COMMAND"
        python3 -u dcm2json.py
    ;;
    *)
        echo $"Usage: $0 {json2dcm|dcm2json}"
        exit 1
esac
