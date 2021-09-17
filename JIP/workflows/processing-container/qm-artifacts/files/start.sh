#!/bin/bash
echo "PREPROCESSING"
python3 JIP.py --mode preprocess --device 0 --datatype inference
echo "INFERENCE"
python3 JIP.py --mode inference --device 0
exit 0
