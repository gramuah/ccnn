#! /bin/bash


export PYTHONUNBUFFERED="True"

# Parameters
GPU_DEV=0
CONFIG_FILE=models/trancos/ccnn/ccnn_trancos_cfg.yml
CAFFE_MODEL=models/pretrained_models/trancos/ccnn/trancos_ccnn.caffemodel
DEPLOY=models/trancos/ccnn/ccnn_deploy.prototxt

LOG="experiments/logs/trancos_ccnn_`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# Time the task
T="$(date +%s)"

# Test Net
python src/test.py --dev ${GPU_DEV} --prototxt ${DEPLOY} --caffemodel ${CAFFE_MODEL} --cfg ${CONFIG_FILE}

T="$(($(date +%s)-T))"
echo "Time in seconds: ${T}"
