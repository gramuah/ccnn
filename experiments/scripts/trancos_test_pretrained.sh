#! /bin/bash

export PYTHONUNBUFFERED="True"

# Parameters, uncomment one of the following set of parameters
GPU_DEV=0
# CCNN
CONFIG_FILE=models/trancos/ccnn/ccnn_trancos_cfg.yml
CAFFE_MODEL=models/pretrained_models/trancos/ccnn/trancos_ccnn.caffemodel
DEPLOY=models/trancos/ccnn/ccnn_deploy.prototxt

# HYDRA 2s
#CONFIG_FILE=models/trancos/hydra2/hydra2_trancos_cfg.yml
#CAFFE_MODEL=models/pretrained_models/trancos/hydra2/trancos_hydra2.caffemodel
#DEPLOY=models/trancos/hydra2/hydra2_deploy.prototxt

# HYDRA 3s
#CONFIG_FILE=models/trancos/hydra3/hydra3_trancos_cfg.yml
#CAFFE_MODEL=models/pretrained_models/trancos/hydra3/trancos_hydra3.caffemodel
#DEPLOY=models/trancos/hydra3/hydra3_deploy.prototxt

# HYDRA 4s
#CONFIG_FILE=models/trancos/hydra4/hydra4_trancos_cfg.yml
#CAFFE_MODEL=models/pretrained_models/trancos/hydra4/trancos_hydra4.caffemodel
#DEPLOY=models/trancos/hydra4/hydra4_deploy.prototxt

LOG="experiments/logs/trancos_ccnn_`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# Time the task
T="$(date +%s)"

# Test Net
python src/test.py --dev ${GPU_DEV} --prototxt ${DEPLOY} --caffemodel ${CAFFE_MODEL} --cfg ${CONFIG_FILE}

T="$(($(date +%s)-T))"
echo "Time in seconds: ${T}"
