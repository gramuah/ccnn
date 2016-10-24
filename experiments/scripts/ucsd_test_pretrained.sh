#! /bin/bash

# Usage:
# ./experiments/scripts/ucsd_train_net.sh GPU_ID

export PYTHONUNBUFFERED="True"

# Parameters
GPU_DEV=0

# Parameters, uncomment one of the following set of parameters
# CCNN
CONFIG_FILE=models/ucsd/ccnn/ccnn_ucsd_cfg.yml
CAFFE_MODEL=models/pretrained_models/ucsd/ccnn/ucsd_ccnn_max.caffemodel
DEPLOY=models/ucsd/ccnn/ccnn_deploy.prototxt # Modify it to choose another dataset

# HYDRA 2s
#CONFIG_FILE=models/ucsd/hydra2/hydra2_ucsd_cfg.yml
#CAFFE_MODEL=models/pretrained_models/ucsd/hydra2/ucsd_hydra2_max.caffemodel
#DEPLOY=models/ucsd/hydra2/hydra2_deploy.prototxt # Modify it to choose another dataset

# HYDRA 3s
#CONFIG_FILE=models/ucsd/hydra3/hydra3_ucsd_cfg.yml
#CAFFE_MODEL=models/pretrained_models/ucsd/hydra3/ucsd_hydra3_max.caffemodel
#DEPLOY=models/ucsd/hydra3/hydra3_deploy.prototxt # Modify it to choose another dataset

LOG="experiments/logs/ucsd_ccnn_`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# Time the task
T="$(date +%s)"

# Test Net
python src/test.py --dev ${GPU_DEV} --prototxt ${DEPLOY} --caffemodel ${CAFFE_MODEL} --cfg ${CONFIG_FILE}

T="$(($(date +%s)-T))"
echo "Time in seconds: ${T}"
