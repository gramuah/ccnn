#! /bin/bash

# Usage:
# ./experiments/scripts/trancos_train_net.sh GPU_ID

export PYTHONUNBUFFERED="True"

# Parameters
GPU_DEV=0

# Uncomment one of the following set of parameters

# CCNN
CONFIG_FILE=models/trancos/ccnn/ccnn_trancos_cfg.yml
CAFFE_MODEL=genfiles/output_models/trancos/ccnn/ccnn_trancos_iter_50000.caffemodel
DEPLOY=models/trancos/ccnn/ccnn_deploy.prototxt
SOLVER=models/trancos/ccnn/ccnn_solver.prototxt

# HYDRA 2s
#CONFIG_FILE=models/trancos/hydra2/hydra2_trancos_cfg.yml
#CAFFE_MODEL=genfiles/output_models/trancos/trancos/hydra2_trancos_iter_30000.caffemodel
#DEPLOY=models/trancos/hydra2/hydra2_deploy.prototxt
#SOLVER=models/trancos/hydra2/hydra2_solver.prototxt

# HYDRA 3s
#CONFIG_FILE=models/trancos/hydra3/hydra3_trancos_cfg.yml
#CAFFE_MODEL=genfiles/output_models/trancos/trancos/hydra3_trancos_iter_30000.caffemodel
#DEPLOY=models/trancos/hydra3/hydra3_deploy.prototxt
#SOLVER=models/trancos/hydra3/hydra3_solver.prototxt

# HYDRA 4s
#CONFIG_FILE=models/trancos/hydra4/hydra4_trancos_cfg.yml
#CAFFE_MODEL=genfiles/output_models/trancos/trancos/hydra4_trancos_iter_30000.caffemodel
#DEPLOY=models/trancos/hydra4/hydra4_deploy.prototxt
#SOLVER=models/trancos/hydra4/hydra4_solver.prototxt

LOG="experiments/logs/trancos_ccnn_`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# Time the task
T="$(date +%s)"

# Generate Features
python src/gen_features.py --cfg ${CONFIG_FILE}

# Train Net
caffe train -solver ${SOLVER}

# Test Net
python src/test.py --dev ${GPU_DEV} --prototxt ${DEPLOY} --caffemodel ${CAFFE_MODEL} --cfg ${CONFIG_FILE}

T="$(($(date +%s)-T))"
echo "Time in seconds: ${T}"
