#! /bin/bash

# Usage:
# ./experiments/scripts/_train_net.sh GPU_ID

export PYTHONUNBUFFERED="True"

# Parameters
GPU_DEV=0

# Uncoment one of the following set of parameters

# CCNN
CONFIG_FILE=models/ucf/ccnn/ccnn_ucf_set_0_cfg.yml # Configure to choose dataset
CAFFE_MODEL=genfiles/output_models/ucf/ccnn/ccnn_ucf_max_iter_50000.caffemodel
DEPLOY=models/ucf/ccnn/ccnn_deploy.prototxt
SOLVER=models/ucf/ccnn/ccnn_max_solver.prototxt

# HYDRA 2s
#CONFIG_FILE=models/ucf/hydra2/hydra2_ucf_cfg.yml
#CAFFE_MODEL=genfiles/output_models/ucf/ucf/hydra2_ucf_max_iter_25000.caffemodel
#DEPLOY=models/ucf/hydra2/hydra2_deploy.prototxt
#SOLVER=models/ucf/hydra2/hydra2_max_solver.prototxt

# HYDRA 3s
#CONFIG_FILE=models/ucf/hydra3/hydra3_ucf_cfg.yml
#CAFFE_MODEL=genfiles/output_models/ucf/ucf/hydra3_ucf_max_iter_25000.caffemodel
#DEPLOY=models/ucf/hydra3/hydra3_deploy.prototxt
#SOLVER=models/ucf/hydra3/hydra3_max_solver.prototxt

LOG="experiments/logs/ucf_ccnn_`date +'%Y-%m-%d_%H-%M-%S'`.txt"
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
