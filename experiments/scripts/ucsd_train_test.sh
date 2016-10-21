#! /bin/bash

# Usage:
# ./experiments/scripts/ucsd_train_net.sh GPU_ID

export PYTHONUNBUFFERED="True"

# Parameters
GPU_DEV=0

# Uncomment one of the following set of parameters

# CCNN
CONFIG_FILE=models/ucsd/ccnn/ccnn_ucsd_cfg.yml
CAFFE_MODEL=genfiles/output_models/ucsd/ccnn/ccnn_ucsd_max_iter_50000.caffemodel
DEPLOY=models/ucsd/ccnn/ccnn_deploy.prototxt
SOLVER=models/ucsd/ccnn/ccnn_max_solver.prototxt

# HYDRA 2s
#CONFIG_FILE=models/ucsd/hydra2/hydra2_ucsd_cfg.yml
#CAFFE_MODEL=genfiles/output_models/ucsd/ucsd/hydra2_ucsd_max_iter_25000.caffemodel
#DEPLOY=models/ucsd/hydra2/hydra2_deploy.prototxt
#SOLVER=models/ucsd/hydra2/hydra2_max_solver.prototxt

# HYDRA 3s
#CONFIG_FILE=models/ucsd/hydra3/hydra3_ucsd_cfg.yml
#CAFFE_MODEL=genfiles/output_models/ucsd/ucsd/hydra3_ucsd_max_iter_25000.caffemodel
#DEPLOY=models/ucsd/hydra3/hydra3_deploy.prototxt
#SOLVER=models/ucsd/hydra3/hydra3_max_solver.prototxt

LOG="experiments/logs/ucsd_ccnn_`date +'%Y-%m-%d_%H-%M-%S'`.txt"
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
