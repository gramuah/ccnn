#! /bin/bash

# Usage:
# ./experiments/scripts/_train_net.sh GPU_ID

export PYTHONUNBUFFERED="True"

# Parameters
GPU_DEV=0

# Modify to run other model
# CCNN
CONFIG_FILE=models/ucf/ccnn/ccnn_ucf_set_ 
CAFFE_MODEL=models/pretrained_models/ucf/ccnn/ucf_ccnn
DEPLOY=models/ucf/ccnn/ccnn_deploy.prototxt
SOLVER=models/ucf/ccnn/ccnn_solver.prototxt

LOG="experiments/logs/ucf_ccnn_`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# Time the task
T="$(date +%s)"

for IX in 0 1 2 3 4
do
  # Test Net
  python src/test.py --dev ${GPU_DEV} --prototxt ${DEPLOY} --caffemodel ${CAFFE_MODEL}_${IX}.caffemodel --cfg ${CONFIG_FILE}${IX}_cfg.yml
done

# Print MAE and MSD
python tools/gen_ucf_results.py --results genfiles/results/ccnn_ucf_set_

T="$(($(date +%s)-T))"
echo "Time in seconds: ${T}"
