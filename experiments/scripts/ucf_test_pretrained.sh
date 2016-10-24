#! /bin/bash

# Usage:
# ./experiments/scripts/_train_net.sh GPU_ID

export PYTHONUNBUFFERED="True"

# Parameters
GPU_DEV=0

# Parameters, uncomment one of the following set of parameters
# CCNN
CONFIG_FILE=models/ucf/ccnn/ccnn_ucf_set_ 
CAFFE_MODEL=models/pretrained_models/ucf/ccnn/ucf_ccnn
DEPLOY=models/ucf/ccnn/ccnn_deploy.prototxt

# HYDRA 2s
#CONFIG_FILE=models/ucf/hydra2/hydra2_ucf_set_
#CAFFE_MODEL=models/pretrained_models/ucf/hydra2/ucf_hydra2
#DEPLOY=models/ucf/hydra2/hydra2_deploy.prototxt

# HYDRA 3s
#CONFIG_FILE=models/ucf/hydra3/hydra3_ucf_set_
#CAFFE_MODEL=models/pretrained_models/ucf/hydra3/ucf_hydra3
#DEPLOY=models/ucf/hydra3/hydra3_deploy.prototxt

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
