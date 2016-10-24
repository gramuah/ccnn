#! /bin/bash

# Usage:
# ./experiments/scripts/_train_net.sh GPU_ID

export PYTHONUNBUFFERED="True"

# Parameters
GPU_DEV=0

# Uncomment one of the following set of parameters

# CCNN
CONFIG_FILE=models/ucf/ccnn/ccnn_ucf_set_ 
CAFFE_MODEL=genfiles/output_models/ucf/ccnn/ccnn_ucf_iter_50000.caffemodel
DEPLOY=models/ucf/ccnn/ccnn_deploy.prototxt
SOLVER=models/ucf/ccnn/ccnn_solver.prototxt

# HYDRA 2s
#CONFIG_FILE=models/ucf/hydra2/hydra2_ucf_set_
#CAFFE_MODEL=genfiles/output_models/ucf/hydra2/hydra2_trancos_iter_25000.caffemodel
#DEPLOY=models/ucf/hydra2/hydra2_deploy.prototxt
#SOLVER=models/ucf/hydra2/hydra2_solver.prototxt

# HYDRA 3s
#CONFIG_FILE=models/ucf/hydra3/hydra3_ucf_set_
#CAFFE_MODEL=genfiles/output_models/ucf/hydra3/hydra3_trancos_iter_25000.caffemodel
#DEPLOY=models/ucf/hydra3/hydra3_deploy.prototxt
#SOLVER=models/ucf/hydra3/hydra3_solver.prototxt

LOG="experiments/logs/ucf_ccnn_`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# Time the task
T="$(date +%s)"

for IX in 0 1 2 3 4
do
  # Generate Features
  python src/gen_features.py --cfg ${CONFIG_FILE}${IX}_cfg.yml

  # Train Net
  caffe train -solver ${SOLVER}

  # Rename net in order to do not overwrite it in the next iteration
  mv  ${CAFFE_MODEL} ${CAFFE_MODEL}_${IX}
  
  # Test Net
  python src/test.py --dev ${GPU_DEV} --prototxt ${DEPLOY} --caffemodel ${CAFFE_MODEL}_${IX} --cfg ${CONFIG_FILE}${IX}_cfg.yml
done

# Print MAE and MSD
python tools/gen_ucf_results.py --results genfiles/results/ccnn_ucf_set_

T="$(($(date +%s)-T))"
echo "Time in seconds: ${T}"
