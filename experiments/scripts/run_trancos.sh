#! /bin/bash

# Usage:
# ./experiments/scripts/train_net.sh GPU_ID

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/trancos_ccnn_`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# Time the task
T="$(date +%s)"

# Generate Features
python src/gen_features.py

# Collect all the features into a txt file
find `pwd`/genfiles/features/trancos_train_feat*.h5 > genfiles/features/train.txt

# Train Net
caffe train -solver models/trancos/ccnn/ccnn_solver.prototxt

# Test Net
python src/test.py

T="$(($(date +%s)-T))"
echo "Time in seconds: ${T}"
