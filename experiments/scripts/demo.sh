#! /bin/bash

# Usage:
# ./experiments/scripts/train_net.sh GPU_ID

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/trancos_ccnn_`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# Time the task
T="$(date +%s)"

# Test Net
python src/test.py

T="$(($(date +%s)-T))"
echo "Time in seconds: ${T}"
