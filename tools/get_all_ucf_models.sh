#! /bin/bash


# Download all the pretrained models using the UCF dataset

# Create dir
mkdir models/pretrained_models/ucf
mkdir models/pretrained_models/ucf/ccnn
mkdir models/pretrained_models/ucf/hydra2
mkdir models/pretrained_models/ucf/hydra3

# Download, untar, move and clean
for FOLD_NUM in 0 1 2 3 4
do
  # Form tar files names
  CCNN_TAR=ucf_ccnn_${FOLD_NUM}.caffemodel.tar.gz
  CCNN_MODEL=ucf_ccnn_${FOLD_NUM}.caffemodel
  HYDRA2_TAR=ucf_hydra2_${FOLD_NUM}.caffemodel.tar.gz
  HYDRA2_MODEL=ucf_hydra2_${FOLD_NUM}.caffemodel
  HYDRA3_TAR=ucf_hydra3_${FOLD_NUM}.caffemodel.tar.gz
  HYDRA3_MODEL=ucf_hydra3_${FOLD_NUM}.caffemodel

  # Get CCNN models
  wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/${CCNN_TAR}
  tar -zxvf ${CCNN_TAR}
  mv ${CCNN_MODEL} models/pretrained_models/ucf/ccnn
  rm ${CCNN_TAR}

  # Get Hydra2 models
  wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/${HYDRA2_TAR}
  tar -zxvf ${HYDRA2_TAR}
  mv ${HYDRA2_MODEL} models/pretrained_models/ucf/hydra2
  rm ${HYDRA2_TAR}

  # Get Hydra3 models
  wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/${HYDRA3_TAR}
  tar -zxvf ${HYDRA3_TAR}
  mv ${HYDRA3_MODEL} models/pretrained_models/ucf/hydra3
  rm ${HYDRA3_TAR}

done
