# Download all the pretrained models using the UCF dataset

Follow these instructions:

1. Download all the models in [this fhared folder](https://universidaddealcala-my.sharepoint.com/:f:/g/personal/gram_uah_es/EkrXzZVeEqZKpiN7GtZNP6EBfYOACzFsTyY6rehtwiwatQ?e=SskaLd).
2. Create the following dirs

```
mkdir models/pretrained_models/ucf
mkdir models/pretrained_models/ucf/ccnn
mkdir models/pretrained_models/ucf/hydra2
mkdir models/pretrained_models/ucf/hydra3
```
3. Organize the files (untar the files, move and clean)
```
for FOLD_NUM in 0 1 2 3 4
do
  # Form tar files names
  CCNN_TAR=ucf_ccnn_${FOLD_NUM}.caffemodel.tar.gz
  CCNN_MODEL=ucf_ccnn_${FOLD_NUM}.caffemodel
  HYDRA2_TAR=ucf_hydra2_${FOLD_NUM}.caffemodel.tar.gz
  HYDRA2_MODEL=ucf_hydra2_${FOLD_NUM}.caffemodel
  HYDRA3_TAR=ucf_hydra3_${FOLD_NUM}.caffemodel.tar.gz
  HYDRA3_MODEL=ucf_hydra3_${FOLD_NUM}.caffemodel

  #CCNN models
  tar -zxvf ${CCNN_TAR}
  mv ${CCNN_MODEL} models/pretrained_models/ucf/ccnn
  rm ${CCNN_TAR}

  #Hydra2 models
  tar -zxvf ${HYDRA2_TAR}
  mv ${HYDRA2_MODEL} models/pretrained_models/ucf/hydra2
  rm ${HYDRA2_TAR}

  #Hydra3 models
  tar -zxvf ${HYDRA3_TAR}
  mv ${HYDRA3_MODEL} models/pretrained_models/ucf/hydra3
  rm ${HYDRA3_TAR}

done
```
