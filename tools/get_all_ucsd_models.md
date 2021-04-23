# Download all the pretrained models using the UCSD dataset

Follow these instructions

1. Download all the models in [this shared folder](https://universidaddealcala-my.sharepoint.com/:f:/g/personal/gram_uah_es/EupJym-OdldIqJDvw4pwZRoBRjLcVwDPnpWSlXrlrHuf4g?e=b7ZGiB)

2. Create dir
```
mkdir models/pretrained_models/ucsd
mkdir models/pretrained_models/ucsd/ccnn
mkdir models/pretrained_models/ucsd/hydra2
mkdir models/pretrained_models/ucsd/hydra3
```

3. Untar, move and clean files
```
#CCNN
tar -zxvf ucsd_ccnn_down.caffemodel.tar.gz
mv ucsd_ccnn_down.caffemodel models/pretrained_models/ucsd/ccnn
rm ucsd_ccnn_down.caffemodel.tar.gz

tar -zxvf ucsd_ccnn_max.caffemodel.tar.gz
mv ucsd_ccnn_max.caffemodel models/pretrained_models/ucsd/ccnn
rm ucsd_ccnn_max.caffemodel.tar.gz

tar -zxvf ucsd_ccnn_min.caffemodel.tar.gz
mv ucsd_ccnn_min.caffemodel models/pretrained_models/ucsd/ccnn
rm ucsd_ccnn_min.caffemodel.tar.gz

tar -zxvf ucsd_ccnn_up.caffemodel.tar.gz
mv ucsd_ccnn_up.caffemodel models/pretrained_models/ucsd/ccnn
rm ucsd_ccnn_up.caffemodel.tar.gz

#HYDRA2

tar -zxvf ucsd_hydra2_down.caffemodel.tar.gz
mv ucsd_hydra2_down.caffemodel models/pretrained_models/ucsd/hydra2
rm ucsd_hydra2_down.caffemodel.tar.gz

tar -zxvf ucsd_hydra2_max.caffemodel.tar.gz
mv ucsd_hydra2_max.caffemodel models/pretrained_models/ucsd/hydra2
rm ucsd_hydra2_max.caffemodel.tar.gz

tar -zxvf ucsd_hydra2_min.caffemodel.tar.gz
mv ucsd_hydra2_min.caffemodel models/pretrained_models/ucsd/hydra2
rm ucsd_hydra2_min.caffemodel.tar.gz

tar -zxvf ucsd_hydra2_up.caffemodel.tar.gz
mv ucsd_hydra2_up.caffemodel models/pretrained_models/ucsd/hydra2
rm ucsd_hydra2_up.caffemodel.tar.gz

#HYDRA3

tar -zxvf ucsd_hydra3_down.caffemodel.tar.gz
mv ucsd_hydra3_down.caffemodel models/pretrained_models/ucsd/hydra3
rm ucsd_hydra3_down.caffemodel.tar.gz

tar -zxvf ucsd_hydra3_max.caffemodel.tar.gz
mv ucsd_hydra3_max.caffemodel models/pretrained_models/ucsd/hydra3
rm ucsd_hydra3_max.caffemodel.tar.gz

tar -zxvf ucsd_hydra3_min.caffemodel.tar.gz
mv ucsd_hydra3_min.caffemodel models/pretrained_models/ucsd/hydra3
rm ucsd_hydra3_min.caffemodel.tar.gz

tar -zxvf ucsd_hydra3_up.caffemodel.tar.gz
mv ucsd_hydra3_up.caffemodel models/pretrained_models/ucsd/hydra3
rm ucsd_hydra3_up.caffemodel.tar.gz
```
