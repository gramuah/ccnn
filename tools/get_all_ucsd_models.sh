#! /bin/bash


# Download all the pretrained models using the UCSD dataset

# Create dir
mkdir models/pretrained_models/ucsd

# Download, untar, create dir, move and clean
mkdir models/pretrained_models/ucsd/ccnn

wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/ucsd_ccnn_down.caffemodel.tar.gz
tar -zxvf ucsd_ccnn_down.caffemodel.tar.gz
mv ucsd_ccnn_down.caffemodel models/pretrained_models/ucsd/ccnn
rm ucsd_ccnn_down.caffemodel.tar.gz

wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/ucsd_ccnn_max.caffemodel.tar.gz
tar -zxvf ucsd_ccnn_max.caffemodel.tar.gz
mv ucsd_ccnn_max.caffemodel models/pretrained_models/ucsd/ccnn
rm ucsd_ccnn_max.caffemodel.tar.gz

wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/ucsd_ccnn_min.caffemodel.tar.gz
tar -zxvf ucsd_ccnn_min.caffemodel.tar.gz
mv ucsd_ccnn_min.caffemodel models/pretrained_models/ucsd/ccnn
rm ucsd_ccnn_min.caffemodel.tar.gz

wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/ucsd_ccnn_up.caffemodel.tar.gz
tar -zxvf ucsd_ccnn_up.caffemodel.tar.gz
mv ucsd_ccnn_up.caffemodel models/pretrained_models/ucsd/ccnn
rm ucsd_ccnn_up.caffemodel.tar.gz

#HYDRA2

mkdir models/pretrained_models/ucsd/hydra2

wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/ucsd_hydra2_down.caffemodel.tar.gz
tar -zxvf ucsd_hydra2_down.caffemodel.tar.gz
mv ucsd_hydra2_down.caffemodel models/pretrained_models/ucsd/hydra2
rm ucsd_hydra2_down.caffemodel.tar.gz

wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/ucsd_hydra2_max.caffemodel.tar.gz
tar -zxvf ucsd_hydra2_max.caffemodel.tar.gz
mv ucsd_hydra2_max.caffemodel models/pretrained_models/ucsd/hydra2
rm ucsd_hydra2_max.caffemodel.tar.gz

wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/ucsd_hydra2_min.caffemodel.tar.gz
tar -zxvf ucsd_hydra2_min.caffemodel.tar.gz
mv ucsd_hydra2_min.caffemodel models/pretrained_models/ucsd/hydra2
rm ucsd_hydra2_min.caffemodel.tar.gz

wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/ucsd_hydra2_up.caffemodel.tar.gz
tar -zxvf ucsd_hydra2_up.caffemodel.tar.gz
mv ucsd_hydra2_up.caffemodel models/pretrained_models/ucsd/hydra2
rm ucsd_hydra2_up.caffemodel.tar.gz

#HYDRA3

mkdir models/pretrained_models/ucsd/hydra3

wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/ucsd_hydra3_down.caffemodel.tar.gz
tar -zxvf ucsd_hydra3_down.caffemodel.tar.gz
mv ucsd_hydra3_down.caffemodel models/pretrained_models/ucsd/hydra3
rm ucsd_hydra3_down.caffemodel.tar.gz

wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/ucsd_hydra3_max.caffemodel.tar.gz
tar -zxvf ucsd_hydra3_max.caffemodel.tar.gz
mv ucsd_hydra3_max.caffemodel models/pretrained_models/ucsd/hydra3
rm ucsd_hydra3_max.caffemodel.tar.gz

wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/ucsd_hydra3_min.caffemodel.tar.gz
tar -zxvf ucsd_hydra3_min.caffemodel.tar.gz
mv ucsd_hydra3_min.caffemodel models/pretrained_models/ucsd/hydra3
rm ucsd_hydra3_min.caffemodel.tar.gz

wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/ucsd_hydra3_up.caffemodel.tar.gz
tar -zxvf ucsd_hydra3_up.caffemodel.tar.gz
mv ucsd_hydra3_up.caffemodel models/pretrained_models/ucsd/hydra3
rm ucsd_hydra3_up.caffemodel.tar.gz













