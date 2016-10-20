#! /bin/bash


# Download all the pretrained models using the TRANCOS dataset

# Create dir and move file
mkdir models/pretrained_models/trancos

# Download, untar, move and clean
wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/trancos_ccnn.caffemodel.tar.gz
tar -zxvf trancos_ccnn.caffemodel.tar.gz
mkdir models/pretrained_models/trancos/ccnn
mv trancos_ccnn.caffemodel models/pretrained_models/trancos/ccnn
rm trancos_ccnn.caffemodel.tar.gz

wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/trancos_hydra2.caffemodel.tar.gz
tar -zxvf trancos_hydra2.caffemodel.tar.gz
mkdir models/pretrained_models/trancos/hydra2
mv trancos_hydra2.caffemodel models/pretrained_models/trancos/hydra2
rm trancos_hydra2.caffemodel.tar.gz

wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/trancos_hydra3.caffemodel.tar.gz
tar -zxvf trancos_hydra3.caffemodel.tar.gz
mkdir models/pretrained_models/trancos/hydra3
mv trancos_hydra3.caffemodel models/pretrained_models/trancos/hydra3
rm trancos_hydra3.caffemodel.tar.gz


wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/trancos_hydra4.caffemodel.tar.gz
tar -zxvf trancos_hydra4.caffemodel.tar.gz
mkdir models/pretrained_models/trancos/hydra4
mv trancos_hydra4.caffemodel models/pretrained_models/trancos/hydra4
rm trancos_hydra4.caffemodel.tar.gz


