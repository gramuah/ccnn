#! /bin/bash

# Download the pretrained models using the TRANCOS dataset.
wget http://agamenon.tsc.uah.es/Personales/rlopez/data/ccnn/trancos_ccnn.caffemodel.tar.gz

# Untar
tar -zxvf trancos_ccnn.caffemodel.tar.gz

# Create dir and move file
mkdir models/pretrained_models/trancos
mv trancos_ccnn.caffemodel models/pretrained_models/trancos

# Clean
rm trancos_ccnn.caffemodel.tar.gz
