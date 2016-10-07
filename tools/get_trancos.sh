#! /bin/bash

# Download dataset
wget http://agamenon.tsc.uah.es/Personales/rlopez/data/trancos/TRANCOS_v3.tar.gz

# Untar
tar -zxvf TRANCOS_v3.tar.gz

# Move and rename
mv TRANCOS_v3 data/TRANCOS

# Clean
rm TRANCOS_v3.tar.gz
