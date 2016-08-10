#! /bin/bash

# Downlaod dataset
wget http://crcv.ucf.edu/data/UCFCrowdCountingDataset_CVPR13.rar

# Extract
mkdir UCFCrowdCountingDataset_CVPR13
rar e UCFCrowdCountingDataset_CVPR13.rar UCFCrowdCountingDataset_CVPR13

# Build Directory tree
mkdir data/UCF
mkdir data/UCF/images
mkdir data/UCF/image_sets
mkdir data/UCF/params

# Move notated images
mv UCFCrowdCountingDataset_CVPR13/*.jpg data/UCF/images
mv UCFCrowdCountingDataset_CVPR13/*.mat data/UCF/params

# Create dot maps
python tools/gen_ucf_dotmaps.py --notationdir data/UCF/params --imdir data/UCF/images

# Clean
rm  UCFCrowdCountingDataset_CVPR13.rar
rm -fr UCFCrowdCountingDataset_CVPR13
