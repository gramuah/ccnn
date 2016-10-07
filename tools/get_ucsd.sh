#! /bin/bash

# Download dataset
wget http://www.svcl.ucsd.edu/projects/peoplecnt/db/ucsdpeds.zip

# Extract
unzip ucsdpeds.zip

# Build Directory tree
mkdir data/UCSD
mkdir data/UCSD/images
mkdir data/UCSD/image_sets
mkdir data/UCSD/params

# Move annotated images
mv ucsdpeds/vidf/vidf1_33_000.y/*.png data/UCSD/images
mv ucsdpeds/vidf/vidf1_33_001.y/*.png data/UCSD/images
mv ucsdpeds/vidf/vidf1_33_002.y/*.png data/UCSD/images
mv ucsdpeds/vidf/vidf1_33_003.y/*.png data/UCSD/images
mv ucsdpeds/vidf/vidf1_33_004.y/*.png data/UCSD/images
mv ucsdpeds/vidf/vidf1_33_005.y/*.png data/UCSD/images
mv ucsdpeds/vidf/vidf1_33_006.y/*.png data/UCSD/images
mv ucsdpeds/vidf/vidf1_33_007.y/*.png data/UCSD/images
mv ucsdpeds/vidf/vidf1_33_008.y/*.png data/UCSD/images
mv ucsdpeds/vidf/vidf1_33_009.y/*.png data/UCSD/images

# Generate datasets
python tools/gen_ucsd_dataset.py --imfolder data/UCSD/images --setsfolder data/UCSD/image_sets

# Get annotations
wget http://www.svcl.ucsd.edu/projects/peoplecnt/db/vidf-cvpr.zip

# Extract
unzip vidf-cvpr.zip

# Generate annotations
python tools/gen_ucsd_dotmaps.py --folder vidf-cvpr/ --output data/UCSD/images
python tools/gen_ucsd_extranotation.py --notation vidf-cvpr/ --output data/UCSD/params

# Clean
rm  ucsdpeds.zip
rm  vidf-cvpr.zip
rm -fr ucsdpeds
rm -fr vidf-cvpr
