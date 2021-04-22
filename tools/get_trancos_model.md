# Download the pretrained models using the TRANCOS dataset.
Direct link(https://universidaddealcala-my.sharepoint.com/:u:/g/personal/gram_uah_es/ERdcsUU57ZFCk28lbmG16WsBDWiU71yRwgJd0kpX-RmM8g?&Download=1)

# Untar 
`tar -zxvf trancos_ccnn.caffemodel.tar.gz`

# Create dir and move file to the pretrained models dir
```
mkdir models/pretrained_models/trancos
mkdir models/pretrained_models/trancos/ccnn
mv trancos_ccnn.caffemodel models/pretrained_models/trancos/ccnn
```

# Clean
`rm trancos_ccnn.caffemodel.tar.gz`
