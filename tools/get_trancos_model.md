# Instructions to obtain TRANCOS models
1. Download the pre-trained models using the TRANCOS dataset.
[Direct link](https://universidaddealcala-my.sharepoint.com/:u:/g/personal/gram_uah_es/ERdcsUU57ZFCk28lbmG16WsBDWiU71yRwgJd0kpX-RmM8g?&Download=1)

2. Untar 
`tar -zxvf trancos_ccnn.caffemodel.tar.gz`

3. Create the needed directory and the move file with the pre-trained model on it.
```
mkdir models/pretrained_models/trancos/ccnn
mv trancos_ccnn.caffemodel models/pretrained_models/trancos/ccnn
```

4. Clean
`rm trancos_ccnn.caffemodel.tar.gz`
