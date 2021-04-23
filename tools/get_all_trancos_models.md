# Download all the pretrained models using the TRANCOS dataset

Follow these instructions

1. Create the following dir

```
mkdir models/pretrained_models/trancos

```

2. Download the models:
  * [trancos_ccnn.caffemodel.tar.gz](https://universidaddealcala-my.sharepoint.com/:u:/g/personal/gram_uah_es/ERdcsUU57ZFCk28lbmG16WsBDWiU71yRwgJd0kpX-RmM8g?&Download=1)
  * [trancos_hydra2.caffemodel.tar.gz](https://universidaddealcala-my.sharepoint.com/:u:/g/personal/gram_uah_es/ESiIMJj29KNEvmJX8pQVaRsBVBwB7EPoRusf6votOV-VTw?&Download=1)
  * [trancos_hydra3.caffemodel.tar.gz](https://universidaddealcala-my.sharepoint.com/:u:/g/personal/gram_uah_es/EfJZknF50eJAvJfacXf1-CQBT7jTuSuGAXkegjuNkpDcNg?&Download=1)
  * [trancos_hydra4.caffemodel.tar.gz](https://universidaddealcala-my.sharepoint.com/:u:/g/personal/gram_uah_es/EfF-6vzJL-9MnbJsCMYzO34BvT3r_o_awIq472BeaLjiMg?&Download=1)

3. Create dirs and extract the files (use this code if you wish)
```
tar -zxvf trancos_ccnn.caffemodel.tar.gz
mkdir models/pretrained_models/trancos/ccnn
mv trancos_ccnn.caffemodel models/pretrained_models/trancos/ccnn

tar -zxvf trancos_hydra2.caffemodel.tar.gz
mkdir models/pretrained_models/trancos/hydra2
mv trancos_hydra2.caffemodel models/pretrained_models/trancos/hydra2

tar -zxvf trancos_hydra3.caffemodel.tar.gz
mkdir models/pretrained_models/trancos/hydra3
mv trancos_hydra3.caffemodel models/pretrained_models/trancos/hydra3

tar -zxvf trancos_hydra4.caffemodel.tar.gz
mkdir models/pretrained_models/trancos/hydra4
mv trancos_hydra4.caffemodel models/pretrained_models/trancos/hydra4

```
4. Optional (clean the downloaded files)
```
rm trancos_ccnn.caffemodel.tar.gz
rm trancos_hydra2.caffemodel.tar.gz
rm trancos_hydra3.caffemodel.tar.gz
rm trancos_hydra4.caffemodel.tar.gz
```
