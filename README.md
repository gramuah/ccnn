# Towards perspective-free object counting with deep learning

By Daniel Oñoro-Rubio and Roberto J. López-Sastre.

GRAM, University of Alcalá, Alcalá de Henares, Spain.

This is the official code implementation of work described the [paper](http://agamenon.tsc.uah.es/Investigacion/gram/publications/eccv2016-onoro.pdf). In this repository, we include the implementation of CCNN and Hydra.

### License

The license information of this project is described in the file "LICENSE.txt".

### Cite us

Was our code useful for you? Please cite us:

    @inproceedings{onoro2016,
        Author = {O\~noro-Rubio, D. and L\'opez-Sastre, R.},
        Title = {Towards perspective-free object counting with deep learning},
        Booktitle = {ECCV},
        Year = {2016}
    }

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)
5. [Usage](#usage)
6. [Download pre-trained models](#download-pre-trained-models)
7. [Download a dataset](#download-a-dataset)
8. [Acknowledgements](#acknowledgements)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  ```

2. Python packages you might not have: `cython`, `python-opencv`, `easydict`, `opencv`


### Requirements: hardware

This code allows the usage of CPU and GPU. We strongly recommend the usage the GPU due it is going to short the training and testing time by a factor of 10 or so.

1. For training, we recommend having a GPU with at least 3GB of memory.

2. For testing, a GPU with 2GB memory is enough.

### Installation (sufficient for the demo)

1. Be sure you have added to your `PATH` the `tools` directory of your `Caffe` compilation:

    ```Shell
    export PATH=$CAFFE_ROOT/build/tools:$PATH
    ```
    
2. Be sure you have added your `pycaffe` compilation into your `PYTHONPATH`:

	```Shell
    export PYTHONPATH=$CAFFE_ROOT/python:$PYTHONPATH
    ```

3. Download the [TRANCOS dataset](http://agamenon.tsc.uah.es/Personales/rlopez/data/trancos/) by executing the following command:

	```Shell
    ./tools/get_trancos.sh
    ```

4. Download the TRANCOS CCNN model. **Comming soon**
    
### Demo

After successfully completing [basic installation](#installation-sufficient-for-the-demo), you'll be ready to run the demo.

    ```
    ./tools/demo.sh
    ```

### Usage

To train an test your own model, you should follow the next steps:

1. Edit the configuration file "ccnn_trancos_cfg.yml" placed into the corresponding model folder.
2. Launch the system by:

	```Shell
    ./experiments/scripts/trancos_train_test.sh
    ```
### Download pre-trained models

Comming soon...


### Download a dataset

In order to download and setup a dataset we recommend to use our scripts:

* [TRANCOS dataset](http://agamenon.tsc.uah.es/Personales/rlopez/data/trancos/):
 
	```Shell
    ./tools/get_trancos.sh
    ```

* [UCSD dataset](http://www.svcl.ucsd.edu/projects/peoplecnt/):

	```Shell
    ./tools/get_ucsd.sh
    ```

**Note:** Make sure the folder "data/" does not already contain the dataset.

### Acknowledgements
This work is supported by the projects of the DGT with references SPIP2014-1468 and SPIP2015-01809, and the project of the MINECO TEC2013-45183-R.
