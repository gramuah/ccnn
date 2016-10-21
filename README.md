# Towards perspective-free object counting with deep learning

By [Daniel Oñoro-Rubio](https://es.linkedin.com/in/daniel-oñoro-71062756) and [Roberto J. López-Sastre](http://agamenon.tsc.uah.es/Personales/rlopez/).

GRAM, University of Alcalá, Alcalá de Henares, Spain.

This is the official code implementation of the work described in our [ECCV 2016 paper](http://agamenon.tsc.uah.es/Investigacion/gram/publications/eccv2016-onoro.pdf). 


This repository provides the implementation of CCNN and Hydra models for object counting.

## Cite us

Was our code useful for you? Please cite us:

    @inproceedings{onoro2016,
        Author = {O\~noro-Rubio, D. and L\'opez-Sastre, R.~J.},
        Title = {Towards perspective-free object counting with deep learning},
        Booktitle = {ECCV},
        Year = {2016}
    }


## License

The license information of this project is described in the file "LICENSE.txt".



## Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#basic-installation-sufficient-for-the-demo)
4. [Demo](#demo)
5. [How to reproduce the results of the paper](#how-to-reproduce-the-results-of-the-paper)
6. [Remarks](#remarks)
7. [Acknowledgements](#acknowledgements)

### Requirements: software

1. Use a Linux distribution. We have developed and tested the code on [Ubuntu](http://www.ubuntu.com/).


2. Requirements for `Caffe` and `pycaffe`. Follow the [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html).

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  ```

3. Python packages you need: `cython`, `python-opencv`, `python-h5py`, `easydict`, `pillow (version >= 3.4.2)`.


### Requirements: hardware

This code allows the usage of CPU and GPU, but we strongly recommend the usage of GPU.

1. For training, we recommend using a GPU with at least 3GB of memory.

2. For testing, a GPU with 2GB of memory is enough.

### Basic installation (sufficient for the demo)

1. Be sure you have added to your `PATH` the `tools` directory of your `Caffe` installation:

    ```Shell
    export PATH=<your_caffe_root_path>/build/tools:$PATH
    ```
    
2. Be sure you have added your `pycaffe` compilation into your `PYTHONPATH`:
    
    ```Shell
    export PYTHONPATH=<your_caffe_root_path>/python:$PYTHONPATH
    ```
    
### Demo

We here provide a demo consisting in predicting the number of vehicles in the test images of the [TRANCOS dataset](http://agamenon.tsc.uah.es/Personales/rlopez/data/trancos/), which was used in our ECCV paper. 

This demo uses the CCNN model described in the paper. The results reported in the paper can be reproduced with this demo.

To run the demo, these are the steps to follow:

1. Download the TRANCOS dataset by executing the following script provided:
	```Shell
	./tools/get_trancos.sh
	```

2. You must have now a new directory with the TRANCOS dataset in the path `data/TRANCOS`.

3. Download the TRANCOS CCNN pretrained model.
	```Shell
	./tools/get_trancos_model.sh
	```

4. Finally, to run the demo, simply execute the following command:
	```Shell
	./tools/demo.sh
	```

### How to reproduce the results of the paper?

We provide here the scripts needed to **train** and **test** all the models (CCNN and Hydra) with the datasets used in our ECCV paper. These are the steps to follow.

#### Download a dataset

In order to download and setup a dataset we recommend to use our scripts. To do so, just place yourself in the $PROJECT directory and run one of the following scripts:

* [TRANCOS dataset](http://agamenon.tsc.uah.es/Personales/rlopez/data/trancos/):
 
	```Shell
    ./tools/get_trancos.sh
    ```

* [UCSD dataset](http://www.svcl.ucsd.edu/projects/peoplecnt/):

	```Shell
    ./tools/get_ucsd.sh
    ```

* [UCF dataset](http://crcv.ucf.edu/data/crowd_counting.php):

	```Shell
    ./tools/get_ucf.sh
    ```

**Note:** Make sure the folder "data/" does not already contain the dataset.


#### Download pre-trained models

All our pre-trained models can be downloaded using the corresponding script:

	```Shell
    ./tools/get_all_DATASET_CHOSEN_models.sh
    ```
Simply substitute DATASET_CHOSEN by: trancos, ucsd or ucf.

#### Test the pretrained models
1. Edit the corresponding script $PROJECT/experiments/scripts/DATASET_CHOSEN_test_pretrained.sh

2. Run the corresponding scripts.

       ```Shell
    ./experiments/scripts/DATASET_CHOSEN_test_pretrained.sh
    ```
Note that this pretrained models will let you reproduce the results in our paper.


#### Train/test the model chosen

1. Edit the launching script (e.g.: $PROJECT/experiments/scripts/DATASET_CHOSEN_train_test.sh).

2. Place you in $PROJECT folder and run the launching script by typing:

	```Shell
    ./experiments/scripts/DATASET_CHOSEN_train_test.sh
    ```


### Remarks

In order to provide a better distribution, this repository *unifies and reimplements* in Python some of the original modules. Due to these changes in the libraries used, the results produced by this software might be slightly different from the ones reported in the paper.


### Acknowledgements
This work is supported by the projects of the DGT with references SPIP2014-1468 and SPIP2015-01809, and the project of the MINECO TEC2013-45183-R.
