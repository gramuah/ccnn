# -*- coding: utf-8 -*-

'''
    @author: Daniel Oñoro Rubio
    @organization: GRAM, University of Alcalá, Alcalá de Henares, Spain
    @copyright: 
    @contact: daniel.onoro@edu.uah.es
    @date: 26/02/2015
'''

"""Towards perspective-free object counting with deep learning

This file specifies default config options for CCNN and Hydra model. 
"""

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# TRANCOS OPTIONS
#
__C.TRANCOS = edict()

# Database params
__C.TRANCOS.DOT_ENDING="dots.png"
__C.TRANCOS.MASK_ENDING="mask.mat"

# Feature extraction params
__C.TRANCOS.PW=115            # Base patch side
__C.TRANCOS.NR=800            # < 1 = dense extraction
__C.TRANCOS.SIG=15.0        
__C.TRANCOS.SPLIT=15          # Create a new file every X images
__C.TRANCOS.USE_MASK=True
__C.TRANCOS.FLIP=True
__C.TRANCOS.USE_PERSPECTIVE=False

# Paths and others
__C.TRANCOS.TRAINVAL_LIST='data/TRANCOS/image_sets/trainval.txt'
__C.TRANCOS.IM_FOLDER='data/TRANCOS/images/'
__C.TRANCOS.TRAINING_LIST='data/TRANCOS/image_sets/training.txt'
__C.TRANCOS.VALIDATION_LIST='data/TRANCOS/image_sets/validation.txt'
__C.TRANCOS.TRAINVAL_LIST='data/TRANCOS/image_sets/trainval.txt'
__C.TRANCOS.TEST_LIST='data/TRANCOS/image_sets/test.txt'
__C.TRANCOS.PERSPECTIVE_MAP=''
__C.TRANCOS.TRAIN_FEAT='genfiles/features/trancos_train_feat'
__C.TRANCOS.VAL_FEAT='genfiles/features/trancos_val_feat'

# CNN model params
__C.TRANCOS.CNN_PW_IN=72    # CNN patch width in
__C.TRANCOS.CNN_PW_OUT=18   # CNN patch width out
__C.TRANCOS.N_SCALES=1     	# HYDRA number of heads
