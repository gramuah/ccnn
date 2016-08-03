#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    @author: Daniel Oñoro Rubio
    @contact: daniel.onoro@edu.uah.es
    @date: 23/09/2015
'''

#===========================================================================
# Dependency loading
#===========================================================================

# File storage
import h5py

# System
import sys, getopt
import time
import os.path
import glob

# Vision and maths
import numpy as np
import caffe
from skimage.transform import resize

# Others
import utils
import test as tst
import gen_features as gfeat

# Plotting
# import matplotlib.pyplot as plt
from pylab import *
from dns.rdatatype import SIG

def load_image_list(ucf_names_file):
    '''
    @brief: Read the txt file that contains the unsorted image names of the UCF
    and returns it as a list.
    '''
    image_names = []
    with open(ucf_names_file, 'r') as f:
        file_lines = f.readlines()
        # Clean lines
        image_names = [fname.strip() for fname in file_lines ]

    return image_names

def main(argv):      
    # Some constants
    K_FOLD = 5
    UCF_N_IMAGES = 50
    
    ucf_names_file = 'data/UCF_CC_50/image_sets/ucf_order.txt'
    
    # Laod unsorted set
    image_names = load_image_list(ucf_names_file)

    # Sanity check
    assert len(image_names) == UCF_N_IMAGES
    
    test_size = UCF_N_IMAGES / K_FOLD
    for i in range(K_FOLD):
        # Get test sets
        test_idx = np.arange(i*test_size,(i+1)*test_size)
        train_mask = np.ones(UCF_N_IMAGES, dtype=np.bool)
        train_mask[test_idx] = False
         
        # Generate test & train datasets files
        gen_UCF_dataset_file(GENFILES + '/data/ccnn_test_set.txt', image_names[test_idx])
        gen_UCF_dataset_file(GENFILES + '/data/ccnn_train_set.txt', image_names[train_mask])
         

  
    print "The end"

def train_cnn(solver_path):
    caffe.set_device(0)
    caffe.set_mode_gpu()
#    caffe.set_mode_cpu()
    solver = caffe.SGDSolver(solver_path)

    # Fully solve it 
    solver.solve()

    return True

def gen_UCF_dataset_file(output_path, image_idx):
    with open(output_path, 'w') as f:
        for ix in image_idx:
            f.write("{}.jpg\n".format(ix))

        f.close()

if __name__ == "__main__":
    main(sys.argv[1:])