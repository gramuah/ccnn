#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    @author: Daniel Oñoro Rubio
    @organization: GRAM, University of Alcalá, Alcalá de Henares, Spain
    @copyright: See LICENSE.txt
    @contact: daniel.onoro@edu.uah.es
    @date: Created on August 2, 2016
'''

"""
This script creates the "dot" maps that contain the ground truth of the UCF in 
the supported format.
"""

import sys, getopt, os
import numpy as np
import scipy.io as scio 
import cv2
from PIL import Image


def dispHelp(arg0):
    print "======================================================"
    print "                       Usage"
    print "======================================================"
    print "\t-h display this message"
    print "\t--noationdir <where the mat files are>"
    print "\t--imdir <where the images are>"


def main(argv):
    UCF_N_IMAGES = 50
    
    notation_folder = 'data/UCF/params/'
    imdir = 'data/UCF/images/'

    # Get parameters
    try:
        opts, _ = getopt.getopt(argv, "h:", ["notationdir=", "imdir="])
    except getopt.GetoptError as err:
        print "Error while parsing parameters: ", err
        dispHelp(argv[0])
        return
    
    for opt, arg in opts:
        if opt == '-h':
            dispHelp(argv[0])
            return
        elif opt in ("--notationdir"):
            notation_folder = arg
        elif opt in ("--imdir"):
            output = arg

    for i in range(UCF_N_IMAGES):
        notation_file = os.path.join( notation_folder, '{}_ann.mat'.format(i+1) )
        img_file = os.path.join( imdir, '{}.jpg'.format(i+1) )
        notation = scio.loadmat(notation_file, struct_as_record=False, squeeze_me=True)
        dots = notation['annPoints'] - 1 # Make 0 index (x,y)

        # Init image
        with Image.open(img_file) as im:
            [width, height] = im.size
        black_im = np.zeros((height,width,3), dtype=np.uint8) # UCSD size
        # Get dots
        dots = dots  -1 # 0 index
        dots = dots.astype(np.int32)
        mask = dots[:,0]<width
        mask = np.logical_and( mask, dots[:,1]<height )
        dots = dots[mask]
        # Draw dots           
        black_im[dots[:,1],dots[:,0],2] = 255
        # Save image
        im_name = os.path.join( imdir, '{}dots.png'.format(i+1) )
        cv2.imwrite(im_name, black_im)
    
    return 0

if __name__ == '__main__':
    main(sys.argv[1:])
