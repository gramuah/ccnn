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
This script creates the "dot" maps that contain the ground truth of the UCSD in 
the supported format.
"""


import sys, getopt, os
import numpy as np
import scipy.io as scio 
import cv2

def dispHelp(arg0):
    print "======================================================"
    print "                       Usage"
    print "======================================================"
    print "\t-h display this message"
    print "\t--folder <where the mat files are>"
    print "\t--output <where the images are>"


def main(argv):
    notation_folder = 'vidf-cvpr/'
    output = 'data/UCSD/images'

    # Get parameters
    try:
        opts, _ = getopt.getopt(argv, "h:", ["folder=", "output="])
    except getopt.GetoptError as err:
        print "Error while parsing parameters: ", err
        dispHelp(argv[0])
        return
    
    for opt, arg in opts:
        if opt == '-h':
            dispHelp(argv[0])
            return
        elif opt in ("--folder"):
            notation_folder = arg
        elif opt in ("--output"):
            output = arg

    for i in range(10):
        notation_file = os.path.join( notation_folder, 'vidf1_33_00{}_frame_full.mat'.format(i) )
        notation = scio.loadmat(notation_file, struct_as_record=False, squeeze_me=True)
        
        frames = notation['frame']
        
        for jx, frame in enumerate(frames):
            
            # Init image
            black_im = np.zeros((158,238,3), dtype=np.uint8) # UCSD size
            
            # Get dots
            loc = frame.loc 
            dots = loc[:, (1,0,2)]  -1 # 0 index
            dots[:,2] = 2 # Set red channel
            dots = dots.astype(np.int32)
            mask = dots[:,0]<158
            mask = np.logical_and( mask, dots[:,1]<238 )
            dots = dots[mask]
            # Draw dots           
            black_im[dots[:,0],dots[:,1],dots[:,2]] = 255
            # Save image
            im_name = os.path.join( output, 'vidf1_33_00{}_f{:03d}dots.png'.format(i, jx+1) )
            cv2.imwrite(im_name, black_im)
    
    return 0

if __name__ == '__main__':
    main(sys.argv[1:])
