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
This script converts the perspective map and the roi of the UCSD dataset
to the supported format.
"""

import sys, getopt, os
import h5py
import numpy as np
import scipy.io as scio 

def dispHelp(arg0):
    print "======================================================"
    print "                       Usage"
    print "======================================================"
    print "\t-h display this message"
    print "\t--folder <where the images are>"


def main(argv):
    notation_folder = 'vidf-cvpr/'
    output = 'data/UCSD/params'

    # Get parameters
    try:
        opts, _ = getopt.getopt(argv, "h:", ["notation=", "output="])
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

    # Extract and convert mask
    mask_file = os.path.join( notation_folder, 'vidf1_33_roi_mainwalkway.mat' )
    m_obj = scio.loadmat(mask_file, struct_as_record=False, squeeze_me=True)
    mask = m_obj['roi'].mask
    
    mask_opt_fname = os.path.join( output, 'ucsd_mask.h5' )
    with h5py.File(mask_opt_fname, 'w') as f:
        f.create_dataset('mask', data=mask)
        f.close()
    
    # Extract and convert perspective map
    pmap_file = os.path.join( notation_folder, 'vidf1_33_dmap3.mat' )
    p_obj = scio.loadmat(pmap_file, struct_as_record=False, squeeze_me=True)
    pmap = p_obj['dmap'].pmapx
    
    # Scale pmap
    pmap = pmap/pmap.min()
    
    mask_opt_fname = os.path.join( output, 'ucsd_pmap_min_norm.h5' )
    with h5py.File(mask_opt_fname, 'w') as f:
        f.create_dataset('pmap', data=pmap)
        f.close()
    
    return 0

if __name__ == '__main__':
    main(sys.argv[1:])
