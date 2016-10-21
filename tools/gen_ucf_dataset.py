#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    @author: Daniel Oñoro Rubio
    @organization: GRAM, University of Alcalá, Alcalá de Henares, Spain
    @copyright: See LICENSE.txt
    @contact: daniel.onoro@edu.uah.es
    @date: Created on Jun 15, 2015
'''

"""
This script creates the defferent test/train sets
"""

import sys, getopt, os
import glob
import numpy as np

def dispHelp(arg0):
    print "======================================================"
    print "                       Usage"
    print "======================================================"
    print "\t-h display this message"
    print "\t--orderfile <ordered image name file>"
    print "\t--setsfolder <where to save the set list>"

def writeToFile(list, file):
    with open(file, 'w') as f:
        for name in list:
            f.write(name)
        
        f.close()

def main(argv):
    conf_folder = './image_sets/'
    order_file =  './images/'
    
    # UCF constans
    K = 5
    UDF_SIZE = 50
    TEST_SIZE = UDF_SIZE/K
    
    # Get parameters
    try:
        opts, _ = getopt.getopt(argv, "h:", ["orderfile=", "setsfolder="])
    except getopt.GetoptError as err:
        print "Error while parsing parameters: ", err
        dispHelp(argv[0])
        return
    
    for opt, arg in opts:
        if opt == '-h':
            dispHelp(argv[0])
            return
        elif opt in ("--orderfile"):
            order_file = arg
        elif opt in ("--setsfolder"):
            conf_folder = arg

    # Read file
    with open(order_file,'r') as f:
        names_list = [name for name in f.readlines()]
        names_list = np.array(names_list) # Cast to numpy for a better indexing
        
    # Perform divisións
    for i in range(K):
        # Get test sets
        test_idx = np.arange(i*TEST_SIZE,(i+1)*TEST_SIZE)
        train_mask = np.ones(UDF_SIZE, dtype=np.bool)
        train_mask[test_idx] = False
        train_ix = np.where(train_mask)
        
        # Write test file
        tf_name = os.path.join( conf_folder, 'test_set_{}.txt'.format(i) )
        writeToFile(names_list[test_idx], tf_name)
        
        # Write train file
        tf_name = os.path.join( conf_folder, 'train_set_{}.txt'.format(i) )
        writeToFile(names_list[train_ix], tf_name)
  
    
    return 0

if __name__ == '__main__':
    main(sys.argv[1:])
