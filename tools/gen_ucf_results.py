#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    @author: Daniel Oñoro Rubio
    @organization: GRAM, University of Alcalá, Alcalá de Henares, Spain
    @copyright: See LICENSE.txt
    @contact: daniel.onoro@edu.uah.es
    @date: Created on August 11, 2016
'''

"""
This script read the obtained results of each fold for the UCF experiment and 
prints its MAE and MSD error.
"""

import _init_paths
import numpy as np
import sys, getopt
import utils as utl


def initTestFromCfg(cfg_file):
    '''
    @brief: initialize all parameter from the cfg file. 
    '''
    
    # Load cfg parameter from yaml file
    cfg = utl.cfgFromFile(cfg_file)
    
    # Fist load the dataset name
    dataset = cfg.DATASET
   
    # Results output foder
    results_file = cfg[dataset].RESULTS_OUTPUT
        
    return (dataset, results_file)


def dispHelp(arg0):
    print "======================================================"
    print "                       Usage"
    print "======================================================"
    print "\t-h display this message"
    print "\t--cfg <config file yaml>"

def main(argv):
    UCF_K = 5
    
    cfg_file = ""

    # Get parameters
    try:
        opts, _ = getopt.getopt(argv, "h:", ["cfg="])
    except getopt.GetoptError as err:
        print "Error while parsing parameters: ", err
        dispHelp(argv[0])
        return
    
    for opt, arg in opts:
        if opt == '-h':
            dispHelp(argv[0])
            return
        elif opt in ("--cfg"):
            cfg_file = arg

    # Read ucf results prefix
    results_prefix = initTestFromCfg(cfg_file)

    # MAE and STD vectors
    mae_v = np.zeros(UCF_K)
    std_v = np.zeros(UCF_K)
    for i in range(UCF_K):
        gt_file = results_prefix + "{}_gt.npy".format(i)
        pred_file = results_prefix + "{}_pred.npy".format(i)
        
        gt = np.load(gt_file)
        pred = np.load(pred_file)
        
        diff = gt - pred
        
        print gt
        print pred
        
        mae_v[i] = np.mean( np.abs( diff ) )
        std_v[i] = np.std(diff)
    
    print "Total results:"
    print "MAE: ", mae_v.mean()
    print "STD: ", std_v.mean()
    
    return 0

if __name__ == '__main__':
    main(sys.argv[1:])
