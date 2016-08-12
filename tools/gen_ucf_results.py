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

import numpy as np
import sys, getopt


def dispHelp(arg0):
    print "======================================================"
    print "                       Usage"
    print "======================================================"
    print "\t-h display this message"
    print "\t--results <config file yaml>"

def main(argv):
    UCF_K = 5
    
    cfg_file = ""

    # Get parameters
    try:
        opts, _ = getopt.getopt(argv, "h:", ["results="])
    except getopt.GetoptError as err:
        print "Error while parsing parameters: ", err
        dispHelp(argv[0])
        return
    
    for opt, arg in opts:
        if opt == '-h':
            dispHelp(argv[0])
            return
        elif opt in ("--results"):
            results_prefix = arg

    # MAE and STD vectors
    mae_v = np.zeros(UCF_K)
    std_v = np.zeros(UCF_K)
    for i in range(UCF_K):
        # Read data
        gt_file = results_prefix + "{}_gt.txt".format(i)
        pred_file = results_prefix + "{}_pred.txt".format(i)
        gt = np.loadtxt(gt_file)
        pred = np.loadtxt(pred_file)

        # Compute error
        diff = gt - pred
        mae_v[i] = np.mean( np.abs( diff ) )
        std_v[i] = np.std(diff)
    
    print "Total results:"
    print "MAE: ", mae_v.mean()
    print "STD: ", std_v.mean()
    
    return 0

if __name__ == '__main__':
    main(sys.argv[1:])
