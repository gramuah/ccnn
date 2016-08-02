# -*- coding: utf-8 -*-

'''
@date: Created on Jun 15, 2015

@author: Daniel Oñoro Rubio
'''

import sys, getopt
import glob
import numpy as np

def dispHelp(arg0):
    print "======================================================"
    print "                       Usage"
    print "======================================================"
    print "\t-h display this message"
    print "\t--cpu_only"
    print "\t--tdev <GPU ID>"
    print "\t--prototxt <caffe prototxt file>"
    print "\t--caffemodel <caffe caffemodel file>"
    print "\t--cfg <config file yaml>"


def main(argv):
    conf_folder = './image_sets/'
    img_folder =  './images/'

    # Get parameters
    try:
        opts, _ = getopt.getopt(argv, "h:", ["imfolder=", "setsfolder="])
    except getopt.GetoptError as err:
        print "Error while parsing parameters: ", err
        dispHelp(argv[0])
        return
    
    for opt, arg in opts:
        if opt == '-h':
            dispHelp(argv[0])
            return
        elif opt in ("--imfolder"):
            conf_folder = arg
        elif opt in ("--setsfolder"):
            img_folder = arg

    # Get UCSD image names    
    image_names = glob.glob(img_folder + '*[0-9].png')
    # Sort frames by name
    image_names = sorted(image_names)
    
    n_imgs = len(image_names)
    
    # Generate Maximal dataset
    # Traning
    training_f = open(conf_folder + 'training_maximal.txt','w')
    for ix in range(600, 1400+5,5):
        slash_pos = image_names[ix].rfind('/')
        name = image_names[ix][slash_pos + 1 : ]
        print name
        training_f.write( name + '\n' )
    training_f.close()
     
    # Testing
    testing_f = open(conf_folder + 'testing_maximal.txt','w')
    for ix in range(600, 1400+5):
        # Check if it belongs to training set
        if ix % 5 == 0:
            continue
        
        slash_pos = image_names[ix].rfind('/')
        name = image_names[ix][slash_pos + 1 : ]
        print name
        testing_f.write( name + '\n' )
    testing_f.close()
    
    
    # Generate Downscale dataset
    # Traning
    training_f = open(conf_folder + 'training_downscale.txt','w')
    for ix in range(1205, 1600+5,5):
        slash_pos = image_names[ix].rfind('/')
        name = image_names[ix][slash_pos + 1 : ]
        print name
        training_f.write( name + '\n' )
    training_f.close()
     
    # Testing
    testing_f = open(conf_folder + 'testing_downscale.txt','w')
    for ix in range(1205, 1600+5):
        # Check if it belongs to training set
        if ix % 5 == 0:
            continue
        
        slash_pos = image_names[ix].rfind('/')
        name = image_names[ix][slash_pos + 1 : ]
        print name
        testing_f.write( name + '\n' )
    testing_f.close()
    
    # Generate Upscale dataset
    # Traning
    training_f = open(conf_folder + 'training_upscale.txt','w')
    for ix in range(805, 1100+5,5):
        slash_pos = image_names[ix].rfind('/')
        name = image_names[ix][slash_pos + 1 : ]
        print name
        training_f.write( name + '\n' )
    training_f.close()
     
    # Testing
    testing_f = open(conf_folder + 'testing_upscale.txt','w')
    for ix in range(805, 1100+5):
        # Check if it belongs to training set
        if ix % 5 == 0:
            continue
        
        slash_pos = image_names[ix].rfind('/')
        name = image_names[ix][slash_pos + 1 : ]
        print name
        testing_f.write( name + '\n' )
    testing_f.close()
 
     # Generate Minimal dataset
    # Traning
    training_f = open(conf_folder + 'training_minimal.txt','w')
    for ix in range(640, 1360+80,80):
        slash_pos = image_names[ix].rfind('/')
        name = image_names[ix][slash_pos + 1 : ]
        print name
        training_f.write( name + '\n' )
    training_f.close()
     
    # Testing
    testing_f = open(conf_folder + 'testing_minimal.txt','w')
    for ix in range(640, 1360+80):
        # Check if it belongs to training set
        if ix % 80 == 0:
            continue
        
        slash_pos = image_names[ix].rfind('/')
        name = image_names[ix][slash_pos + 1 : ]
        print name
        testing_f.write( name + '\n' )
    testing_f.close()
    
    
    return 0

if __name__ == '__main__':
    main(sys.argv[1:])
