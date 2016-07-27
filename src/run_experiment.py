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
import test as tst
import gen_features as gfeat

# Plotting
# import matplotlib.pyplot as plt
from pylab import *
from dns.rdatatype import SIG


def main(argv):      
    # Paths
    DATA='data/TRANCOS/'
    GENFILES='genfiles/'
    CONFIGS='configs/cnn_trancos/'
    
    # Generated files
    TRAIN_FEAT='data/trancos_train_cnn_feat'
    VAL_FEAT='data/trancos_val_cnn_feat'
    
    # Patterns
    DOT_ENDING="dots.png"
    MASK_ENDING="mask.mat"
    
    # Choose one
    USE_MASK="--usemask"
#     USE_MASK=""

    # Params
    PW='115'            # Base patch side
    NR='800'            # < 1 = dense extraction
    SIG='15.0'        
    SPLIT='75'        # Split every 25 images into a new file
    FLIP='--flip'
#     FLIP=""
    
    # CNN Files
    CNN_PW_IN='72'    # CNN patch width in
    CNN_PW_OUT='18'     # CNN patch width out
    SOLVER='hidra_solver.prototxt'
    PROTOTXT='hidra_deploy.prototxt'
    CAFFEMODEL='best/trancos_cnn_iter_30000.caffemodel_1'
    
    status_file = 'hidra_status2.npy'
    game_file = 'game2.npy'
    n_exp = 10
    
    # Repeat and collect results
    if(os.path.isfile(status_file)):
        start = np.load(status_file) + 1 # I want to start the next itt hence I sum 1
        game = np.load(game_file)
        print "Starting at: ", start
    else:
        start = 0
        game = np.zeros( (n_exp, 4) )
    
    for i in range(start, n_exp):
        # Generate Training Features
        gen_feat_params = ['--imfolder', DATA + '/images', '--output', GENFILES + '/'+ TRAIN_FEAT,
                           '--names', DATA + '/image_sets/training.txt',
                           '--ending', DOT_ENDING, '--pw_base', PW, '--pw_norm',
                           CNN_PW_IN, '--pw_dens',CNN_PW_OUT, '--sig', SIG, '--nr', NR,
                           '--split', SPLIT, FLIP]
        gfeat.main( gen_feat_params )
        training_data_files = glob.glob(GENFILES + '/' + TRAIN_FEAT + '*.h5')
        with open(GENFILES + '/data/aux_train.txt','w') as f:
            for fname in training_data_files:
                f.write(fname + '\n')
            f.close()
            
        # Generate Training Features
        gen_feat_params = ['--imfolder', DATA + '/images', '--output', GENFILES + '/'+ VAL_FEAT,
                           '--names', DATA + '/image_sets/validation.txt',
                           '--ending', DOT_ENDING, '--pw_base', PW, '--pw_norm',
                           CNN_PW_IN, '--pw_dens',CNN_PW_OUT, '--sig', SIG, '--nr', NR,
                           '--split', SPLIT, FLIP]
        gfeat.main( gen_feat_params )    
        testing_data_files = glob.glob(GENFILES + '/' + VAL_FEAT + '*.h5')
        with open(GENFILES + '/data/aux_test.txt','w') as f:
            for fname in testing_data_files:
                f.write(fname + '\n')
            f.close()
           
        # All training dataset
        all_train_data_files = training_data_files + testing_data_files
        with open(GENFILES + '/data/train.txt','w') as f:
            for fname in all_train_data_files:
                f.write(fname + '\n')
            f.close()
          
        # Train CNN
        tnt = 10   # Count down
        while not train_cnn(CONFIGS + '/' + SOLVER) and tnt > 0:
            tnt -= 1
             
        if tnt <= 0:
            print "Could not train CNN. It explodes!"
            continue
         
        os.system("cp " + GENFILES + '/' + CAFFEMODEL + ' ' + GENFILES + '/' + CAFFEMODEL + "_" + str(i) )
        
        # Test
        g = tst.main( ['--imfolder', DATA + '/images', '--timnames', DATA + '/image_sets/test.txt',
                   '--dending', DOT_ENDING, USE_MASK, '--mask', MASK_ENDING, '--pw', PW,
                   '--sig', SIG, '--prototxt', CONFIGS + '/' + PROTOTXT,
                    '--caffemodel', GENFILES + '/' + CAFFEMODEL] )
    
        return 0
    
        game[i,...] = g
    
        np.save(game_file, game)
        np.save(status_file, i)
        
    
    print "The end"

def train_cnn(solver_path):
    caffe.set_device(0)
    caffe.set_mode_gpu()
#    caffe.set_mode_cpu()
    solver = caffe.SGDSolver(solver_path)

    # Let's give some iteration to see wheter it explodes
    solver.step(50) 

    # store the train loss
    loss = solver.net.blobs['loss'].data

    # If CNN explode
    if np.isnan(loss) or loss == float('Inf'):
        return False

    # Fully solve it 
    solver.solve()

    return True

if __name__ == "__main__":
    main(sys.argv[1:])
