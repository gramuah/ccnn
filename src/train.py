# -*- coding: utf-8 -*-

'''
    @author: Daniel Oñoro Rubio
    @contact: daniel.onoro@edu.uah.es
    @date: 08/07/2015
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

# Vision and maths
import numpy as np
from caffe import layers as L
from caffe import params as P
import caffe
from skimage.transform import resize

# Others
import utils
import test as tst

# Plotting
# import matplotlib.pyplot as plt
from pylab import *


#===========================================================================
# Code 
#===========================================================================

def initSolver(train_val_file, w_decay):
    
    proto = """net: "%s"
test_iter: 100
test_interval: 200
base_lr: 0.0005
lr_policy: "inv"
power: 0.75
gamma: 0.001
stepsize: 100
display: 20
max_iter: 500
momentum: 0.9
weight_decay: %.5f
snapshot: 500
snapshot_prefix: "genfiles/models/sig_trancos_cnn"
solver_mode: GPU"""%(train_val_file,w_decay)

    return proto

def initCNN(g_sig_list):
    
    proto_str = """name: "UCSD_CNN"
layer {
  name: "data"
  type: "HDF5Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  hdf5_data_param {
    source: "genfiles/data/quick_train.txt"
    batch_size: 64
    #shuffle: true
  }
}
layer {
  name: "data"
  type: "HDF5Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  hdf5_data_param {
    source: "genfiles/data/quick_test.txt"
    batch_size: 64
    #shuffle: true
  }
}
# Convolutional Layers
################################################################################
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 32
    pad: 3
    kernel_size: 7
    stride: 1
    weight_filler {
      type: "gaussian"
      std: %.5f
    }

    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}

layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 32
    pad: 3
    kernel_size: 7
    stride: 1
    weight_filler {
      type: "gaussian"
      std: %.5f
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "conv2"
  top: "conv2"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}

layer {
  name: "conv3"
  type: "Convolution"
  bottom: "pool2"
  top: "conv3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    kernel_size: 5
    stride: 1
    pad: 2
    weight_filler {
      type: "gaussian"
      std: %.5f
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu3"
  type: "ReLU"
  bottom: "conv3"
  top: "conv3"
}

layer {
  name: "fc4"
  type: "Convolution"
  bottom: "conv3"
  top: "fc4"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 1000
    kernel_size: 1
    stride: 1
    pad: 0
    weight_filler {
      type: "gaussian"
      std: %.5f
    }
    bias_filler {
      type: "constant"
#      value: 1
    }
  }
}
layer {
  name: "relu4"
  type: "ReLU"
  bottom: "fc4"
  top: "fc4"
}

layer {
  name: "fc5"
  type: "Convolution"
  bottom: "fc4"
  top: "fc5"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 400
    kernel_size: 1
    stride: 1
    pad: 0
    weight_filler {
      type: "gaussian"
      std: %.5f
    }
    bias_filler {
      type: "constant"
#      value: 1
    }
  }
}
layer {
  name: "relu5"
  type: "ReLU"
  bottom: "fc5"
  top: "fc5"
}

layer {
  name: "fc6"
  type: "Convolution"
  bottom: "fc5"
  top: "fc6"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 1
    kernel_size: 1
    stride: 1
    pad: 0
    weight_filler {
      type: "constant"
      value: 0
    }
    bias_filler {
      type: "constant"
#      value: 1
    }
  }
}

layer {
  name: "loss"
  type: "EuclideanLoss"
  bottom: "fc6"
  bottom: "label"
  top: "loss"
}

"""%(g_sig_list[0],g_sig_list[1],g_sig_list[2],g_sig_list[3],g_sig_list[4])

    return proto_str

def dispHelp():
    print "======================================================"
    print "                       Usage"
    print "======================================================"
    print "\t-h display this message"
    print "\t--tfeat <training features file>"
    print "\t--vfeat <validation features files>"


def euloss(bottom0, bottom1):
    diff = bottom0.data - bottom1.data.reshape( bottom0.data.shape )
    loss = np.sum(diff**2) / bottom0.num / 2.
    
    return loss

def euloss2(bottom0, bottom1):
    sum_gt = bottom1.data.sum(axis = 1).sum(axis = 1)
    diff = bottom0.data - sum_gt.reshape( bottom0.data.shape )
    loss = np.sum(diff**2) / bottom0.num / 2.
    
    return loss

def train_cnn(train_feat_path, val_feat_path, solver_path):
    
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_path)
    
    # Let's check whether the training works    
    solver.step(25)  # SGD by Caffe

    # store the train loss
    loss = solver.net.blobs['loss'].data
    
    print loss
    
    # If CNN explode
    if loss == float('Inf'):
        return False
    
    if np.isnan(loss):
        return False
    
    # Fully solve it 
    solver.solve()
    
    return True

def main(argv):      
    # Features file
    train_feat_path = './genfiles/train_cnn_feat.h5'
    val_feat_path = './genfiles/val_cnn_feat.h5'
    
    # Solver path
    solver_path = './configs/cnn_trancos/solver.prototxt'
    
    # Opt data path
    opt_file = "/home/dani/training_data.h5"
    tmp_train_val = "./configs/cnn_trancos/tmp_train_val.prototxt"
    tmp_solver = "./configs/cnn_trancos/tmp_solver.prototxt"
    
    # Get parameters
    try:
        opts, _ = getopt.getopt(argv, "h:", ["tfeat=", "vfeat=", "solver="])
    except getopt.GetoptError as err:
        print "Error while parsing parameters: ", err
        dispHelp()
        return
    
    for opt, arg in opts:
        if opt == '-h':
            dispHelp(argv[0])
            return
        elif opt in ("--tfeat"):
            train_feat_path = arg
        elif opt in ("--vfeat"):
            val_feat_path = arg
        elif opt in ("--solver"):
            solver_path = arg
    
    print "Choosen parameters:"
    print "-------------------"
    print "Training precomputed features: ", train_feat_path
    print "Validation precomputed features: ", val_feat_path
    print "Solver file: ", solver_path
    print "==================="

    # Generate Solver File
    solver_proto = initSolver(tmp_train_val, 0.001)
    with open(tmp_solver,'w') as f:
        f.write(solver_proto)
        f.close()
    
    # Permute
    gaus_sig_mat = utils.cartesian( ( [0.1, 0.01, 0.001, 0.0001], [0.3, 0.2, 0.01, 0.001], [0.3, 0.2, 0.01, 0.001], [0.3, 0.2, 0.01, 0.001], [0.1, 0.05, 0.2] ) )
    it_saver = 5
    # Repeat and collect results
    if(os.path.isfile('status.npy')):
        start = np.load('status.npy')
        results = np.load('results.npy')
        print "Starting at: ", start
    else:
        start = 0
        results = np.zeros( gaus_sig_mat.shape[0] )

    
    # Repeat and collect results
    for i in range( start, gaus_sig_mat.shape[0] ):
        
        sig_v = gaus_sig_mat[i]
         
        print "Preparing for:", sig_v
         
        train_val_proto = initCNN( sig_v )
        with open(tmp_train_val,'w') as f:
            f.write(train_val_proto)
            f.close()
          
        if not train_cnn(train_feat_path, val_feat_path, tmp_solver):
            print "Such CNN, very explode!!!!!!!!!!!!!!!!!!!!"
            continue
         
        # Test
        game_err = tst.main([])
        print game_err.shape
        results[i] = game_err[3]
        
        print "Game: ", results[i]
        
        if i % it_saver == 0:
            print "Saving Status..."
            np.save('results.npy', results)
            np.save('status.npy', i)
        
    np.save('results.npy', results)
    
    print "The end"

if __name__ == "__main__":
    main(sys.argv[1:])