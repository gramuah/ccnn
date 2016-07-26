# -*- coding: utf-8 -*-

'''
    @author: Daniel Oñoro Rubio
    @contact: daniel.onoro@edu.uah.es
    @date: 27/02/2015
'''
#===========================================================================
# Dependency loading
#===========================================================================

# File storage
import h5py
import scipy.io as sio

# System
import signal
import sys, getopt
signal.signal(signal.SIGINT, signal.SIG_DFL)
import time
import glob
# Vision and maths
import numpy as np
from math import floor
from utils import *
from gen_features import genDensity, loadImage, zoomSingleIm
import caffe
from sklearn.ensemble import ExtraTreesRegressor

# Plotting
import matplotlib.pyplot as plt

import cv2

#===========================================================================
# Code 
#===========================================================================
class CaffePredictor:
    def __init__(self, prototxt, caffemodel, meanimg):
        # Use CPU
#        caffe.set_mode_cpu()
        # Use GPU
        caffe.set_mode_gpu()
        
        # Load a precomputed caffe model
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        
        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data_s0'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1)) # It's already RGB
#         self.transformer.set_mean('data', np.load(meanimg).mean(1).mean(1))  # mean pixel
#         self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
#         self.transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
        # Reshape net for the single input
        b_shape = self.net.blobs['data_s0'].data.shape
        self.net.blobs['data_s0'].reshape(b_shape[0],b_shape[1],b_shape[2],b_shape[3])
        self.net.blobs['data_s1'].reshape(b_shape[0],b_shape[1],b_shape[2],b_shape[3])

    # take an array of shape (n, height, width) or (n, height, width, channels)
    # and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    def vis_square(self, data, padsize=1, padval=0):
        data -= data.min()
        data /= data.max()
        
        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
        
        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        
        plt.imshow(data)
        plt.show()


    # Probably it is not the eficient way to do it...
    def process(self, im, base_pw):
        # Compute dense positions where to extract patches
        [heith, width] = im.shape[0:2]
        pos = get_dense_pos(heith, width, base_pw, stride=10)

        # Initialize density matrix and vouting count
        dens_map = np.zeros( (heith, width), dtype = np.float32 )   # Init density to 0
        count_map = np.zeros( (heith, width), dtype = np.int32 )     # Number of votes to divide
        
        # draw patches size
        rois = np.zeros( (heith, width, 3), np.uint8 ) 
        
        # Iterate for all patches
        for p in pos:
            # Compute displacement from centers
            dx=dy=int(base_pw/2)
    
            x,y=p
            sx=slice(x-dx,x+dx+1,None)
            sy=slice(y-dy,y+dy+1,None)
            
            crop_im=im[sx,sy,...]
    
            # Draw squares
            color = np.random.randint(0,255,size = 3)
            cv2.rectangle(rois, (y-dy, x-dx), (y+dy, x+dx), color, 1)
    
    #         print crop_im.shape
            h, w = crop_im.shape[0:2]
    
            if h!=w or (h<=0):
    #             print "Patch out of boundaries: ", h, w
                continue
            
            zoom_im = zoomSingleIm(crop_im)
            
            # Load and forward CNN
            self.net.blobs['data_s0'].data[...] = self.transformer.preprocess('data', crop_im.copy())
            self.net.blobs['data_s1'].data[...] = self.transformer.preprocess('data', zoom_im.copy())
            self.net.forward()
            
#             print self.net.blobs['data'].data[0].shape
#             plt.imshow(self.net.blobs['data'].data[0,0], cmap='gray')
#             plt.show()
            
#             plt.subplot(1,2,1)
#             plt.imshow(self.transformer.deprocess('data', self.net.blobs['data'].data[0]))
#             plt.subplot(1,2,2)
#             self.vis_square(self.net.blobs['conv1'].data[0, :36], padsize=1, padval=0)

            # Take the output from the last layer
            # Access to the last layer of the net, second element of the tuple (layer, caffe obj)
            pred = self.net.blobs.items()[-1][1].data
#             pred = np.ones( 18*18 ) / 324.0 * pred

#             pred = self.net.blobs['fc6'].data
            
            # Make it squared
            p_side = int(np.sqrt( len( pred.flatten() ) )) 
            pred = pred.reshape(  (p_side, p_side) )
            
            # Resize it back to the original size
            pred = resizeDensityPatch(pred, crop_im.shape[0:2])
            
            pred[pred<0] = 0
        
#             print pred.sum()
        
#             print pred.sum()
            # Sumup density map into density map and increase count of votes
            dens_map[sx,sy] += pred
            count_map[sx,sy] += 1

#         plt.imshow(dens_map)
#         plt.show()

#         plt.imshow(rois)
#         plt.show()

        # Remove Zeros
        count_map[ count_map == 0 ] = 1

        # Average density map
        dens_map = dens_map / count_map        
         
#         plt.subplot(1,2,1)
#         plt.imshow(dens_map)
#         plt.subplot(1,2,2)
#         plt.imshow(count_map)
#         plt.show()        
        
        return dens_map
        
'''
    @brief: Compute the game metric error. Recursive function.
    @param test: test density.
    @param gt: ground truth density.
    @param cur_lvl: current game level.
    @param tar_lvl: target game level.
    @return game: return game metric.
'''
def gameRec(test, gt, cur_lvl, tar_lvl):
    
    # get sizes
    dim = test.shape
    
    assert dim == gt.shape
    
    if cur_lvl == tar_lvl:
        return np.abs( np.sum( test ) - np.sum( gt ) )
    else:

        # Creating the four slices
        y_half = int( dim[0]/2 )
        x_half = int( dim[1]/2 )
        
        dens_slice = []
        dens_slice.append( test[ 0:y_half, 0:x_half ] )
        dens_slice.append( test[ 0:y_half, x_half:dim[1] ] )
        dens_slice.append( test[ y_half:dim[0], 0:x_half] )
        dens_slice.append( test[ y_half:dim[0], x_half:dim[1] ] )

        gt_slice = []
        gt_slice.append( gt[ 0:y_half, 0:x_half ] )
        gt_slice.append( gt[ 0:y_half, x_half:dim[1] ] )
        gt_slice.append( gt[ y_half:dim[0], 0:x_half] )
        gt_slice.append( gt[ y_half:dim[0], x_half:dim[1] ] )

        res = np.zeros(4)
        for a in range(4):
            res[a] = gameRec(dens_slice[a], gt_slice[a], cur_lvl + 1, tar_lvl)
    
        return np.sum(res);      

'''
    @brief: Compute the game metric error.
    @param test: test density.
    @param gt: ground truth density.
    @param lvl: game level. lvl = 0 -> mean absolute error.
    @return game: return game metric.
'''
def gameMetric(test, gt, lvl):
    return gameRec(test, gt, 0, lvl)        

#===========================================================================
# Some helpers functions
#===========================================================================
def testOnImg(CNN, im, gtdots, pw, mask = None):
    
    # Process Image
    resImg = CNN.process(im, pw) 

    # Mask image if provided
    if mask != None:
        resImg = resImg * mask
        gtdots = gtdots * mask

    npred=resImg.sum()
    ntrue=gtdots.sum()

    # Visualize
#    plt.subplot(1,3,1)
#    plt.imshow(im)
#    plt.title("Raw")
#    plt.subplot(1,3,2)
#    plt.imshow(resImg)
#    plt.title("Prediction")
#    plt.subplot(1,3,3)
#    plt.imshow(gtdots)
#    plt.title("Ground truth")
#    plt.show()

    return ntrue,npred,resImg,gtdots

def dispHelp(arg0):
    print "======================================================"
    print "                       Usage"
    print "======================================================"
    print "\t-h display this message"
    print "\t--tfeat <test features file>"
    print "\t--imfolder <test image folder>"
    print "\t--timnames <test image names txt file>"
    print "\t--dending <dot image ending pattern>"
    print "\t--usemask <flag to use masks>"
    print "\t--mask <mask path (h5 file)>"
    print "\t--model <random forest model to store>"
    print "\t--pw <patch size. Default 7>"
    print "\t--nr <number of patches per image. Default 500>"
    print "\t--sig <sigma for the density images. Default 2.5>"
    print "\t--prototxt <caffe prototxt file>"
    print "\t--caffemodel <caffe caffemodel file>"
    print "\t--meanim <mean image npy file>"
    print "\t--pmap <perspective file>"


def main(argv):      
    # Set default values
    # Use mask
    use_mask = False
    
    # Mask pattern ending
    mask_ending = 'mask.mat'
    
    # Img patterns ending
    dot_ending = 'dots.png'
    
    # Test vars
    test_names_file = './data/TRANCOS/image_sets/quick_validation.txt'
    
    # Im folder
    im_folder = './data/TRANCOS/images/'
    
    # Batch size
    b_size = -1

    # CNN vars
    prototxt_path = './configs/cnn_trancos/tmp_deploy.prototxt'
    caffemodel_path = './genfiles/models/sig_trancos_cnn_iter_1000.caffemodel'
    mean_im_path = './genfiles/trancos_mean.npy'

    # Patch parameters
    pw=115 # Patch with 
    sigmadots = 15.0 # Densities sigma
    mx_game = 4 # Max game target
        
    # Get parameters
    try:
        opts, _ = getopt.getopt(argv, "h:", ["imfolder=", "timnames=", 
            "dending=", "pw=", "sig=", "usemask", "mask=",
            "prototxt=", "caffemodel=", "meanim="])
    except getopt.GetoptError as err:
        print "Error while parsing parameters: ", err
        dispHelp(argv[0])
        return
    
    for opt, arg in opts:
        if opt == '-h':
            dispHelp(argv[0])
            return
        elif opt in ("--imfolder"):
            im_folder = arg
        elif opt in ("--timnames"):
            test_names_file = arg
        elif opt in ("--dending"):
            dot_ending = arg
        elif opt in ("--mask"):
            mask_ending = arg
        elif opt in ("--usemask"):
            use_mask = True
        elif opt in ("--pw"):
            pw = int(arg)
        elif opt in ("--sig"):
            sigmadots = float(arg)
        elif opt in ("--prototxt"):
            prototxt_path = arg
        elif opt in ("--caffemodel"):
            caffemodel_path = arg
        elif opt in ("--meanim"):
            mean_im_path = arg
    
    print "Choosen parameters:"
    print "-------------------"
    print "Test data base location: ", im_folder
    print "Test inmage names: ", test_names_file
    print "Dot image ending: ", dot_ending
    print "Use mask: ", use_mask
    print "Mask path: ", mask_ending
    print "Patch width (pw): ", pw
    print "Sigma for each dot: ", sigmadots
    print "Prototxt path: ", prototxt_path
    print "Caffemodel path: ", caffemodel_path
    print "Mean image path: ", mean_im_path
    print "Batch size: ", b_size
    print "==================="
    
    print "----------------------"
    print "Preparing for Testing"
    print "======================"

    print "Reading image file names:"
    im_names = np.loadtxt(test_names_file, dtype='str')

    # Perform test
    ntrueall=[]
    npredall=[]
    
    # Init GAME
    n_im = len( im_names )
    game_table = np.zeros( (n_im, mx_game) )
    
    # Init CNN
    CNN = CaffePredictor(prototxt_path, caffemodel_path, mean_im_path)
    
    print 
    print "Start prediction ..."
    count = 0
    gt_vector = np.zeros((len(im_names)))
    pred_vector = np.zeros((len(im_names)))

    for ix, name in enumerate(im_names):
        # Get image paths
        im_path = extendName(name, im_folder)
        dot_im_path = extendName(name, im_folder, use_ending=True, pattern=dot_ending)

        # Read image files
        im = loadImage(im_path, color = True)
        dot_im = vigra.impex.readImage(dot_im_path).view(np.ndarray).swapaxes(0, 1).squeeze()
        
        # Generate features
        dens_im = genDensity(dot_im, sigmadots)
        
        mask = None
        if use_mask:
            mask_im_path = extendName(name, im_folder, use_ending=True, pattern=mask_ending)
            mask = sio.loadmat(mask_im_path, chars_as_strings=1, matlab_compatible=1)
            mask = mask.get('BW')
        
        s=time.time()
        ntrue,npred,resImg,gtdots=testOnImg(CNN, im, dens_im, pw, mask)
        print "image : %d , ntrue = %.2f ,npred = %.2f , time =%.2f sec"%(count,ntrue,npred,time.time()-s)
    
#        plt.subplot(1,3,1)
#        plt.imshow(im)
#        plt.title("Input Image", color='white')
#        plt.axis('off')
#        plt.subplot(1,3,2)
#        plt.imshow(resImg)
#        plt.title("Hydra pred.= %.1f"%npred, color='white')
#        plt.axis('off')
#        plt.subplot(1,3,3)
#        plt.imshow(gtdots)
#        plt.title("Ground truth = %.1f"%ntrue, color='white')
#        plt.axis('off')
#        
##        plt.show()
#        plt.savefig('genfiles/images/%.3d.png'%(ix), bbox_inches='tight', facecolor='black')

        gt_vector[ix] = ntrue
        pred_vector[ix] = npred
    
    
        # Hold predictions and originasl
        ntrueall.append(ntrue)
        npredall.append(npred)
        
        # Compute game metric
        for l in range(mx_game):
            game_table[count, l] = gameMetric(resImg, gtdots, l)
    
#         print "Game: ", game_table[count, :]
        count = count +1
            
    ntrueall=np.asarray(ntrueall)
    npredall=np.asarray(npredall)
    print "done ! mean absolute error %.2f" % np.mean(np.abs(ntrueall-npredall))

    # Print Game results
    for l in range(mx_game):
        print "GAME for level %d: %.2f " % (l, np.mean( game_table[:,l] ))
    
    np.save('up_gt2.npy',gt_vector)
    np.save('up_pred2.npy',pred_vector)
    
    return game_table.mean(axis=0)

if __name__=="__main__":
    main(sys.argv[1:])
