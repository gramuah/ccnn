# -*- coding: utf-8 -*-

'''
    @author: Daniel Oñoro Rubio
    @contact: daniel.onoro@edu.uah.es
    @date: 26/02/2015
'''

#===========================================================================
# Dependency loading
#===========================================================================

# Parallel computing
from joblib import Parallel, delayed  
import multiprocessing
import caffe.io as cio

# System
import sys, getopt
import time

# File storage
import h5py
import scipy.io as sio

# Vision and maths
import vigra
import numpy as np
from utils import *
import skimage.io
from scipy.ndimage.filters import gaussian_filter 
from skimage.transform import resize

# Plotting
import matplotlib.pyplot as plt


#===========================================================================
# Code 
#===========================================================================

'''
    @brief: This function gets a dotted image and returns its density map.
    @param dots: annotated dot image.
    @param sigmadots: density radius.
    @return: density map for the input dots image.
'''
def genDensity(dot_im, sigmadots):
    # Take only red channel
    dot = dot_im[:, :, 0]
    
    # print dot.max(),dot.min()
    dot /= 255
    dot = vigra.filters.gaussianSmoothing(dot, sigmadots).squeeze()
        
    return dot.view(np.ndarray).astype(np.float32)

'''
    @brief: This function gets a dotted image and returns its density map 
    scaling responses with the perspective map.
    @param dots: annotated dot image.
    @param sigmadots: density radius.
    @param pmap: perspective map
    @return: density map for the input dots image.
'''
def genPDensity(dot_im, sigmadots, pmap):
    # Initialize density map
    dmap = np.zeros( (dot_im.shape[0], dot_im.shape[1]), np.float32 )

    # Get notation positions
    pos_list = getGtPos(dot_im)
    for pos in pos_list:
        x,y = pos
        g = 1/pmap[x,y] 
        
        h = np.zeros_like(dmap)
        h[x,y] = 1.0
        h = gaussian_filter( h, sigmadots*g)

        dmap = dmap + h
    
    return dmap

def getGtPos(dot_im):
    '''
    @brief: This function gets a dotted image and returns the ground truth positions.
    @param dots: annotated dot image.
    @return: matrix with the notated object positions.
    '''
    dot = dot_im[:, :, 0]/dot_im[:, :, 0].max()
    
    # Find positions
    pos = np.where(dot == 1)
    
    pos = np.asarray( (pos[0],pos[1]) ).T
    
    return pos

def cropAtPos(im, pos, pw):
    '''
    @brief: Crop patches from im at the position pos with a with of pw.
    @param im: input image.
    @param pos: position list.
    @param pw: patch with.
    @return: returns a list with the patches.    
    '''
    
    dx=dy=pw/2
    
    lpatch = []
    for p in pos:
        x,y=p
        sx=slice(x-dx,x+dx+1,None)
        sy=slice(y-dy,y+dy+1,None)
        
        crop_im=im[sx,sy,...]
        
        lpatch.append(crop_im)                

    return lpatch

def loadImage(filename, color=True):
    """
    Load an image converting from grayscale or alpha as needed.
    Take
    filename: string
    color: flag for color format. True (default) loads as RGB while False
        loads as intensity (if image is already grayscale).
    Give
    image: an image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    """
    img = skimage.img_as_float(skimage.io.imread(filename, as_grey=not color)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img

def computeMeanIm(lim):
    '''
    @brief: This funciton get a list of images and return the mean image.
    @param lim: List of images.
    '''
    mean_im = np.zeros(lim[0].shape, np.float32)
    
    for im in lim:
        mean_im += im
    
    mean_im = mean_im / len(lim)
    
    return mean_im

def trasposeImages(lim):
    '''
    @brief: Traspose all the image of the list lim.
    '''
    opt_list = []
    for i in range( len(lim) ):
        opt_list.append( lim[i].transpose(2,0,1) )
        
    return opt_list

def hFlipImages(lim):
    '''
    @brief: Perform an horizontal flip of the input image list.
    '''
    flim = []
    for im in lim:
        flim.append( np.fliplr(im) )
    
    return flim

def zoomSingleIm(i):
    h,w,_ = i.shape
    crop = i[h/4:3*h/4, w/4:3*w/4]
    return resize(crop, (h,w))
    

def zoomImages(lim):
    '''
    @brief: Crop and return a 50% zoomed image list.
    '''
    opt_list = []
    for i in lim:
        res = zoomSingleIm(i)
        opt_list.append( res.copy() )
        
    return opt_list

def dispHelp():
    print "======================================================"
    print "                       Usage"
    print "======================================================"
    print "\t-h display this message"
    print "\t--imfolder <image folder>"
    print "\t--names <image names txt file>"
    print "\t--ending <dot image ending pattenr>"
    print "\t--output <features file>"
    print "\t--pw_base <base patch width>"
    print "\t--pw_norm <normalize patch width>"
    print "\t--pw_dens <density patch width>"
    print "\t--nr <number of patches per image. Default 500. If it is lower than 1 it will be performed a dense extraction>"
    print "\t--sig <sigma for the density images. Default 2.5>"
    print "\t--split <split the data in files with the specified size>"
    print "\t--flip <add an horizon flipped copy>"

    
def main(argv):
    
    # Set default values
    im_folder = './data/UCSD/images/'
    output_file = './genfiles/train_data'
    im_list_file = './data/UCSD/image_sets/training_downscale.txt'
    
    # Img patterns
    dot_ending = 'dots.png'

    # Features extraction vars
    pw_base = 33  # Patch width 
    pw_norm = 72  # Patch width
    pw_dens = 18  # Patch width
    sigmadots = 4.0  # Densities sigma
    Nr = 800  # Number of patches extracted from the compute_mean images
    
    # Others
    split_size = 5
    do_flip = False
    
    # Get parameters
    try:
        opts, _ = getopt.getopt(argv, "h:", ["imfolder=", "output=", "names=",
          "ending=", "pw_base=", "pw_norm=", "pw_dens=", "sig=", "nr=", "meanim=", "split=", "flip"])
    except getopt.GetoptError:
        dispHelp()
        return
    
    for opt, arg in opts:
        if opt == '-h':
            dispHelp(argv[0])
            return
        elif opt in ("--imfolder"):
            im_folder = arg
        elif opt in ("--output"):
            output_file = arg
        elif opt in ("--names"):
            im_list_file = arg
        elif opt in ("--ending"):
            dot_ending = arg
        elif opt in ("--pw_base"):
            pw_base = int(arg)
        elif opt in ("--pw_norm"):
            pw_norm = int(arg)
        elif opt in ("--pw_dens"):
            pw_dens = int(arg)
        elif opt in ("--nr"):
            Nr = int(arg)
        elif opt in ("--sig"):
            sigmadots = float(arg)
        elif opt in ("--split"):
            split_size = int(arg)
        elif opt in ("--flip"):
            do_flip = True
    
    print "Choosen parameters:"
    print "-------------------"
    print "Data base location: ", im_folder
    print "Image names file: ", im_list_file 
    print "Output file:", output_file
    print "Dot image ending: ", dot_ending
    print "Patch width (pw_base): ", pw_base
    print "Patch width (pw_norm): ", pw_norm
    print "Patch width (pw_dens): ", pw_dens
    print "Number of patches per image: ", Nr
    print "Sigma for each dot: ", sigmadots
    print "Split size: ", split_size
    print "Flip images: ", do_flip
    print "==================="
    
    print "Reading image file names:"
    im_names = np.loadtxt(im_list_file, dtype='str')

    ldens = []
    lpos = []
    lpatches0 = []
    lpatches1 = []
    file_count = 0
    for ix, name in enumerate(im_names):
        print "Processing image: ", name
        # Get image paths
        im_path = extendName(name, im_folder)
        dot_im_path = extendName(name, im_folder, use_ending=True, pattern=dot_ending)
        
        # Read image files
        im = loadImage(im_path, color = True)
        dot_im = vigra.impex.readImage(dot_im_path).view(np.ndarray).swapaxes(0, 1).squeeze()

        # Collect features from random locations
        dens_im = genDensity(dot_im, sigmadots)
        
#         height, width, _ = im.shape
#         pos = get_dense_pos(height, width, pw_base=pw_base, stride = 5 )
        pos = genRandomPos(im.shape, pw_base, Nr)

        # Collect original patches
        patch = cropAtPos(im, pos, pw_base)
#         patch = cropPerspective(im, pos, pmap, pw_base)
        
        # Collect dens patches
        dpatch = cropAtPos(dens_im, pos, pw_base)
#         dpatch = cropPerspective(dens_im, pos, pmap, pw_base)
        
        # Resize images
        patch = resizePatches(patch, (pw_norm,pw_norm))
        dpatch = resizeListDens(dpatch, (pw_dens, pw_dens)) # 18 is the output size of the paper

        # Flip function
        if do_flip:
            fpatch = hFlipImages(patch)
            fdpatch = hFlipImages(dpatch)

            patch_half_flip = zoomImages(patch)
            
            # Add flipped data
            lpatches0.append( fpatch )
            lpatches1.append( patch_half_flip )
            ldens.append( fdpatch )
            lpos.append( pos )

        
        patch_half = zoomImages(patch)
        
#         punrolled = []
#         for p in dpatch:
#             punrolled.append( np.array( p.ravel()[:,np.newaxis,np.newaxis] ) )
         
#         punrolled = np.vstack( punrolled )
        
        # Store features and densities 
#         ldens.append(punrolled)
        ldens.append( dpatch )
        lpos.append(pos)
        lpatches0.append(patch)
        lpatches1.append(patch_half)
        
        # Save it into a file
        if split_size > 0 and (ix + 1) % split_size == 0:
            # Prepare for saving
            ldens = np.vstack(ldens)
            lpos = np.vstack(lpos)
            lpatches0 = np.vstack(lpatches0)
            lpatches1 = np.vstack(lpatches1)

            lpatches0 = trasposeImages(lpatches0)
            lpatches1 = trasposeImages(lpatches1)
            
            opt_num_name = output_file + str(file_count) + ".h5"
            print "Saving data file: ", opt_num_name
            print "Saving {} patches examples of {}".format(len(lpatches0), lpatches0[0].shape)
            print "Saving {} densities examples of {}".format(len(ldens), ldens[0].shape)
        
            # Compress data and save
            comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
            with h5py.File(opt_num_name, 'w') as f:
                f.create_dataset('data_s0', data=lpatches0, **comp_kwargs)
                f.create_dataset('data_s1', data=lpatches1, **comp_kwargs)
                f.create_dataset('label', data=ldens, **comp_kwargs)
                f.close()

            # Increase file counter
            file_count += 1

            # Clean memory
            ldens = []
            lpos = []
            lpatches0 = []
            lpatches1 = []
    
    ## Last save
    if len(lpatches0) >0:
        # Prepare for saving
        ldens = np.vstack(ldens)
        lpos = np.vstack(lpos)
        lpatches0 = np.vstack(lpatches0)
        lpatches1 = np.vstack(lpatches1)
        
        lpatches0 = trasposeImages(lpatches0)
        lpatches1 = trasposeImages(lpatches1)
        
        print "Saving data..."
        print "Saving {} patches examples of {}".format(len(lpatches0), lpatches0[0].shape)
        print "Saving {} densities examples of {}".format(len(ldens), ldens[0].shape)
    
        # Compress data and save
        comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
        with h5py.File(output_file + '.h5', 'w') as f:
            f.create_dataset('data_s0', data=lpatches0, **comp_kwargs)
            f.create_dataset('data_s1', data=lpatches1, **comp_kwargs)
            f.create_dataset('label', data=ldens, **comp_kwargs)
            f.close()
    
    print "--------------------"    
    print "Finish!"
    
if __name__ == "__main__":
    main(sys.argv[1:])
