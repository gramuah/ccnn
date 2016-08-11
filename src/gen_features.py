#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    @author: Daniel Oñoro Rubio
    @organization: GRAM, University of Alcalá, Alcalá de Henares, Spain
    @copyright: See LICENSE.txt
    @contact: daniel.onoro@edu.uah.es
    @date: 26/02/2015
'''

"""
Feature files generator. This code read the specified database and creates all
the necessary files to perform a train.
"""

#===========================================================================
# Dependency loading
#===========================================================================
# System
import sys, getopt

# File storage
import h5py

# Vision and maths
import numpy as np
import utils as utl 
import skimage.io
from scipy.ndimage.filters import gaussian_filter 
from skimage.transform import resize


#===========================================================================
# Code 
#===========================================================================

def genDensity(dot_im, sigmadots):
    '''
    @brief: This function gets a dotted image and returns its density map.
    @param: dots: annotated dot image.
    @param: sigmadots: density radius.
    @return: density map for the input dots image.
    '''
    
    # Take only red channel
    dot = dot_im[:, :, 0]
    dot = gaussian_filter(dot, sigmadots)
        
    return dot.astype(np.float32)

def genPDensity(dot_im, sigmadots, pmap):
    '''
        @brief: This function gets a dotted image and returns its density map 
        scaling responses with the perspective map.
        @param dots: annotated dot image.
        @param sigmadots: density radius.
        @param pmap: perspective map
        @return: density map for the input dots image.
    '''
    
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

def extractEscales(lim, n_scales):
    '''
    @brief: Crop and return n_scaled patches.
    '''
    out_list = []
    for im in lim:
        ph, pw = im.shape[0:2] # get patch width and height
        scaled_im_list = []
        for s in range(n_scales):
            ch = s * ph / (2*n_scales)
            cw = s * pw / (2*n_scales)
            
            crop_im = im[ch:ph-ch, cw:pw - cw]
            
            scaled_im_list.append(resize(crop_im, (ph,pw)))
        
        out_list.append(scaled_im_list)
        
    return out_list

def initGenFeatFromCfg(cfg_file):
    '''
    @brief: initialize all parameter from the cfg file. 
    '''
    
    # Load cfg parameter from yaml file
    cfg = utl.cfgFromFile(cfg_file)
    
    # Fist load the dataset name
    dataset = cfg.DATASET
    
    # Set values
    im_folder = cfg[dataset].IM_FOLDER
    im_list_file = cfg[dataset].TRAINVAL_LIST
    output_file = cfg[dataset].TRAIN_FEAT
    feature_file_path = cfg[dataset].TRAIN_FEAT_LIST
    
    # Img patterns
    dot_ending = cfg[dataset].DOT_ENDING

    # Features extraction vars
    pw_base = cfg[dataset].PW  # Patch width 
    pw_norm = cfg[dataset].CNN_PW_IN  # Patch width
    pw_dens = cfg[dataset].CNN_PW_OUT  # Patch width
    sigmadots = cfg[dataset].SIG  # Densities sigma
    Nr = cfg[dataset].NR  # Number of patches extracted from the compute_mean images
    n_scales = cfg[dataset].N_SCALES        # Number of scales
    
    # Others
    split_size = cfg[dataset].SPLIT
    do_flip = cfg[dataset].FLIP
    use_perspective = cfg[dataset].USE_PERSPECTIVE
    perspective_path = cfg[dataset].PERSPECTIVE_MAP
    is_colored = cfg[dataset].COLOR
    resize_im = cfg[dataset].RESIZE

        
    return (dataset, im_folder, im_list_file, output_file, feature_file_path, 
        dot_ending, pw_base, pw_norm, pw_dens, sigmadots, Nr, n_scales, split_size, 
        do_flip, perspective_path, use_perspective, is_colored, resize_im)

def dispHelp():
    print "======================================================"
    print "                       Usage"
    print "======================================================"
    print "\t-h display this message"
    print "\t--cfg <config file yaml>"
    
def main(argv):
    # Init cfg file
    cfg_file = ''
    
    # Get parameters
    try:
        opts, _ = getopt.getopt(argv, "h:", ["cfg="])
    except getopt.GetoptError:
        dispHelp()
        return
    
    for opt, arg in opts:
        if opt == '-h':
            dispHelp(argv[0])
            return
        elif opt in ("--cfg"):
            cfg_file = arg
            
    print "Loading configuration file: ", cfg_file
    (dataset, im_folder, im_list_file, output_file, feature_file_path, 
    dot_ending, pw_base, pw_norm, pw_dens, sigmadots, Nr, n_scales, split_size, 
    do_flip, perspective_path, use_perspective, is_colored, resize_im) = initGenFeatFromCfg(cfg_file)
    
    print "Choosen parameters:"
    print "-------------------"
    print "Dataset: ", dataset
    print "Data base location: ", im_folder
    print "Image names file: ", im_list_file 
    print "Output file:", output_file
    print "Output feature names file:", feature_file_path
    print "Dot image ending: ", dot_ending
    print "Patch width (pw_base): ", pw_base
    print "Patch width (pw_norm): ", pw_norm
    print "Patch width (pw_dens): ", pw_dens
    print "Number of patches per image: ", Nr
    print "Perspective map: ", perspective_path
    print "Use perspective:", use_perspective
    print "Sigma for each dot: ", sigmadots
    print "Number of scales: ", n_scales
    print "Split size: ", split_size
    print "Flip images: ", do_flip
    print "Resize images: ", resize_im
    print "==================="
    
    print "Reading perspective file"
    if use_perspective:
        pers_file = h5py.File(perspective_path,'r')
        pmap = np.array( pers_file['pmap'] )
        pers_file.close()
    
    print "Creating feature names file:"
    feature_file = open(feature_file_path, 'w')
    feature_file.close() # Create empty file
    
    print "Reading image file names:"
    im_names = np.loadtxt(im_list_file, dtype='str')

    ldens = []
    lpos = []
    lpatches = []
    file_count = 0
    for ix, name in enumerate(im_names):
        print "Processing image: ", name
        # Get image paths
        im_path = utl.extendName(name, im_folder)
        dot_im_path = utl.extendName(name, im_folder, use_ending=True, pattern=dot_ending)
        
        # Read image files
        im = loadImage(im_path, color = is_colored)
        dot_im = loadImage(dot_im_path, color = True)

        # Do ground truth
        if use_perspective:
            dens_im = genPDensity(dot_im, sigmadots, pmap)
        else:
            dens_im = genDensity(dot_im, sigmadots)

        if resize_im > 0:
            # Resize image
            im = utl.resizeMaxSize(im, resize_im)
            gt_sum = dens_im.sum()
            dens_im = utl.resizeMaxSize(dens_im, resize_im)
            dens_im = dens_im * gt_sum / dens_im.sum()

        # Collect features from random locations        
#         height, width, _ = im.shape
#         pos = get_dense_pos(height, width, pw_base=pw_base, stride = 5 )
        pos = utl.genRandomPos(im.shape, pw_base, Nr)

        # Collect original patches
        patch = cropAtPos(im, pos, pw_base)
#         patch = cropPerspective(im, pos, pmap, pw_base)
        
        # Collect dens patches
        dpatch = cropAtPos(dens_im, pos, pw_base)
#         dpatch = cropPerspective(dens_im, pos, pmap, pw_base)
        
        # Resize images
        patch = utl.resizePatches(patch, (pw_norm,pw_norm))
        dpatch = utl.resizeListDens(dpatch, (pw_dens, pw_dens)) # 18 is the output size of the paper

        # Flip function
        if do_flip:
            fpatch = hFlipImages(patch)
            fdpatch = hFlipImages(dpatch)

            fscales = extractEscales(fpatch, n_scales)

            # Add flipped data
            lpatches.append( fscales )
            ldens.append( fdpatch )
            lpos.append( pos )

        
        # Store features and densities 
        ldens.append( dpatch )
        lpos.append(pos)
        lpatches.append( extractEscales(patch, n_scales) )
        
        # Save it into a file
        if split_size > 0 and (ix + 1) % split_size == 0:
            # Prepare for saving
            ldens = np.vstack(ldens)
            lpos = np.vstack(lpos)
            patches_list = np.vstack(lpatches[:])
            
            opt_num_name = output_file + str(file_count) + ".h5"
            print "Saving data file: ", opt_num_name
            print "Saving {} examples".format(len(ldens))
        
            # Compress data and save
            feature_file = open(feature_file_path, 'a')
            comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
            with h5py.File(opt_num_name, 'w') as f:
                f.create_dataset('label', data=ldens, **comp_kwargs)
                
                # Save all scales data
                for s in range(n_scales):
                    dataset_name = 'data_s{}'.format(s) 
                    print "Creating dataset: ", dataset_name
                    f.create_dataset(dataset_name, data=trasposeImages(patches_list[:,s,...]), **comp_kwargs)
                f.close()
            feature_file.write(opt_num_name + '\n')
            feature_file.close()

            # Increase file counter
            file_count += 1

            # Clean memory
            ldens = []
            lpos = []
            lpatches = []
    
    ## Last save
    if len(lpatches) >0:
        # Prepare for saving
        ldens = np.vstack(ldens)
        lpos = np.vstack(lpos)
        patches_list = np.vstack(lpatches[:])
        
        opt_num_name = output_file + ".h5"
        print "Saving data file: ", opt_num_name
        print "Saving {} examples".format(len(ldens))
    
        # Compress data and save
        feature_file = open(feature_file_path, 'a')
        comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
        with h5py.File(opt_num_name, 'w') as f:
            f.create_dataset('label', data=ldens, **comp_kwargs)
            # Save all scales data
            for s in range(n_scales):
                dataset_name = 'data_s{}'.format(s) 
                print "Creating dataset: ", dataset_name
                f.create_dataset(dataset_name, data=trasposeImages(patches_list[:,s,...]), **comp_kwargs)
            f.close()
        feature_file.write(opt_num_name + '\n')
        feature_file.close()
    
    print "--------------------"    
    print "Finish!"
    
if __name__ == "__main__":
    main(sys.argv[1:])