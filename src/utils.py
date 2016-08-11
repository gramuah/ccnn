import os 
import glob
import vigra
import numpy as np
import scipy.io
from skimage.transform import resize

def resizeMaxSize(im, max_size):
    """
    Resize an image leting its maximun size to max_size without modifing its 
    aspect ratio.
    @param im: input image
    @param max_size: maxinum size
    @return: resized image
    """

    h,w = im.shape[0:2]
    
    im_resized = None
    if w > h:
        s = float(max_size)/w
        im_resized = resize(im, (int(h*s), max_size))
    else:
        s = float(max_size)/h
        im_resized = resize(im, (max_size, int(w*s)))
    
    return im_resized    

def cfgFromFile(filename):
    """Load a config file."""
    import yaml
    from easydict import EasyDict as edict
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    return yaml_cfg

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def get_dense_pos(heith, width, pw, stride = 1):
    '''
    @brief: Generate a dense list of patch position.
    @param heith: image height.
    @param width: image width.
    @param pw: patch with.
    @param stride: stride.
    @return: returns a list with the patches positions.
    '''    
    # Compute patch halfs
    dx=dy=pw/2
    # Create a combination which corresponds to all the points of a dense
    # extraction     
    pos = cartesian( (range(dx, heith - dx, stride), range(dy, width -dy, stride) ) )
#    return pos
    bot_line = cartesian( (heith - dx -1, range(dy, width -dy, stride) ) )
    right_line = cartesian( (range(dx, heith - dx, stride), width -dy - 1) )
    return np.vstack( (pos, bot_line, right_line) )


def cropPerspective(im, pos, pmap, pw):
    '''
    @brief: Crop patches from im at the position pos with a width of pw multiply
    by the perspective map at pmap.
    @param im: input image.
    @param pos: position list.
    @param pw: patch with.
    @return: returns a list with the patches.    
    '''
    
    dx=dy=pw/2
    
    lpatch = []
    for p in pos:
        # Get gain
        g = pmap[p[0],p[1]]
        aux_pw = pw / g
        
        # Diferential
        dx=dy=int(aux_pw/2)

        x,y=p
        sx=slice(x-dx,x+dx+1,None)
        sy=slice(y-dy,y+dy+1,None)
        
        # Compute patch size
        sx_size = sx.stop - sx.start
        sy_size = sy.stop - sy.start
        
        crop_im=im[sx,sy,...]

#         print crop_im.shape
        h, w = crop_im.shape[0:2]

        if h!=w or (h<=0):
#             print "Patch out of boundaries: ", h, w
            continue
        
        lpatch.append(crop_im)                

    return lpatch

def resizeDensityPatch(patch, opt_size):
    '''
    @brief: Take a density map and resize it to the opt_size.
    @param patch: input density map.
    @param opt_size: output size.
    @return: returns resized version of the density map.    
    '''
    # Get patch size
    h, w = patch.shape[0:2]
    
    # Total sum
    patch_sum = patch.sum()

    # Normalize values between 0 and 1. It is in order to performa a resize.    
    p_max = patch.max()
    p_min = patch.min()
    # Avoid 0 division
    if patch_sum !=0:
        patch = (patch - p_min)/(p_max - p_min)
    
    # Resize
    patch = resize(patch, opt_size)
    
    # Return back to the previous scale
    patch = patch*(p_max - p_min) + p_min
    
    # Keep count
    res_sum = patch.sum()
    if res_sum != 0:
        return patch * (patch_sum/res_sum)

    return patch

def resizeListDens(patch_list, psize):
    for ix, patch in enumerate(patch_list):
        # Keep count
        patch_list[ix] = resizeDensityPatch(patch, psize)
            
    return patch_list

def resizePatches(patch_list, psize):
    for ix, patch in enumerate(patch_list):
        # Get patch size
        h, w, _ = patch.shape
        
        # Resize
        patch = resize(patch, psize)
        patch_list[ix] = patch
    
    return patch_list

def genRandomPos(imSize, pw, N):
    ih=imSize[0]
    iw=imSize[1]
    
    dx=dy=pw/2
    
    y=np.random.randint(dy,iw-dy,N).reshape(N,1)
    x=np.random.randint(dx,ih-dx,N).reshape(N,1)
    
    return np.hstack((x,y))

def batch(iterable, n = 1):
    '''
        @brief: Batch an iterable object.
        Example:
            for x in batch(range(0, 10), 3):
            ...     print x
            ... 
            [0, 1, 2]
            [3, 4, 5]
            [6, 7, 8]
            [9]
        @param iterable: iterable object.
        @param n: batch size.
        @return splits: return the iterable object splits
    '''
    
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx+n, l)]


def extendName(name, im_folder, use_ending=False, pattern=[]):
    '''
        @brief: This gets a file name format and adds the root directory and change 
        the extension if needed
        @param fname: file name.
        @param im_folder: im_folder path to add to each file name.
        @param use_ending: flag use to change the file extension.
        @param pattern: string that will substitute the original file ending. 
        @return new_name: list which contains all the converted names.
    '''    
    final_name = im_folder + os.path.sep + name
    
    if use_ending:
        l_dot = final_name.rfind('.')
        final_name = final_name[0:l_dot] + pattern
    
    return final_name

def extendImNames(txt_file, im_folder, use_ending=False, pattern=[]):
    '''
        @brief: This function gets a txt file that contains a list of file names 
        and extend each name with the path to the root folder and/or change the 
        extension.
        @param txt_file: text file which contains a file name list.
        @param im_folder: im_folder path to add to each file name.
        @param use_ending: flag use to change the file extension.
        @param pattern: string that will substitute the original file ending. 
        @return: names_list: list which contains all the converted names.
    '''
    txt_names = np.loadtxt(txt_file, dtype='str')
    
    names = []
    for name in txt_names:
        final_name = extendName(name, im_folder, use_ending, pattern)
        names.append(final_name)
    
    return names

def importImagesFolder(im_names, skip=1, stop=-1, verbose=True):
    '''import all images from a folder that follow a certain pattern'''
    
    count = 0
    imgs = []
    for name in im_names[::skip]:
        if verbose: print name
        img = vigra.impex.readImage(name).view(np.ndarray).swapaxes(0, 1).squeeze()
        imgs.append(img)
        count += 1
        if count >= stop and stop != -1:
            break
    
    return imgs

def getMasks(fnames):
    
    masks = []
    for name in fnames:
        bw = scipy.io.loadmat(name, chars_as_strings=1, matlab_compatible=1)
        masks.append(bw.get('BW'))
    
    return masks


def shuffleWithIndex(listv, seed=None):
    # Shuffle a list and return the indexes
    if seed != None: np.random.seed(seed)
    listvp = np.asarray(listv, dtype=object)
    ind = np.arange(len(listv))
    ind = np.random.permutation(ind)
    listvp = listvp[ind]
    listvp = list(listvp)
    return listvp, ind
    

def takeIndexFromList(listv, ind):
    listvp = np.asarray(listv, dtype=object)
    return list(listvp[ind])


def shuffleRows(array):
    ind = np.arange(array.shape[0])
    np.random.shuffle(ind)
    array = np.take(array, ind, axis=0)
    return array, ind
    
def generateRandomOdd(pwbase, treeCount):
    # Generate random odd numbers in the interavel [0,pwbase]
    res = []
    count = 0
    while count < treeCount:
        ext = np.random.randint(0, pwbase, 1)
        if np.mod(ext, 2) == 1:
            res.append(ext)
            count += 1
    
    return res
