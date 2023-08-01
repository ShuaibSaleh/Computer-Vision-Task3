from PIL import Image
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import cv2
from itertools import compress

from scipy.ndimage import maximum_filter
from scipy.ndimage import map_coordinates
from scipy.ndimage import convolve1d as conv1
from scipy.ndimage import convolve as conv2


from skimage.io import imread
from skimage.transform import ProjectiveTransform, SimilarityTransform, AffineTransform
from skimage.measure import ransac

from utils import gaussian2, maxinterp, circle_points

import time



## The previous part illustrated OpenCV's built-in capabilities.
## Let's try to do Harris corner extraction and matching using our own
## implementation in a less black-box manner.

## Familiarize yourself with the harris function
def harris(im, sigma=1.0, relTh=0.0001, k=0.04):
    im = im.astype(np.float) # Make sure im is float
    
    # Get smoothing and derivative filters
    g,_,_,_,_,_ = gaussian2(sigma)
    _, gx, gy, _, _, _ = gaussian2(np.sqrt(0.5))
    
    # Partial derivatives
    Ix = conv2(im, -gx, mode='constant')
    Iy = conv2(im, -gy, mode='constant')
    
    # Components of the second moment matrix
    Ix2Sm = conv2(Ix**2, g, mode='constant')
    Iy2Sm = conv2(Iy**2, g, mode='constant')
    IxIySm = conv2(Ix*Iy, g, mode='constant')
    
    # Determinant and trace for calculating the corner response
    detC = (Ix2Sm*IxIySm)-(Iy2Sm**2)
    traceC = Ix2Sm+IxIySm
    
    # Corner response function R
    # "Corner": R > 0
    # "Edge": R < 0
    # "Flat": |R| = small
    R = detC-k*traceC**2
    maxCornerValue = np.amax(R)
    
    # Take only the local maxima of the corner response function
    fp = np.ones((3,3))
    fp[1,1] = 0
    maxImg = maximum_filter(R, footprint=fp, mode='constant')
    
    # Test if cornerness is larger than neighborhood
    cornerImg = R>maxImg
    
    # Threshold for low value maxima
    y, x = np.nonzero((R>relTh*maxCornerValue)*cornerImg) 
    
    # Convert to float
    x = x.astype(np.float)
    y = y.astype(np.float)
    
    # Remove responses from image borders to reduce false corner detections
    r, c = R.shape
    idx = np.nonzero((x<2)+(x>c-3)+(y<2)+(y>r-3))[0]
    x = np.delete(x,idx)
    y = np.delete(y,idx)
    
    # Parabolic interpolation
    for i in range(len(x)):
        _,dx=maxinterp((R[int(y[i]), int(x[i])-1], R[int(y[i]), int(x[i])], R[int(y[i]), int(x[i])+1]))
        _,dy=maxinterp((R[int(y[i])-1, int(x[i])], R[int(y[i]), int(x[i])], R[int(y[i])+1, int(x[i])]))
        x[i]=x[i]+dx
        y[i]=y[i]+dy
        
    return x, y, cornerImg



def match_imgs_using_SSD (path1,path2):

    # Load images
    img1 = imread(path1)
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)/255.0
    img2 = imread(path2)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)/255.0

    #resize images
    gray1 = cv2.resize(gray1,(400,400))
    img1 = cv2.resize(img1,(400,400))
    gray2 = cv2.resize(gray2,(400,400))
    img2 = cv2.resize(img2,(400,400))   

    start_time = time.time()
    # Harris corner extraction, take a look at the source code above
    x1, y1, _ = harris(gray1)
    x2, y2, _ = harris(gray2)

    ## We pre-allocate the memory for the 15*15 image patches extracted
    ## around each corner point from both images
    patch_size=15
    npts1=x1.shape[0]
    npts2=x2.shape[0]
    patches1=np.zeros((patch_size, patch_size, npts1))
    patches2=np.zeros((patch_size, patch_size, npts2))

    ## The following part extracts the patches using bilinear interpolation
    k=(patch_size-1)/2.
    xv,yv=np.meshgrid(np.arange(-k,k+1),np.arange(-k, k+1))
    for i in range(npts1):
        patch = map_coordinates(gray1, (yv + y1[i], xv + x1[i]))
        patches1[:,:,i] = patch
    for i in range(npts2):
        patch = map_coordinates(gray2, (yv + y2[i], xv + x2[i]))
        patches2[:,:,i] = patch

    ## We compute the sum of squared differences (SSD) of pixels' intensities
    ## for all pairs of patches extracted from the two images
    distmat = np.zeros((npts1, npts2))
    for i1 in range(npts1):
        for i2 in range(npts2):
            distmat[i1,i2]=np.sum((patches1[:,:,i1]-patches2[:,:,i2])**2)

    ## Next we compute pairs of patches that are mutually nearest neighbors
    ## according to the SSD measure
    ss1 = np.amin(distmat, axis=1)
    ids1 = np.argmin(distmat, axis=1)
  
    ids2 = np.argmin(distmat, axis=0)
    pairs = []
    for k in range(npts1):
        if k == ids2[ids1[k]]:
            pairs.append(np.array([k, ids1[k], ss1[k]]))
    pairs = np.array(pairs)

    ## We sort the mutually nearest neighbors based on the SSD
    
    id_ssd = np.argsort(pairs[:,2], axis=0)


    ## Next we visualize the best number matches which are mutual nearest neighbors
    ## and have the smallest SSD values
    Nvis = 40
    montage = np.concatenate((img1, img2), axis=1)

    plt.figure(figsize=(16, 8))
    plt.suptitle(f"The best {np.minimum(len(id_ssd), Nvis)} matches according to SSD measure", fontsize=20)
    plt.imshow(montage, cmap='gray')
    plt.title(f'The best {np.minimum(len(id_ssd), Nvis)} matches')
    for k in range(np.minimum(len(id_ssd), Nvis)):
        l = id_ssd[k]
        plt.plot(x1[int(pairs[l, 0])], y1[int(pairs[l, 0])], 'rx')
        plt.plot(x2[int(pairs[l, 1])] + gray1.shape[1], y2[int(pairs[l, 1])], 'rx')
        plt.plot([x1[int(pairs[l, 0])], x2[int(pairs[l, 1])]+gray1.shape[1]], 
            [y1[int(pairs[l, 0])], y2[int(pairs[l, 1])]])
        
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"processing time: {processing_time} s")
    path = "static/images/output/SSD_match.jpg"
    plt.savefig(path)


    return path , processing_time




def match_imgs_using_NCC (path1,path2):

    # Load images
    img1 = imread(path1)
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)/255.0
    img2 = imread(path2)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)/255.0

    #resize images
    gray1 = cv2.resize(gray1,(400,400))
    img1 = cv2.resize(img1,(400,400))
    gray2 = cv2.resize(gray2,(400,400))
    img2 = cv2.resize(img2,(400,400))   

    start_time = time.time()
    # Harris corner extraction, take a look at the source code above
    x1, y1, _ = harris(gray1)
    x2, y2, _ = harris(gray2)

    ## We pre-allocate the memory for the 15*15 image patches extracted
    ## around each corner point from both images
    patch_size=15
    npts1=x1.shape[0]
    npts2=x2.shape[0]
    patches1=np.zeros((patch_size, patch_size, npts1))
    patches2=np.zeros((patch_size, patch_size, npts2))

    ## The following part extracts the patches using bilinear interpolation
    k=(patch_size-1)/2.
    xv,yv=np.meshgrid(np.arange(-k,k+1),np.arange(-k, k+1))
    for i in range(npts1):
        patch = map_coordinates(gray1, (yv + y1[i], xv + x1[i]))
        patches1[:,:,i] = patch
    for i in range(npts2):
        patch = map_coordinates(gray2, (yv + y2[i], xv + x2[i]))
        patches2[:,:,i] = patch

    ## Compute Normalized cross correlation for each windows
    ncc = np.zeros((npts1, npts2))
    for i1 in range(npts1):
        for i2 in range(npts2):
            n1 = patches1[:,:,i1] - np.mean(patches1[:,:,i1])
            n2 = patches2[:,:,i2] - np.mean(patches2[:,:,i2])
            ncc[i1,i2] = np.sum(n1*n2)/np.sqrt(np.sum(n1**2)*np.sum(n2**2))
        
    ## Next we compute pairs of patches that are mutually nearest neighbors
    ## according to the ncc measure
    ncc1 = np.amax(ncc, axis=1)
    ids1 = np.argmax(ncc, axis=1)
 
    ids2 = np.argmax(ncc, axis=0)

    pairs = []
    for k in range(npts1):
        if k == ids2[ids1[k]]:
            pairs.append(np.array([k, ids1[k], ncc1[k]]))
    pairs = np.array(pairs)

    ## We sort the mutually nearest neighbors based on the ncc

    id_ncc = np.argsort(pairs[:,2], axis=0)[::-1]
    
    ## Next we visualize the best number of matches which are mutual nearest neighbors
    ## and have the smallest ncc values
    Nvis = 40
    montage = np.concatenate((img1, img2), axis=1)

    plt.figure(figsize=(16, 8))
    plt.suptitle(f"The best { np.maximum(len(id_ncc), Nvis) } matches according to ncc measure", fontsize=20)
    plt.imshow(montage, cmap='gray')
    plt.title(f'The best { np.maximum(len(id_ncc), Nvis) } matches')
    for k in range(np.maximum(len(id_ncc), Nvis)):
        l = id_ncc[k]
        plt.plot(x1[int(pairs[l, 0])], y1[int(pairs[l, 0])], 'rx')
        plt.plot(x2[int(pairs[l, 1])] + gray1.shape[1], y2[int(pairs[l, 1])], 'rx')
        plt.plot([x1[int(pairs[l, 0])], x2[int(pairs[l, 1])]+gray1.shape[1]], 
            [y1[int(pairs[l, 0])], y2[int(pairs[l, 1])]])
        
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"processing time: {processing_time} s")
    path = "static/images/output/NCC_match.jpg"
    plt.savefig(path)


    return path , processing_time
