# Import necessary modules
import sys
import numpy as np
from numpy import *
from scipy import stats
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import *
import glob
from sklearn.model_selection import train_test_split
import datetime
import math
import os.path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from multiprocessing import Pool
import time
from skimage import measure, morphology, segmentation
import scipy.ndimage as ndimage
import dicom

import keras
from keras.layers.core import *
from keras.layers.normalization import *
from keras.layers.pooling import *
from keras.layers import Input
from keras.layers.convolutional import *
from keras.regularizers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.models import Model, Sequential
from keras.models import load_model
import tensorflow as tf

import warnings
import skfuzzy as fuzz
warnings.simplefilter("ignore", DeprecationWarning)

#INPUT_SCAN_FOLDER = '/data/kaggle_cancer_2017/stage1/'
#OUTPUT_FOLDER = '/data/kaggle_preprocessed_output/'

# For Kaggle I have added  sample_image directory only
INPUT_SCAN_FOLDER = '/Volumes/My Passport for Mac/larger_balanced_sample_images/'
OUTPUT_FOLDER = '../../../../Desktop/'

THRESHOLD_HIGH = 700
THRESHOLD_LOW = -1100

# fix random seed for reproducibility
np.random.seed(17)


# Simple Function to show the slice at full resolustion normal imshow would downscale this image.
# It can accept either (image_width, image_height) array or (image_width, image_height, 1) numpy as input.
# Optional Value range is a tuple of fixed max value and min value. This is useful if you do not want color 
#  to change between different scan slices.

def show_slice(arr, value_range = None):
    if len (list(arr.shape)) > 2:
        arr2 = arr.copy()
        arr2 = np.reshape (arr, (arr.shape[0],arr.shape[1]))
    else:
        arr2 = arr

    dpi = 80
    margin = 0.05 # (5% of the width/height of the figure...)
    xpixels, ypixels = arr2.shape[0], arr2.shape[1]

    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * ypixels / dpi, (1 + margin) * xpixels / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    if value_range is None:
        plt.imshow(arr2, cmap=plt.cm.gray)
    else:        
        ax.imshow(arr2, vmin=value_range[0], vmax=1, cmap=plt.cm.gray, interpolation='none')
    plt.show()

def preprocess_all_scans_mp (in_folder, out_folder, demo=False):

    dicom_folder_list = [ name for name in os.listdir(in_folder) if os.path.isdir(os.path.join(in_folder, name)) ]
   
    # For Testing feed just load one scan
    print dicom_folder_list
    for patient_id in dicom_folder_list:
        segment_pad_and_save_ct_scan_as_npz  (patient_id, demo=False)
    
    # if not demo:
    #     # Multi-threaded processes to utilize all available CPUs for this task. Note that many threads will block on IO
    #     # so creating more than number of CPUs.    
    #     thread_pool = Pool(32)
    #     thread_pool.map (segment_pad_and_save_ct_scan_as_npz, dicom_folder_list)
        
    #     # Cleanup
    #     thread_pool.close()
    #     thread_pool.join_thread()
        
def segment_pad_and_save_ct_scan_as_npz (scanid, demo=False):
    
    print scanid
    scan_dir = INPUT_SCAN_FOLDER + str(scanid)
    np.set_printoptions(threshold='nan') 
    scan = load_scan_as_HU_nparray(scan_dir)
    
    # For demo reduce number of slices to 5 to save time
    if demo:
        scan = scan[78:82]
    
    if demo:
        print ("----Loaded Scan and Converted to HU units----")
        print ("Shape: ", scan.shape)
        show_slice (scan[3])

    
    scan = seperate_lungs_and_pad (scan)
    
    if demo:
        print ("----Segmented Lung and Padded/Trimmed to have 256 slices----")
        print ("Shape: ", scan.shape)
        show_slice (scan[3])
        
    scan = threshold_and_normalize_scan (scan)
    
    if demo:
        print ("----Thresholded and Normalized----")
        print ("Shape: ", scan.shape)
        show_slice (scan[3])
        radius = 3
        numNeighbors = 8
        numPoints = numNeighbors * radius
        lbp_data = feature.local_binary_pattern(scan[3], numPoints, radius)
        n_bins = numNeighbors * (numNeighbors - 1) + 3
        print lbp_data
        (hist_data, _) = np.histogram(lbp_data.ravel(), bins=n_bins, range=(0, lbp_data.ravel().max()))
        plt.plot(hist_data)
        plt.show()   
    
    # Save output file to compressed npz file for easy reading.
    if not demo:
        out_file = OUTPUT_FOLDER + 'larger_balanced_sample_preprocessed/' + scanid + '.npz'    
        np.savez_compressed (out_file, scan)
    
# Load the scans in given folder path
def load_scan_as_HU_nparray(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
    
    image = np.stack([s.pixel_array for s in slices])
    
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)        


def seperate_lungs_and_pad(scan):
    
    # Make total 256 slices fill in -1100 as exterme value 
    segmented_scan = np.full ((256, 512, 512), THRESHOLD_LOW)
    
    for i, image in enumerate (scan):
        print i
        # Ignore all slices later than 255 if required.
        if (i == 256):
            break
        
        # Creation of the internal Marker
        marker_internal = image < -400
        marker_internal = segmentation.clear_border(marker_internal)
        marker_internal_labels = measure.label(marker_internal)
        areas = [r.area for r in measure.regionprops(marker_internal_labels)]
        areas.sort()
        if len(areas) > 2:
            for region in measure.regionprops(marker_internal_labels):
                if region.area < areas[-2]:
                    for coordinates in region.coords:                
                           marker_internal_labels[coordinates[0], coordinates[1]] = 0
        marker_internal = marker_internal_labels > 0
        
        # Creation of the external Marker
        external_a = ndimage.binary_dilation(marker_internal, iterations=10)
        external_b = ndimage.binary_dilation(marker_internal, iterations=55)
        marker_external = external_b ^ external_a
        
        # Creation of the Watershed Marker matrix
        marker_watershed = np.zeros((512, 512), dtype=np.int)
        marker_watershed += marker_internal * 255
        marker_watershed += marker_external * 128

        # Creation of the Sobel-Gradient
        sobel_filtered_dx = ndimage.sobel(image, 1)
        sobel_filtered_dy = ndimage.sobel(image, 0)
        sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
        sobel_gradient *= 255.0 / np.max(sobel_gradient)

        # Watershed algorithm
        watershed = morphology.watershed(sobel_gradient, marker_watershed)

        # Reducing the image created by the Watershed algorithm to its outline
        outline = ndimage.morphological_gradient(watershed, size=(3,3))
        outline = outline.astype(bool)

        # Performing Black Top-hat Morphology for reinclusion
        # Creation of the disk-kernel and increasing its size a bit
        blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1, 0, 0]]
        blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
        
        # Perform the Black-Hat
        outline += ndimage.black_tophat(outline, structure=blackhat_struct)

        # Use the internal marker and the Outline that was just created to generate the lungfilter
        lungfilter = np.bitwise_or(marker_internal, outline)
        
        # Close holes in the lungfilter
        # fill_holes is not used here, since in some slices the heart would be reincluded by accident
        lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)

        # Apply the lungfilter (note the filtered areas being assigned 30 HU)
        segmented_scan[i] = np.where(lungfilter == 1, image, 30*np.ones((512, 512)))
        
    return segmented_scan

def threshold_and_normalize_scan (scan):
    scan = scan.astype(np.float32)
    scan [scan < THRESHOLD_LOW] = THRESHOLD_LOW
    scan [scan > THRESHOLD_HIGH] = THRESHOLD_HIGH
    
    # Maximum absolute value of any pixel .
    max_abs = abs (max(THRESHOLD_LOW, THRESHOLD_HIGH, key=abs))
    
    # This will bring values between -1 and 1
    scan /= max_abs
    
    return scan

# if OUTPUT_FOLDER:
#     os.makedirs (OUTPUT_FOLDER+'larger_balanced_sample_preprocessed/')
    
# For full preprocessing you should to set demo=False
preprocess_all_scans_mp (INPUT_SCAN_FOLDER, OUTPUT_FOLDER, demo=False)