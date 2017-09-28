## NOTE: Moved sample images / data to desktop

# Import necessary modules
import numpy as np
import pandas as pd
import dicom
import os
import csv
import scipy.ndimage
import matplotlib.pyplot as plt
import cPickle
import cv2

from sklearn.neural_network import MLPClassifier
from skimage import data, feature, measure, morphology #scikit-image
from sklearn import svm, metrics #scikit-learn
import sklearn.preprocessing as pre
from scipy import stats
from mpl_toolkits.mplot3d.art3d import Poly3DCollection #3d-plotting

import warnings
import skfuzzy as fuzz
warnings.simplefilter("ignore", DeprecationWarning)

INPUT_SCAN_FOLDER = '../../../../Desktop/larger_balanced_sample_preprocessed/'
OUTPUT_FOLDER = '../../../../Desktop/larger_balanced_sample_preprocessed/'

# Define related constants
# GROUND_TRUTH = 'stage1_labels.csv'
GROUND_TRUTH = '/Volumes/My Passport for Mac/larger_sample_patient_truths.csv'
patients = os.listdir(INPUT_SCAN_FOLDER)
patients.sort()

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

def load_scan_data(patientid):
	data = np.load(OUTPUT_FOLDER + patientid + '.npz')['arr_0']
	return data


def calculate_lbp(slice_data, numPoints, numNeighbors, radius, eps=1e-7):
	lbp_data = feature.local_binary_pattern(slice_data, numPoints, radius, method="uniform")
	n_bins = numPoints + 2
	(hist_data, _) = np.histogram(lbp_data.ravel(), bins=np.arange(0, n_bins+1), range=(0, n_bins))

	# Currently assuming last point in histogram is most frequent, getting rid of it (represents grey matter)
	hist_data = hist_data[:-1]

	# Normalize the histogram
	hist_data = hist_data.astype("float")
	hist_data /= (hist_data.sum() + eps)

	return hist_data

def perform_lbp(scan_data, slice_center=80):

	# Define base/center slice
	base_slice = scan_data[slice_center]
	output_training_data = []

	# Create LBPs and histograms for every other slice, centered around the base slice upto 10 either side
	for i in range(len(scan_data)):
		if i <= 10 and i % 2 == 0:

			# Create LBP vector
			radius = 3
			numNeighbors = 8
			numPoints = numNeighbors * radius

			# Create LBP for base slice
			if i == 0:
				base_slice = np.squeeze(base_slice)
				hist_data = calculate_lbp(base_slice, numPoints, numNeighbors, radius)
				output_training_data.append(hist_data)
			
			# Create LBP for other slices (plus and minus)
			else:
				data_slice_plus = np.squeeze(scan_data[80+i])
				data_slice_minus = np.squeeze(scan_data[80-i])
				
				hist_data_plus = calculate_lbp(data_slice_plus, numPoints, numNeighbors, radius)
				hist_data_minus = calculate_lbp(data_slice_minus, numPoints, numNeighbors, radius)

				output_training_data.append(hist_data_plus)
				output_training_data.append(hist_data_minus)

	return output_training_data


# Main function: Applies above preprocessing steps to all the data (only needs to be done offline once)
def main():
	dicom_folder_list = [ name.split('.')[0] for name in os.listdir(INPUT_SCAN_FOLDER)]

	# Store ground truths
	with open(GROUND_TRUTH, 'rb') as ground_truth_file:
	    reader = csv.reader(ground_truth_file)
	    data_listed = list(reader)

	truth_list_sample_sorted = []
	truth_list_patients = [data_listed[i][0] for i in range(1,len(data_listed))]
	truth_list_truths = [data_listed[i][1] for i in range(1,len(data_listed))]
	for i in range(len(dicom_folder_list)):
		try:
			truth_index = truth_list_patients.index(truth_list_patients[i])
			truth_list_sample_sorted.append(int(truth_list_truths[truth_index]))
		except Exception as e:
			print e
			if dicom_folder_list[i] != '.DS_Store':
				truth_list_sample_sorted.append(0) #for patients not in the data set, assume a ground truth of zero

	# Initialize Training Set
	training_data_set = []
	training_truths_set = []

	for i in range(len(dicom_folder_list)):
		print dicom_folder_list[i]
		data = load_scan_data(dicom_folder_list[i])
		hist_data = perform_lbp(data)
		training_data_set.append(hist_data)
		print i
		training_truths_set.append([truth_list_sample_sorted[i]]*len(hist_data)) 

	# Calculate Similarity via Chi Squared Distance (a smaller distance -> greater similarity)
	chi_predictions = []
	for patient_number in range(len(training_data_set)):
		current_patient = training_data_set[patient_number]
		remaining_data = [x for i,x in enumerate(training_data_set) if i!= patient_number]
		remaining_truth = [x for i,x in enumerate(truth_list_sample_sorted) if i != patient_number]

		similarities = []
		for remaining_patient_num in range(len(remaining_data)):
			patient = remaining_data[remaining_patient_num]
			chi_vals = []
			for hist_count in range(len(patient)):
				test = stats.chisquare(current_patient[hist_count], patient[hist_count])
				chi_vals.append(test[0])

			avg = np.mean(chi_vals)
			similarities.append(avg)

		chi_predictions.append(remaining_truth[similarities.index(min(similarities))])

	# Output Metrics for Chi Squared Distance Based Approach
	log_loss_error = metrics.log_loss(truth_list_sample_sorted, chi_predictions)
	print "Predictions"
	print chi_predictions
	print "Truth"
	print truth_list_sample_sorted
	print "Log Loss Error: " + str(log_loss_error)

	correct_count = 0
	total_neg_count = 0
	total_pos_count = 0
	false_pos_count = 0
	false_neg_count = 0

	# Calculate metrics
	for i in range(len(chi_predictions)):
		if chi_predictions[i] == truth_list_sample_sorted[i]:
			correct_count += 1
		elif chi_predictions[i] == 1 and truth_list_sample_sorted[i] == 0:
			false_pos_count += 1
		elif chi_predictions[i] == 0 and truth_list_sample_sorted[i] == 1:
			false_neg_count += 1
		if truth_list_sample_sorted[i] == 0:
			total_neg_count += 1
		elif truth_list_sample_sorted[i] == 1:
			total_pos_count += 1
	
	# Output metrics to console
	print "Number of False Positives: " + str(false_pos_count)
	print "Number of False Negatives: " + str(false_neg_count)
	print "False Positive Rate: " + str(float(false_pos_count) / float(total_neg_count) * 100) + "%"
	print "Overall Percent Accuracy: " + str(float(correct_count) / float(len(truth_list_sample_sorted)) * 100) + "%"

# Run main function if script is called independantly and explicitly 
if __name__ == "__main__":
    main()
