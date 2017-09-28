# ## NOTE: Moved sample images / data to desktop

# # Import necessary modules
# import numpy as np
# import pandas as pd
# import dicom
# import os
# import scipy.ndimage
# import matplotlib.pyplot as plt

# from skimage import measure, morphology #scikit-image
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection #3d-plotting

# # Define related constants
# INPUT_FOLDER = 'Files/sample_images/'
# patients = os.listdir(INPUT_FOLDER)
# patients.sort()

# # Load the scans in given folder path
# def load_scan(path):
#     slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
#     slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
#     try:
#         slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
#     except:
#         slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
#     for s in slices:
#         s.SliceThickness = slice_thickness
        
#     return slices

# # Get image data and then convert pixel intensity to Hounsfield units (HU), measure of radiodensity
# def get_pixels_hu(slices):
#     image = np.stack([s.pixel_array for s in slices])
#     # Convert to int16 (from sometimes int16), 
#     # should be possible as values should always be low enough (<32k)
#     image = image.astype(np.int16)

#     # Set outside-of-scan pixels to 0
#     # The intercept is usually -1024, so air is approximately 0
#     image[image == -2000] = 0
    
#     # Convert pixel intensity to Hounsfield units (HU) (radiodensity)
#     for slice_number in range(len(slices)):
        
#         intercept = slices[slice_number].RescaleIntercept
#         slope = slices[slice_number].RescaleSlope
        
#         if slope != 1:
#             image[slice_number] = slope * image[slice_number].astype(np.float64)
#             image[slice_number] = image[slice_number].astype(np.int16)
            
#         image[slice_number] += np.int16(intercept)
    
#     return np.array(image, dtype=np.int16)


# # Resample all images at constant rate due to variance in scanning resolutions for slices of different CT scans
# def resample(image, scan, new_spacing=[1,1,1]):
#     # Determine current pixel spacing
#     spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

#     resize_factor = spacing / new_spacing
#     new_real_shape = image.shape * resize_factor
#     new_shape = np.round(new_real_shape)
#     real_resize_factor = new_shape / image.shape
#     new_spacing = spacing / real_resize_factor
    
#     image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
#     return image, new_spacing


# # Plot lung in 3D
# def plot_3d(image, threshold=-300):
    
#     # Position the scan upright, 
#     # so the head of the patient would be at the top facing the camera
#     p = image.transpose(2,1,0)
    
#     verts, faces, normals, values = measure.marching_cubes(p, threshold)

#     fig = plt.figure(figsize=(5, 5))
#     ax = fig.add_subplot(111, projection='3d')

#     # Fancy indexing: `verts[faces]` to generate a collection of triangles
#     mesh = Poly3DCollection(verts[faces], alpha=0.70)
#     face_color = [0.45, 0.45, 0.75]
#     mesh.set_facecolor(face_color)
#     ax.add_collection3d(mesh)

#     ax.set_xlim(0, p.shape[0])
#     ax.set_ylim(0, p.shape[1])
#     ax.set_zlim(0, p.shape[2])

#     plt.show()

# # Determines largest solid volume within given image
# def largest_label_volume(im, bg=-1):
#     vals, counts = np.unique(im, return_counts=True)

#     counts = counts[vals != bg]
#     vals = vals[vals != bg]

#     if len(counts) > 0:
#         return vals[np.argmax(counts)]
#     else:
#         return None

# # Lung segmentation mask: extracts solid relevant structures from within the lung
# def segment_lung_mask(image, fill_lung_structures=True):
    
#     # not actually binary, but 1 and 2. 
#     # 0 is treated as background, which we do not want
#     binary_image = np.array(image > -320, dtype=np.int8)+1
#     labels = measure.label(binary_image)
    
#     # Pick the pixel in the very corner to determine which label is air.
#     # Improvement: Pick multiple background labels from around the patient
#     # More resistant to "trays" on which the patient lays cutting the air 
#     # around the person in half
#     background_label = labels[0,0,0]
    
#     #Fill the air around the person
#     binary_image[background_label == labels] = 2
    
    
#     # Method of filling the lung structures (that is superior to something like 
#     # morphological closing)
#     if fill_lung_structures:
#         # For every slice we determine the largest solid structure
#         for i, axial_slice in enumerate(binary_image):
#             axial_slice = axial_slice - 1
#             labeling = measure.label(axial_slice)
#             l_max = largest_label_volume(labeling, bg=0)
            
#             if l_max is not None: #This slice contains some lung
#                 binary_image[i][labeling != l_max] = 1

    
#     binary_image -= 1 #Make the image actual binary
#     binary_image = 1-binary_image # Invert it, lungs are now 1
    
#     # Remove other air pockets insided body
#     labels = measure.label(binary_image, background=0)
#     l_max = largest_label_volume(labels, bg=0)
#     if l_max is not None: # There are air pockets
#         binary_image[labels != l_max] = 0
 
#     return binary_image


# def get_my_image(slices):
    
#     images = []
#     for sl in slices:
        
#         img = sl.pixel_array
#         img[img == -2000] = 0.
        
#         images.append((1.*img)/np.max(img))
        
#     return np.array(images)

# # Main function: Applies above preprocessing steps to all the data (only needs to be done offline once)
# def main():

#     #Preprocess all existing data
#     for i in range(len(patients)):

#         # Console output
#         print("\nPreprocessing data from patient " + str(i) + "\n");

#         # Load the CT scan data
#         patient = load_scan(INPUT_FOLDER + patients[i])
#         patient_pixels = get_pixels_hu(patient)

#         # Resample data to be consistent / fixed at 1 mm x 1 mm x 1 mm
#         pix_resampled, spacing = resample(patient_pixels, patient, [1,1,1])
#         print("Shape before resampling\t", patient_pixels.shape)
#         print("Shape after resampling\t", pix_resampled.shape)

#         # Apply segmentation mask to extract relevant information
#         segmented_lungs = segment_lung_mask(pix_resampled, False)
#         segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
#         masked_lungs = segmented_lungs_fill - segmented_lungs
        
#         # Trim masked image data to a fixed size of 200 slices
#         trim_size = 200
#         offset = len(masked_lungs) - trim_size
#         if (offset % 2 != 0):
#             masked_lungs = masked_lungs[(offset/2)+1:]
#         else:
#             masked_lungs = masked_lungs[(offset/2):]
#         masked_lungs = masked_lungs[:-(offset/2)]

#         # Save masked data   
#         np.save('Preprocessed Data/' + patient[0].PatientsName + '_masked_lung', patient_pixels)

# # Run main function if script is called independantly and explicitly 
# if __name__ == "__main__":
#     main()

# # Define related constants
# INPUT_FOLDER = 'Files/sample_images/'
# GROUND_TRUTH = 'Files/stage1_labels.csv'
# patients = os.listdir(INPUT_FOLDER)
# patients.sort()

# # Normalization
# MIN_BOUND = -1000.0
# MAX_BOUND = 400.0
# def normalize(image):
#     image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
#     image[image>1] = 1.
#     image[image<0] = 0.
#     return image

# # Zero center the image
# PIXEL_MEAN = 0.25
# def zero_center(image):
#     image = image - PIXEL_MEAN
#     return image

# # Store ground truths
# with open(GROUND_TRUTH, 'rb') as ground_truth_file:
#     reader = csv.reader(ground_truth_file)
#     data_listed = list(reader)

# truth_list_sample_sorted = []
# truth_list_patients = [data_listed[i][0] for i in range(1,len(data_listed))]
# truth_list_truths = [data_listed[i][1] for i in range(1,len(data_listed))]

# for i in range(len(patients)):
# 	try:
# 		truth_index = truth_list_patients.index(patients[i])
# 		truth_list_sample_sorted.append(truth_list_truths[truth_index])
# 	except Exception as e:
# 		print e
# 		truth_list_sample_sorted.append(0) #for patients not in the data set, assume a ground truth of zero

# # Generate training set for the SVM classifier
# def create_SVM_training_data():

# 	# Initialize training set
# 	lbp_training_set = {}
# 	lbp_training_set['Patient Hist'] = []
# 	lbp_training_set['Patient Truth'] = np.zeros((len(patients), 1))
# 	lbp_training_set['Patient Truth'] = truth_list_sample_sorted

# 	for patient_number in range(len(patients)):

# 		# Initialize training set
# 		patient = patients[patient_number]
# 		lbp_training_set['Patient Hist'].append([])
# 		# histogram = np.zeros(59)

# 		# Console output
# 		print("\nAdding patient " + str(patient_number) + "\n")

# 		# Load image
# 		ct_scan = np.load('Preprocessed Data/' + patient + '_not_masked.npy')
# 		print("\nLength is " + str(len(ct_scan)) + "\n")

# 		# Generate LBP Histogram for every 4th slice (creates 50 histograms in total per CT Scan)
# 		for slice_number in range(len(ct_scan)):
# 			index = 0

# 			# Generate LBP histogram for every 4th slice
# 			if (slice_number % 4 == 0):
			
# 				# Store slice image
# 				image = ct_scan[slice_number]

# 				# Minor preprocessing
# 				image = normalize(image)
# 				image = zero_center(image)

# 				# Create LBP vector
# 				radius = 3
# 				numNeighbors = 8
# 				numPoints = numNeighbors * radius
# 				lbp = feature.local_binary_pattern(image, numPoints, radius)

# 				# Convert vector to histogram
# 				lbp = feature.local_binary_pattern(image, numPoints, radius, method="uniform")
# 				n_bins = numNeighbors * (numNeighbors - 1) + 3
# 				(hist, _) = np.histogram(lbp.ravel(), bins=n_bins, range=(0, lbp.ravel().max()))
# 				# print pre.minmax_scale(hist)

# 				# Return the histogram of Local Binary Patterns
# 				lbp_training_set['Patient Hist'][patient_number].append([0]*len(hist))
# 				lbp_training_set['Patient Hist'][patient_number][index] = pre.minmax_scale(hist)

# 				# Icrement counter
# 				index += 1

# 	# Return Output		
# 	return lbp_training_set 


# # Main function: Applies above preprocessing steps to all the data (only needs to be done offline once)
# def main():

# 	# Create training set for SVM
# 	training_set = create_SVM_training_data()

#  # 	# Save training data offline
# 	with open("svm_training_data.txt", "wb") as training_file:
# 		cPickle.dump(training_set, training_file)

# 	# Test load
# 	with open("svm_training_data.txt", "rb") as training_file:
# 		training_set = cPickle.load(training_file)

# 	# Initialize SVM
# 	clf = svm.SVC(gamma=0.0001, C=100)

# 	# Test & Train using Leave One Out process
# 	predictions = []
# 	for excluded_patient_number in range(len(patients)):

# 		# Console Output
# 		print("\nLeaving patient " + str(excluded_patient_number) + " out\n")

# 		# Select excluded patient/CT scan
# 		excluded_patient_data = training_set['Patient Hist'][excluded_patient_number]
# 		excluded_patient_truth = training_set['Patient Truth'][excluded_patient_number]

# 		# Initialize X/Y SVM inputs
# 		X = []
# 		Y = []

# 		# Iterate over remaining data
# 		remaining_data = [x for i,x in enumerate(training_set['Patient Hist']) if i != excluded_patient_number]
# 		remaining_truth = [x for i,x in enumerate(training_set['Patient Truth']) if i != excluded_patient_number]

# 		# Create X/Y inpuy arrays on remaining data
# 		for remaining_patient_number in range(len(remaining_data)):

# 			patient_training_data = training_set['Patient Hist'][remaining_patient_number]
# 			patient_training_truth = training_set['Patient Truth'][remaining_patient_number]

# 			Y = Y + ([patient_training_truth] * len(patient_training_data))

# 			for slice_number in range(len(patient_training_data)):
# 				X.append(patient_training_data[slice_number])

# 		# Train SVM
# 		clf.fit(X, Y)

# 		# Test on Excluded Patient
# 		slice_predictions = []
# 		for slice_number in range(len(excluded_patient_data)):
# 			slice_predictions.append(float(clf.predict(excluded_patient_data[slice_number])[0]))

# 		predictions.append(sum(slice_predictions) / float(len(slice_predictions)))

# 	# View output
# 	true_output = [float(a) for a in training_set['Patient Truth']]
# 	log_loss_error = metrics.log_loss(true_output, predictions)
# 	print predictions
# 	print true_output
# 	print log_loss_error



	# # Initialize SVM
	# clf = svm.SVC(gamma=0.0001, C=100)
	# svm_predictions = []

	# # Train & Test using Leave One Out Process
	# for excluded_patient_number in range(len(dicom_folder_list)):

	# 	# Console Output
	# 	print("\nLeaving patient " + str(excluded_patient_number) + " out\n")

	# 	# Select excluded patient/CT scan
	# 	excluded_patient_data = training_data_set[excluded_patient_number]
	# 	excluded_patient_truth = truth_list_sample_sorted[excluded_patient_number]

	# 	# Initialize X/Y SVM inputs
	# 	X = []
	# 	Y = []

	# 	# Iterate over remaining data
	# 	X = [x for i,x in enumerate(training_data_set) if i != excluded_patient_number]
	# 	Y = [x for i,x in enumerate(truth_list_sample_sorted) if i != excluded_patient_number]

	# 	# Train SVM
	# 	X = np.asarray(X)
	# 	Y = np.asarray(Y)
	# 	print X.shape
	# 	print Y.shape
	# 	clf.fit(X, Y)

	# 	# Test on Excluded Patient
	# 	test_X = np.asarray(excluded_patient_data)
	# 	test_Y = np.asarray(excluded_patient_truth)
	# 	test_prediction = clf.predict(excluded_patient_data)
	# 	svm_predictions.append(test_prediction[0])

	# # Output Metrics for SVM Based Approach
	# log_loss_error = metrics.log_loss(truth_list_sample_sorted, svm_predictions)
	# print "Predictions"
	# print svm_predictions
	# print "Truth"
	# print truth_list_sample_sorted
	# print "Log Loss Error: " + str(log_loss_error)
	# correct_count = 0
	# for i in range(len(svm_predictions)):
	# 	if svm_predictions[i] == truth_list_sample_sorted[i]:
	# 		correct_count+=1
	# print "Percent Accuracy: " + str(float(correct_count) / float(len(truth_list_sample_sorted)) * 100) + "%"