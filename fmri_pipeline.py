# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:29:55 2025

@author: shili
"""
import glob
import os
from PIL import Image
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.image import resample_to_img
import matplotlib.pyplot as plt

os.chdir('parent/directory')

# Create NIFTI atlas using ABBA registration on ImageJ https://abba-documentation.readthedocs.io/en/latest/
# Macro script for ImageJ to convert putlines of the registered brain regions to masks is roi_to_mask.ijm

df = pd.read_excel('annotation_label_IDs.xlsx') #annotaion IDs from AIDAmri

label_dict = df.set_index('acronym')['id'].to_dict()

input_folder = 'masks'

right = np.zeros((256, 256, 30))
left = np.zeros((256, 256, 30))
masks = np.zeros((256, 256, 30))

for i in range(30):
    z = f"{i:02}"  # Zero-padded index
    image_name = f"slice_{z}"
    image_files = glob.glob(f"{input_folder}/**/*{image_name}*.tif", recursive=True)
    regions_combined = np.zeros((256, 256))

    # Process each file
    for k in image_files:
        if "Right" in k:
            image = Image.open(k)
            image_array = np.array(image)
            modified_image_array = np.rot90(image_array, k=-1)
            right[:, :, i] = modified_image_array
        elif "Left" in k:
            image = Image.open(k)
            image_array = np.array(image)
            modified_image_array = np.rot90(image_array, k=-1)
            left[:, :, i] = modified_image_array
        else:
            # Extract region key
            region_key =  k[14:].split('.')[0] 
            if region_key in label_dict:
                new_pixel_value = label_dict[region_key]
                print(f"Processing region: {region_key} with ID {new_pixel_value}")
                image = Image.open(k)
                image_array = np.array(image)
                # Replace pixel values
                image_array = image_array 
                modified_image_array = np.where(image_array != 0, new_pixel_value, image_array)
                modified_image_array = np.rot90(modified_image_array, k=-1)
                regions_combined += modified_image_array
    masks[:, :, i] = regions_combined

masks_right = masks * right
largest_right_label = masks_right.max()
masks_right_label = np.where(masks_right > 0, masks_right + largest_right_label, 0)
masks_hemi = masks*left + masks_right_label

T2mri = nib.load('path/to/T2-weighted-image-used-in-ABBA')
atlas_nifti = nib.Nifti1Image(masks_hemi, T2mri.affine, T2mri.header)
nib.save(atlas_nifti, 'path/to/abba-atlas.nii')


# For this pipeline, fMRI image was preprocessed in CONN toolbox using the following functions:
# functional Realignment (subject motion estimation and correction)
# functional Slice timing correction (STC; correction for inter-slice differeces in acquisition time)
# functional Smoothing (spatial convolution with Gaussian kernel), 0.5 mm
# functional Band-pass filtering (temporal filtering of BOLD data),  0.01 to 0.3 Hz. 

fmri_img = nib.load('path/to/processed-fmri')
fmri_data = fmri_img.get_fdata()

# Resample atlas from T2-weighted dimensions to fMRI dimensions
resampled_atlas = resample_to_img(source_img=atlas_nifti, target_img=fmri_img, interpolation='nearest')
resampled_atlas_data = resampled_atlas.get_fdata()
unique_labels, voxel_counts = np.unique(resampled_atlas_data, return_counts=True)

# Initialize NiftiLabelsMasker
masker = NiftiLabelsMasker(labels_img=resampled_atlas, verbose=1, standardize=True, detrend=True)
masker.fit(fmri_img)
report = masker.generate_report()
report.save_as_html("path/to/report.html")

# Load motion parameters
motion_params = np.loadtxt("path/to/CONN-output/rp_fMRI.txt")  

# Calculate the global signal by averaging over all the voxels at each time point
global_signal = np.mean(fmri_data, axis=(0, 1, 2))  # Mean over x, y, z dimensions

combined_confounds = np.column_stack([motion_params, global_signal])

time_series = masker.fit_transform(fmri_img, confounds=combined_confounds)

# Initialize ConnectivityMeasure (e.g., correlation or partial correlation)
connectivity_measure = ConnectivityMeasure(kind='correlation')  # Options: 'correlation', 'partial correlation', etc.

# Compute the connectivity matrix
connectivity_matrix = connectivity_measure.fit_transform([time_series])[0]  # Shape: (n_regions, n_regions)

# Plot the connectivity matrix
plt.imshow(connectivity_matrix, cmap='viridis', vmax=1, vmin=-1)
plt.colorbar()
plt.title("Connectivity Matrix (Python)")
plt.show()

# z-transform the data
z_transformed = 0.5 * np.log((1 + connectivity_matrix) / (1 - connectivity_matrix))

# Plot the z-transformed connectivity matrix
plt.imshow(z_transformed, cmap='viridis', vmax=3, vmin=0)
plt.colorbar()
plt.title("Connectivity Matrix (Python)")
plt.show()