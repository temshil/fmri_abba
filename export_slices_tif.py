# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:27:39 2025

@author: shili
"""

import nibabel as nib
import numpy as np
from PIL import Image
import os

# Load the MRI file
input_file = 'path/to/registeredT2w'  # Replace with your file path
output_folder = 'path/to/slices'   # Output folder for saving PNG slices

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load MRI data
mri_img = nib.load(input_file)
mri_data = mri_img.get_fdata()

# Save each slice as a TIFF file
num_slices = mri_data.shape[2]
for i in range(num_slices):
    # Normalize the slice to 0-255
    slice_data = mri_data[:, :, i]
    slice_norm = ((slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255).astype(np.uint8)

    # Rotate the slice 90 degrees clockwise
    slice_image = Image.fromarray(slice_norm).rotate(+90)

    # Save as TIFF
    output_path = os.path.join(output_folder, f'slice_{i:02d}.tif')
    slice_image.save(output_path)

print(f"Saved {num_slices} slices as TIFF files in {output_folder}")