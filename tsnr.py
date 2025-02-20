# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:59:38 2025

@author: shili
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Flip NIfTI images along a specified axis.")
    parser.add_argument('--path1', type=str,
                        help='Input path for fMRI file')
    parser.add_argument('--path2', type=str,
                        help='Input path for mask file')
    parser.add_argument('--axis', type=int,
                        help='Axis to flip along (0, 1, or 2) (DEFAULT: 2)')
    return parser.parse_args()

def plot_tSNR(path1, path2, axis):
    # Load fMRI data
    fmri_img = nib.load(path1)
    fmri_data = fmri_img.get_fdata()
    
    # Load mask data
    mask_img = nib.load(path2)
    mask_data = mask_img.get_fdata()

    # Convert mask to binary and flip along the specified axis
    binary_mask = (mask_data > 0).astype(np.float32)
    flipped_mask = np.flip(binary_mask, axis=axis)

    # Adjust dimensions if fMRI data is 4D
    if fmri_data.ndim == 4:
        flipped_mask = np.expand_dims(flipped_mask, axis=-1)

    # Apply the mask (element-wise multiplication)
    masked_data = fmri_data * flipped_mask

    # Compute tSNR (temporal Signal-to-Noise Ratio)
    mean_signal = np.mean(masked_data, axis=-1)  
    std_signal = np.std(masked_data, axis=-1)    
    tsnr = np.divide(mean_signal, std_signal, where=(std_signal != 0))  # Avoid division by zero
    print("Max tSNR ", np.max(tsnr))

    # Select slices to plot
    total_slices = tsnr.shape[2]
    middle_start = max(0, total_slices // 2 - 10)
    middle_end = min(total_slices, middle_start + 20)  
    slice_indices = np.arange(middle_start, middle_end)

    # Setup figure with 2 rows, 5 columns
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.ravel()

    # Plot tSNR slices
    for i, idx in enumerate(slice_indices):
        ax = axes[i]
        rotated_slice = np.rot90(tsnr[:, :, idx]) 
        im = ax.imshow(rotated_slice, cmap='hot', origin='lower')
        ax.set_title(f'Slice {idx}')
        ax.axis('off')

    # Hide unused subplots
    for i in range(len(slice_indices), len(axes)):
        axes[i].axis('off')

    # Adjust layout and add colorbar
    plt.tight_layout()
    fig.suptitle('Temporal SNR Across 10 Slices', fontsize=16, y=1.02)
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    out_path = re.sub(r"\.nii(\.gz)?$", ".png", path1)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    
def main():
    args = parse_args()
    path1 = args.path1
    path2 = args.path2
    if args.axis is None:
        axis = 2
    else:
        axis = args.axis
    print(f'Calculating tSNR for {path1}')
    plot_tSNR(path1, path2, axis)

if __name__ == '__main__':
    main()
