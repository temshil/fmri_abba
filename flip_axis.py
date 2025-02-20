# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:05:54 2024

@author: shili
"""

import nibabel as nii
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flip NIfTI images along a specified axis.")
    parser.add_argument('--in_path', type=str,
                        help='Input path for NIfTI file')
    parser.add_argument('--axis', type=int,
                        help='Axis to flip along (0, 1, or 2) (DEFAULT: 2)')
    return parser.parse_args()

def flip_nifti(in_path,axis):
    nifti_img = nii.load(in_path)
    data = nifti_img.get_fdata()
    flipped_data = np.flip(data, axis=axis) 
    flipped_img = nii.Nifti1Image(flipped_data, nifti_img.affine)
    flipped_img.update_header()
    if in_path.endswith(".nii"):
        out_path = in_path.replace(".nii", "_flipped.nii")
    else:
        out_path = in_path.replace(".nii.gz", "_flipped.nii.gz")
    nii.save(flipped_img, out_path)
    
def main():
    args = parse_args()
    in_path = args.in_path
    if args.axis is None:
        axis = 2
    else:
        axis = args.axis
    print(f'Flipping NIfTI images along axis {axis}...')
    flip_nifti(in_path, axis)


if __name__ == '__main__':
    main()