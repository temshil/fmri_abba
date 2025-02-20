# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:24:15 2025

@author: shili
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
import re
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Flip NIfTI images along a specified axis.")
    parser.add_argument('--project_dir', type=str,
                        help='Input path for the parent directory')
    parser.add_argument('--subject', type=str,
                        help='Input subject name')
    parser.add_argument('--session', type=str,
                        help='Input session name')
    parser.add_argument('--matrix', type=str,
                        help='Input matrix that you want to plot: Pearson coefficient (R), Z-transformed R (Z), p-values associated with R (P)')
    return parser.parse_args()

def plotmat(path, out_path):
    mat_data = loadmat(path)
    data = mat_data['matrix']
    arr = mat_data['label']
    
    def clean_labels(arr):
        return [re.sub(r"^.*?(L|R)", r"\1", label) for label in arr]

    arr_mod = clean_labels(arr)
    
    # Plot the heatmap
    plt.figure(figsize=(25, 20))
    plt.imshow(data, cmap='viridis', aspect='auto')  # You can choose other colormaps like 'plasma', 'inferno', etc.
    plt.colorbar(label='Color Scale')  # Adds a color scale

    plt.xticks(ticks=np.arange(data.shape[1]), labels=arr_mod, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(data.shape[0]), labels=arr_mod)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')

def main():
    args = parse_args()
    project_dir = args.project_dir
    subject = args.subject
    session = args.session
    matrix = args.matrix
    data_path = f'{project_dir}/sub-{subject}/ses-{session}'
    path = f'{data_path}/func/regr/Matrix_Pcorr{matrix}_Split.sub-{subject}.mat'
    print(f'Saving the plot for {path}')
    out_path = re.sub(r"\.mat?$", ".png", path)
    plotmat(path,out_path)
    
    
if __name__ == '__main__':
    main()