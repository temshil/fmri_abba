# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:33:43 2025

@author: shili
"""

import nibabel as nib
import numpy as np
import pandas as pd

anat_atlas = nib.load('C:/Users/shili/Downloads/ARA_annotationR+2000.nii.gz')

anat_data = anat_atlas.get_fdata()

labels = np.unique(anat_data)

# Load the text file (assuming space or tab as a delimiter)
anat_annot = pd.read_csv("C:/Users/shili/Downloads/ARA_annotationR+2000.nii.txt", sep="\t", header=None, dtype=str)  # Ensure strings are read properly

# Filter rows where the second column contains the substring (case-sensitive)
l_amyg_annot = anat_annot[anat_annot[1].str.contains(r"^L_.*amyg.*nucleus", na=False, case=False)]  

l_amyg_annot.loc[:, 0] = l_amyg_annot[0].astype(float)

l_amyg_annot = l_amyg_annot.loc[l_amyg_annot[0].isin(labels)]

l_amyg_mask = np.isin(anat_data, l_amyg_annot[0])

r_amyg_annot = anat_annot[anat_annot[1].str.contains(r"^R_.*amyg.*nucleus", na=False, case=False)]  

r_amyg_annot.loc[:, 0] = r_amyg_annot[0].astype(float)

r_amyg_annot = r_amyg_annot.loc[r_amyg_annot[0].isin(labels)]

r_amyg_mask = np.isin(anat_data, r_amyg_annot[0])

fmri_atlas = nib.load('C:/Users/shili/Downloads/annoVolume+2000_rsfMRI.nii.gz')
fmri_data = fmri_atlas.get_fdata()

fmri_data[l_amyg_mask] = 191
fmri_data[r_amyg_mask] = 2191

new_fmri_atlas = nib.Nifti1Image(fmri_data, fmri_atlas.affine, fmri_atlas.header)
nib.save(new_fmri_atlas, 'C://Users//shili//Downloads//annoVolume+2000_rsfMRI.nii.gz')

fmri_atlas = nib.load('C:/Users/shili/Downloads/annoVolume.nii.gz')
fmri_data = fmri_atlas.get_fdata()

fmri_data[l_amyg_mask] = 191
fmri_data[r_amyg_mask] = 191

new_fmri_atlas = nib.Nifti1Image(fmri_data, fmri_atlas.affine, fmri_atlas.header)
nib.save(new_fmri_atlas, 'C://Users//shili//Downloads//annoVolume.nii.gz')