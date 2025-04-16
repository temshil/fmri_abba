import os
import nibabel as nib
import numpy as np
from scipy.stats import pearsonr, ttest_rel
from statsmodels.stats.multitest import multipletests

def filter_labels_with_files_multiple_dirs(text_file, directories):
    labels = []
    with open(text_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                label = parts[1].replace(',', '')
                labels.append(label)

    common_labels = []
    for label in labels:
        if all(os.path.isfile(os.path.join(dir_path, f"{label}.nii.gz")) for dir_path in directories):
            common_labels.append(label)
    return common_labels

def analyze_fmri(fmri_path, roi):
    fmri_img = nib.load(fmri_path)
    fmri_data = np.flip(fmri_img.get_fdata(), axis=2)

    mask_path = fmri_path.split(".")[0] + 'SmoothBet_mask.nii.gz'
    mask_data = nib.load(mask_path).get_fdata().astype(bool)
    flattened_data = fmri_data[mask_data]

    roi_path = os.path.join(os.path.dirname(fmri_path), "ROIs", f"{roi}.nii.gz")
    roi_mask = nib.load(roi_path).get_fdata().astype(bool)

    mask_flat = mask_data.flatten()
    roi_flat = roi_mask.flatten()
    roi_within_mask = roi_flat[mask_flat]

    roi_ts = flattened_data[roi_within_mask].mean(axis=0)
    correlations = np.array([pearsonr(roi_ts, voxel_ts)[0] for voxel_ts in flattened_data])

    correlation_map = np.zeros(mask_data.shape)
    correlation_map[mask_data] = correlations
    correlation_map = np.clip(correlation_map, -0.9999999999999999, 0.9999999999999999)
    z_map = 0.5 * np.log((1 + correlation_map) / (1 - correlation_map))

    return z_map

base_dir = "C:/Users/shili/Downloads/mridata/MOV10Project"
anno_path = os.path.join(base_dir, "annoVolume+2000rsfMRI.nii.txt")

sessions = ["before","after"]
subjects = ["DEL1", "DEL2", "DEL3", "DEL4", "WT1", "WT2", "WT3", "WT4"]
for ses in sessions:
    dirs = [
        os.path.join(base_dir, f"sub-{sub}", f"ses-{ses}", "func", "ROIs")
        for sub in subjects
    ]
    
    rois = filter_labels_with_files_multiple_dirs(anno_path, dirs)
    
    out_dir_rawp = os.path.join(base_dir, "..", "fmri_analysis", f"{ses}", "rawp")
    out_dir_adjp = os.path.join(base_dir, "..", "fmri_analysis", f"{ses}", "adjp")
    os.makedirs(out_dir_rawp, exist_ok=True)
    os.makedirs(out_dir_adjp, exist_ok=True)
    
    for roi in rois:
        zmaps_DEL = [analyze_fmri(os.path.join(base_dir, f"sub-DEL{i+1}", f"ses-{ses}", "func", f"sub-DEL{i+1}_ses-{ses}_seepi.nii.gz"), roi) for i in range(4)]
        zmaps_WT  = [analyze_fmri(os.path.join(base_dir, f"sub-WT{i+1}",  f"ses-{ses}", "func", f"sub-WT{i+1}_ses-{ses}_seepi.nii.gz"), roi) for i in range(4)]
    
        cond1_zmaps = np.stack(zmaps_DEL)
        cond2_zmaps = np.stack(zmaps_WT)
        
        all_zmaps = np.concatenate([cond1_zmaps, cond2_zmaps], axis=0)
    
        valid_voxel_mask = np.all(all_zmaps != 0, axis=0)
    
        cond1_masked = np.where(valid_voxel_mask, cond1_zmaps, np.nan)
        cond2_masked = np.where(valid_voxel_mask, cond2_zmaps, np.nan)
        
        t_vals, p_vals = ttest_rel(cond1_masked, cond2_masked, axis=0, nan_policy='omit')
        p_vals_flat = p_vals.flatten()
        _, pvals_corrected, _, _ = multipletests(p_vals_flat, method='fdr_bh')
        p_vals_fdr = pvals_corrected.reshape(p_vals.shape)
    
        p_thresh_map = np.where(p_vals < 0.05, p_vals, 0).astype(np.float32)
        fdr_thresh_map = np.where(p_vals_fdr < 0.05, p_vals_fdr, 0).astype(np.float32)
        
        fmri_img = nib.load(os.path.join(base_dir, "sub-DEL1", "ses-after", "func", "sub-DEL1_ses-after_seepi.nii.gz"))
        affine = fmri_img.affine
    
        rawp_img = nib.Nifti1Image(p_thresh_map, affine)
        adjp_img = nib.Nifti1Image(fdr_thresh_map, affine)
    
        nib.save(rawp_img, os.path.join(out_dir_rawp, f"{roi}_rawp.nii.gz"))
        nib.save(adjp_img, os.path.join(out_dir_adjp, f"{roi}_adjp.nii.gz"))