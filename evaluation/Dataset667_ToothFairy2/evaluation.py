import glob
import os
import SimpleITK as sitk
import numpy as np
import argparse
from medpy.metric import binary
from nnunetv2.paths import nnUNet_raw
from tqdm import tqdm
import surface_distance as sd

# Updated dataset name
DATASET = "Dataset667_ToothFairy2_97cases"

# Labels as specified in the dataset.json file
LABELS = {
    11: "Upper Right Central Incisor",
    12: "Upper Right Lateral Incisor",
    13: "Upper Right Canine",
    14: "Upper Right First Premolar",
    15: "Upper Right Second Premolar",
    16: "Upper Right First Molar",
    17: "Upper Right Second Molar",
    18: "Upper Right Third Molar (Wisdom Tooth)",
    # Skipping NA labels (19, 20, 29, 30, 39, 40)
    21: "Upper Left Central Incisor",
    22: "Upper Left Lateral Incisor",
    23: "Upper Left Canine",
    24: "Upper Left First Premolar",
    25: "Upper Left Second Premolar",
    26: "Upper Left First Molar",
    27: "Upper Left Second Molar",
    28: "Upper Left Third Molar (Wisdom Tooth)",
    31: "Lower Left Central Incisor",
    32: "Lower Left Lateral Incisor",
    33: "Lower Left Canine",
    34: "Lower Left First Premolar",
    35: "Lower Left Second Premolar",
    36: "Lower Left First Molar",
    37: "Lower Left Second Molar",
    38: "Lower Left Third Molar (Wisdom Tooth)",
    41: "Lower Right Central Incisor",
    42: "Lower Right Lateral Incisor",
    43: "Lower Right Canine",
    44: "Lower Right First Premolar",
    45: "Lower Right Second Premolar",
    46: "Lower Right First Molar",
    47: "Lower Right Second Molar",
    48: "Lower Right Third Molar (Wisdom Tooth)"
}

def read_file(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def read_file_and_spacing(path):
    file = sitk.ReadImage(path)
    spacing = file.GetSpacing()
    return sitk.GetArrayFromImage(file), spacing

def dice(pred, label):
    pred = pred.astype(bool)
    label = label.astype(bool)
    if pred.sum() + label.sum() == 0:
        return 1.0
    return 2.0 * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())

def masked_dice_score(pred, gt, mask):
    """
    Compute the masked Dice score between prediction and ground truth within a given mask.

    Parameters:
    - pred (np.ndarray): Binary predicted segmentation (0 or 1).
    - gt (np.ndarray): Binary ground truth segmentation (0 or 1).
    - mask (np.ndarray): Binary mask (1 = include voxel, 0 = exclude voxel).

    Returns:
    - dice (float): Masked Dice score. Returns NaN if the mask region has no valid voxels.
    """
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)

    # Apply mask
    masked_pred = pred * mask
    masked_gt = gt * mask

    intersection = np.sum(masked_pred * masked_gt)
    pred_sum = np.sum(masked_pred)
    gt_sum = np.sum(masked_gt)

    if pred_sum == 0 and gt_sum == 0:
        return np.nan

    dice = 2.0 * intersection / (pred_sum + gt_sum)
    return dice


def hd(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = binary.hd95(pred, gt)
        return hd95
    else:
        return np.nan  # Return NaN if no prediction or ground truth is present'''

def hd95(pred, gt, voxelspacing):
    if pred.sum() > 0 and gt.sum() > 0:
        # Compute the surface distances
        surface_distances = sd.compute_surface_distances(gt, pred, voxelspacing)
        # Compute the 95th percentile distance
        hd95 = sd.compute_robust_hausdorff(surface_distances, percent=95)
        return hd95
    else:
        return np.nan  # Return NaN if no prediction or ground truth is present

def test(model, fold, test_set, dataset):

    test_dir = os.path.join("/work/grana_maxillo/Mamba3DMedModels/data/test_sets", test_set)
    infer_path = f'inferTs_{model}_{test_set}'

    label_list_mha = sorted(glob.glob(os.path.join(test_dir, 'labels', '*mha')))
    label_list_nifti = sorted(glob.glob(os.path.join(test_dir, 'labels', '*nii.gz')))
    label_list = label_list_mha + label_list_nifti

    infer_list_mha = sorted(glob.glob(os.path.join(nnUNet_raw, dataset, infer_path, '*mha')))
    infer_list_nifti = sorted(glob.glob(os.path.join(nnUNet_raw, dataset, infer_path, '*nii.gz')))
    infer_list = infer_list_mha + infer_list_nifti
    print(f'Found {len(label_list)} labels and {len(infer_list)} predictions')
    print("Loading data...")

    # Initialize dictionaries to store metrics per label
    dice_scores = {label_id: [] for label_id in LABELS.keys()}
    hd95_scores = {label_id: [] for label_id in LABELS.keys()}

    # output_dir = os.path.join(nnUNet_raw, DATASET, infer_path, fold)
    output_dir = os.path.join(nnUNet_raw, dataset, infer_path)
    os.makedirs(output_dir, exist_ok=True)
    fw = open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'w')

    excluded_classes = [8, 9, 10]

    for label_path, infer_path in tqdm(zip(label_list, infer_list), total=len(label_list)):
        label = read_file(label_path)
        infer, spacing = read_file_and_spacing(infer_path)

        fw.write('*' * 40 + '\n')
        fw.write(f'File: {os.path.basename(infer_path)}\n')

        # "valid_voxels" is True if the gt voxel is NOT one of the excluded classes
        valid_voxels = ~np.isin(label, excluded_classes)
        valid_voxels_mask = valid_voxels.astype(np.uint8)

        for label_id, label_name in LABELS.items():
            gt_binary = (label == label_id).astype(np.uint8)
            pred_binary = (infer == label_id).astype(np.uint8)

            #dice_score = dice(pred_binary, gt_binary)
            #hd95_score = hd95(pred_binary, gt_binary, spacing)

            # 3) Compute masked Dice for the target class
            dice_val = masked_dice_score(pred_binary, gt_binary, valid_voxels_mask)

            dice_scores[label_id].append(dice_val)
            #hd95_scores[label_id].append(hd95_score)

            fw.write(f'Label {label_id} ({label_name}):\n')
            fw.write(f'  Dice Score: {dice_val:.4f}\n')
            '''if not np.isnan(hd95_score):
                fw.write(f'  HD95: {hd95_score:.4f}\n')
            else:
                fw.write('  HD95: Undefined (empty prediction or ground truth)\n')'''

        fw.write('*' * 40 + '\n\n')

    # Calculate average metrics per label
    fw.write('=' * 40 + '\n')
    fw.write('Average Metrics per Label:\n')

    for label_id, label_name in LABELS.items():
        avg_dice = np.nanmean(dice_scores[label_id])
        avg_hd95 = np.nanmean(hd95_scores[label_id])

        fw.write(f'Label {label_id} ({label_name}):\n')
        fw.write(f'  Average Dice Score: {avg_dice:.4f}\n')
        if not np.isnan(avg_hd95):
            fw.write(f'  Average HD95: {avg_hd95:.4f}\n')
        else:
            fw.write('  Average HD95: Undefined\n')

    # Calculate overall average Dice and HD95
    all_dice_scores = [score for scores in dice_scores.values() for score in scores if not np.isnan(score)]
    all_hd95_scores = [score for scores in hd95_scores.values() for score in scores if not np.isnan(score)]

    overall_avg_dice = np.mean(all_dice_scores)
    overall_avg_hd95 = np.mean(all_hd95_scores) if all_hd95_scores else np.nan

    fw.write('=' * 40 + '\n')
    fw.write(f'Overall Average Dice Score: {overall_avg_dice:.4f}\n')
    if not np.isnan(overall_avg_hd95):
        fw.write(f'Overall Average HD95: {overall_avg_hd95:.4f}\n')
    else:
        fw.write('Overall Average HD95: Undefined\n')

    fw.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name")
    parser.add_argument("fold", help="fold name")
    parser.add_argument("config", help="config")
    parser.add_argument("test", help="test set", choices=["Radboud", "Cui", "Cui_15"])
    parser.add_argument("dataset", help="Exact name of the dataset")

    args = parser.parse_args()

    test(args.model, args.fold, args.test, args.dataset)
