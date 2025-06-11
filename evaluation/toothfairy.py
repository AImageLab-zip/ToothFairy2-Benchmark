import sys
import glob
import os
import SimpleITK as sitk
import numpy as np
from medpy.metric import binary
import argparse
from tqdm import tqdm

def read_nii(path):
    itk_img = sitk.ReadImage(path)
    spacing = np.array(itk_img.GetSpacing())
    return sitk.GetArrayFromImage(itk_img), spacing

def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())

def process_label(label):
    iac = label == 1
    return iac

def hd(pred,gt):
    if pred.sum() > 0 and gt.sum()>0:
        hd95 = binary.hd95(pred, gt)
        return  hd95
    else:
        return 0

def test(dataset_folder, inferts_folder):
    if dataset_folder[0] == '/':
        path = dataset_folder
    else:
        path = os.path.join('/work/grana_maxillo/ECCV_MICCAI/U-Mamba/data/nnUNet_raw/', dataset_folder)

    label_path = os.path.join(path, 'labelsTs', '*.nii.gz')
    infer_path = os.path.join(path, inferts_folder, '0', '*nii.gz')
    print(f'Looking into {label_path=} and {infer_path=}')

    label_list = sorted(glob.glob(label_path))
    infer_list = sorted(glob.glob(infer_path))
    print(f'Found {len(label_list)} labels and {len(infer_list)} predictions')

    dice_IAC=[]
    hd_IAC=[]
    
    output_file = os.path.join(path, inferts_folder, '0', 'dice_pre.txt')
    fw = open(output_file, 'w')
    # fw = sys.stdout
    
    for label_path, infer_path in tqdm(zip(label_list,infer_list), total=len(label_list)):
        label, spacing = read_nii(label_path)
        infer, spacing = read_nii(infer_path)
        label_iac = process_label(label)
        infer_iac = process_label(infer)
        
        dice_IAC.append(dice(label_iac, infer_iac))
        hd_IAC.append(hd(label_iac, infer_iac))
    
    fw.write(f'Mean_Dice_IAC: {np.mean(dice_IAC)}\n')
    fw.write(f'Mean_HD_IAC: {np.mean(hd_IAC)}\n')

    fw.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", help="dataset folder")
    parser.add_argument("inferTs_folder", help="inferTs folder")
    args = parser.parse_args()
    dataset_folder = args.dataset_folder
    inferts_folder = args.inferTs_folder
    test(dataset_folder, inferts_folder)
