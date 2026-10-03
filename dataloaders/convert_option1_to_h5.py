#!/usr/bin/env python3
"""
Convert nnUNet Option 1 preprocessed data (.b2nd / .pkl) to HDF5 (.h5) format for PICK.
This script is strictly READ-ONLY on input files and writes new .h5 files to output_dir.
It preserves the original 6-class labels (0..5) so that both Binary and Multi-class
tasks can share the same dataset.
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import h5py
from tqdm import tqdm

try:
    import blosc2
    HAS_BLOSC2 = True
except ImportError:
    HAS_BLOSC2 = False


def load_b2nd_array(file_path):
    """Load an array from a blosc2 file (.b2nd)."""
    if not HAS_BLOSC2:
        raise ImportError(
            "The 'blosc2' Python library is required to read .b2nd files. "
            "Please install it with: pip install blosc2"
        )
    # blosc2.open provides a ndarray-like interface
    try:
        arr = blosc2.open(file_path, mode='r')
        return arr[:]
    except Exception as e:
        # Fallback to blosc2.load_array if available
        try:
            return blosc2.load_array(file_path)
        except Exception:
            raise RuntimeError(f"Failed to read blosc2 file {file_path}: {e}")


def convert_dataset(input_dir, output_dir, splits_file=None, fold=0):
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    
    # 1. Discover cases from .b2nd files
    # Option 1 saves images as <case_id>.b2nd and segs as <case_id>_seg.b2nd
    all_b2nd = glob.glob(os.path.join(input_dir, "*.b2nd"))
    seg_files = set(glob.glob(os.path.join(input_dir, "*_seg.b2nd")))
    image_files = [f for f in all_b2nd if f not in seg_files]
    
    print(f"[INFO] Found {len(image_files)} image files and {len(seg_files)} segmentation files in {input_dir}")
    if len(image_files) == 0:
        print("[ERROR] No image .b2nd files found! Please verify the input directory path.")
        return

    converted_cases = []
    failed_cases = []

    for img_path in tqdm(image_files, desc="Converting to .h5"):
        case_id = os.path.basename(img_path).replace(".b2nd", "")
        seg_path = os.path.join(input_dir, f"{case_id}_seg.b2nd")
        
        if not os.path.exists(seg_path):
            print(f"[WARN] Segmentation file missing for {case_id}, skipping.")
            failed_cases.append(case_id)
            continue
            
        out_h5_path = os.path.join(output_dir, "data", f"{case_id}.h5")
        
        try:
            # Read image and seg
            img_data = load_b2nd_array(img_path)
            seg_data = load_b2nd_array(seg_path)
            
            # Squeeze channel dim if present (e.g., shape (1, Z, Y, X) -> (Z, Y, X))
            if img_data.ndim == 4 and img_data.shape[0] == 1:
                img_data = img_data[0]
            if seg_data.ndim == 4 and seg_data.shape[0] == 1:
                seg_data = seg_data[0]
                
            img_data = img_data.astype(np.float32)
            seg_data = seg_data.astype(np.uint8)
            
            # Write to HDF5
            with h5py.File(out_h5_path, 'w') as h5f:
                h5f.create_dataset('image', data=img_data, compression='gzip')
                h5f.create_dataset('label', data=seg_data, compression='gzip')
                
            converted_cases.append(case_id)
        except Exception as e:
            print(f"[ERROR] Failed to convert {case_id}: {e}")
            failed_cases.append(case_id)

    print(f"\n[SUMMARY] Successfully converted: {len(converted_cases)} cases.")
    if failed_cases:
        print(f"[SUMMARY] Failed: {len(failed_cases)} cases.")

    # 2. Generate Split text files for all 5 folds
    splits_dir = os.path.join(output_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    if splits_file and os.path.exists(splits_file):
        print(f"[INFO] Parsing all folds from {splits_file}")
        with open(splits_file, 'r') as sf:
            splits_data = json.load(sf)

        if isinstance(splits_data, list):
            num_folds = len(splits_data)
            print(f"[INFO] Found {num_folds} folds in {splits_file}")
            for f_idx, fold_info in enumerate(splits_data):
                raw_train = set(fold_info.get('train', []))
                raw_val = set(fold_info.get('val', []))

                f_train = [c for c in converted_cases if c in raw_train]
                f_val = [c for c in converted_cases if c in raw_val]

                f_train_path = os.path.join(splits_dir, f"fold{f_idx}_train.txt")
                f_val_path = os.path.join(splits_dir, f"fold{f_idx}_val.txt")

                with open(f_train_path, 'w') as f:
                    for c in sorted(f_train):
                        f.write(f"{c}\n")
                with open(f_val_path, 'w') as f:
                    for c in sorted(f_val):
                        f.write(f"{c}\n")

                print(f"  -> Fold {f_idx}: {len(f_train)} train, {len(f_val)} val cases saved to splits/")

            # Also create default train.txt / val.txt pointing to fold 0
            default_fold = fold if fold < num_folds else 0
            default_train_src = os.path.join(splits_dir, f"fold{default_fold}_train.txt")
            default_val_src = os.path.join(splits_dir, f"fold{default_fold}_val.txt")
            import shutil
            shutil.copy(default_train_src, os.path.join(output_dir, "train.txt"))
            shutil.copy(default_val_src, os.path.join(output_dir, "val.txt"))
            shutil.copy(default_val_src, os.path.join(output_dir, "test.txt"))
    else:
        print("[INFO] Splits file not provided or found. Generating default 80/20 train/val split.")
        converted_cases_sorted = sorted(converted_cases)
        split_idx = int(len(converted_cases_sorted) * 0.8)
        train_cases = converted_cases_sorted[:split_idx]
        val_cases = converted_cases_sorted[split_idx:]

        with open(os.path.join(output_dir, "train.txt"), 'w') as f:
            for c in train_cases:
                f.write(f"{c}\n")
        with open(os.path.join(output_dir, "val.txt"), 'w') as f:
            for c in val_cases:
                f.write(f"{c}\n")
        with open(os.path.join(output_dir, "test.txt"), 'w') as f:
            for c in val_cases:
                f.write(f"{c}\n")

    print(f"[SUCCESS] Data conversion and 5-fold splits completed! All files saved in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert Option 1 nnUNet preprocessed data to HDF5 for PICK")
    parser.add_argument("--input_dir", type=str, 
                        default="/home/u001015/nnUNet_data/nnUNet_preprocessed/Dataset001_BHSD/nnUNetPlans_Option1_CTClip_3d_fullres",
                        help="Path to nnUNet Option 1 preprocessed 3d_fullres directory")
    parser.add_argument("--output_dir", type=str,
                        default="/home/u001015/dataset/BHSD_h5",
                        help="Path to directory where .h5 and split files will be created")
    parser.add_argument("--splits_file", type=str,
                        default="/home/u001015/nnUNet_data/nnUNet_preprocessed/Dataset001_BHSD/splits_multilabel.json",
                        help="Path to splits_multilabel.json or splits_final.json")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold index to extract for train/val split (default: 0)")
    args = parser.parse_args()

    convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        splits_file=args.splits_file,
        fold=args.fold
    )


if __name__ == "__main__":
    main()
