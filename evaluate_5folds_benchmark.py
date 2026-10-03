#!/usr/bin/env python3
"""
Evaluate and Benchmark 5-Fold Cross Validation for PICK against nnU-Net.
Calculates Out-of-Fold (OOF) Mean Dice +/- Std across all 5 folds.
Outputs detailed class-wise metrics and comparative benchmark tables.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataloaders.dataset_bhsd import BHSDDataset
from networks.net_factory import net_factory
from test_util import calculate_metric_percase

CLASS_NAMES = ["Background", "Epidural", "Intraparenchymal", "Intraventricular", "Subarachnoid", "Subdural"]


def load_model_weights(model, pth_path):
    state = torch.load(str(pth_path), map_location='cuda:0')
    if 'net' in state:
        model.load_state_dict(state['net'])
    else:
        model.load_state_dict(state)
    return model


def evaluate_single_case_sliding_window(model, image, num_classes, patch_size=(32, 160, 160), stride_z=8, stride_xy=32):
    """Run 3D sliding window inference."""
    z, y, x = image.shape
    pz = max(patch_size[0] - z, 0)
    py = max(patch_size[1] - y, 0)
    px = max(patch_size[2] - x, 0)
    if pz > 0 or py > 0 or px > 0:
        image = np.pad(image, [(0, pz), (0, py), (0, px)], mode='constant', constant_values=0)

    w_z, w_y, w_x = image.shape
    sz = max(int(np.ceil((w_z - patch_size[0]) / stride_z)) + 1, 1)
    sy = max(int(np.ceil((w_y - patch_size[1]) / stride_xy)) + 1, 1)
    sx = max(int(np.ceil((w_x - patch_size[2]) / stride_xy)) + 1, 1)

    score_map = np.zeros((num_classes, w_z, w_y, w_x), dtype=np.float32)
    cnt = np.zeros((w_z, w_y, w_x), dtype=np.float32)

    for iz in range(sz):
        zs = min(iz * stride_z, w_z - patch_size[0])
        for iy in range(sy):
            ys = min(iy * stride_xy, w_y - patch_size[1])
            for ix in range(sx):
                xs = min(ix * stride_xy, w_x - patch_size[2])

                patch = image[zs:zs + patch_size[0], ys:ys + patch_size[1], xs:xs + patch_size[2]]
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).cuda().float()

                _, out, _ = model(patch_tensor)
                prob = F.softmax(out, dim=1).squeeze(0).cpu().numpy()

                score_map[:, zs:zs + patch_size[0], ys:ys + patch_size[1], xs:xs + patch_size[2]] += prob
                cnt[zs:zs + patch_size[0], ys:ys + patch_size[1], xs:xs + patch_size[2]] += 1.0

    score_map = score_map / np.expand_dims(cnt, axis=0)
    pred = np.argmax(score_map, axis=0)[:z, :y, :x]
    return pred


def cal_multiclass_dice(pred, gt, num_classes=6):
    class_dices = []
    for c in range(1, num_classes):
        p_c = (pred == c).astype(np.float32)
        g_c = (gt == c).astype(np.float32)
        intersection = np.sum(p_c * g_c)
        total = np.sum(p_c) + np.sum(g_c)
        if total == 0:
            dice = 1.0
        else:
            dice = (2.0 * intersection) / total
        class_dices.append(dice)
    return class_dices


def benchmark_experiment(exp_dir, root_path, task='binary', model_name='VNet', gpu='0'):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    is_binary = (task == 'binary')
    num_classes = 2 if is_binary else 6
    patch_size = (32, 160, 160)

    print("==========================================================================")
    print(f"=== BENCHMARK 5-FOLD CROSS VALIDATION: {os.path.basename(exp_dir)} ===")
    print(f"=== Task: {task.upper()} ({num_classes} classes) | Model: {model_name} ===")
    print("==========================================================================")

    fold_results = {}
    all_fold_dices = []
    per_class_all_folds = []  # Only for multiclass

    for fold in range(5):
        fold_sub = os.path.join(exp_dir, f"fold_{fold}")
        
        # Candidate checkpoint paths (preference to self_train best model)
        candidates = [
            os.path.join(fold_sub, "self_train", f"{model_name}_best_model.pth"),
            os.path.join(fold_sub, f"{model_name}_best_model.pth"),
            os.path.join(fold_sub, "pre_train", f"{model_name}_best_model.pth"),
        ]
        
        ckpt_path = None
        for cand in candidates:
            if os.path.exists(cand):
                ckpt_path = cand
                break

        if ckpt_path is None:
            print(f"[WARN] No checkpoint found for fold {fold} in {fold_sub}, skipping.")
            continue

        print(f"\n[FOLD {fold}] Loading checkpoint: {ckpt_path}")
        model = net_factory(net_type=model_name, in_chns=1, class_num=num_classes, mode="test")
        model = nn.DataParallel(model).cuda()
        model = load_model_weights(model, ckpt_path)
        model.eval()

        val_dataset = BHSDDataset(
            base_dir=root_path,
            split='val',
            binary=is_binary,
            patch_size=patch_size,
            fold=fold
        )

        fold_case_dices = []
        fold_per_class_list = []

        with torch.no_grad():
            for i in tqdm(range(len(val_dataset)), desc=f"Evaluating Fold {fold}"):
                sample, _ = val_dataset[i]
                image, label = sample['image'], sample['label']

                pred = evaluate_single_case_sliding_window(
                    model=model,
                    image=image,
                    num_classes=num_classes,
                    patch_size=patch_size
                )

                if is_binary:
                    gt_bin = (label > 0).astype(np.uint8)
                    pred_bin = (pred > 0).astype(np.uint8)
                    if np.sum(gt_bin) == 0 and np.sum(pred_bin) == 0:
                        d = 1.0
                    elif np.sum(gt_bin) == 0 or np.sum(pred_bin) == 0:
                        d = 0.0
                    else:
                        d = calculate_metric_percase(pred_bin, gt_bin)[0]
                    fold_case_dices.append(d)
                else:
                    c_dices = cal_multiclass_dice(pred, label, num_classes=num_classes)
                    fold_per_class_list.append(c_dices)
                    fold_case_dices.append(float(np.mean(c_dices)))

        fold_mean_dice = float(np.mean(fold_case_dices))
        fold_results[f"fold_{fold}"] = {
            "mean_dice": fold_mean_dice,
            "num_cases": len(fold_case_dices)
        }
        all_fold_dices.append(fold_mean_dice)

        if not is_binary and fold_per_class_list:
            fold_cls_mean = np.mean(np.array(fold_per_class_list), axis=0).tolist()
            fold_results[f"fold_{fold}"]["per_class_dice"] = {
                CLASS_NAMES[c+1]: fold_cls_mean[c] for c in range(5)
            }
            per_class_all_folds.append(fold_cls_mean)

        print(f"--> [FOLD {fold} RESULT] Mean Dice: {fold_mean_dice:.4f}")

    if not all_fold_dices:
        print("[ERROR] No fold evaluations completed. Please check experiment directory.")
        return

    overall_mean = float(np.mean(all_fold_dices))
    overall_std = float(np.std(all_fold_dices))

    summary = {
        "experiment": os.path.basename(exp_dir),
        "task": task,
        "model": model_name,
        "completed_folds": len(all_fold_dices),
        "overall_mean_dice": overall_mean,
        "overall_std_dice": overall_std,
        "folds": fold_results
    }

    if not is_binary and per_class_all_folds:
        cls_5fold_mean = np.mean(np.array(per_class_all_folds), axis=0)
        cls_5fold_std = np.std(np.array(per_class_all_folds), axis=0)
        summary["subclass_benchmark"] = {
            CLASS_NAMES[c+1]: {
                "mean_dice": float(cls_5fold_mean[c]),
                "std_dice": float(cls_5fold_std[c])
            }
            for c in range(5)
        }

    # Save summary
    out_json = os.path.join(exp_dir, "benchmark_5folds_summary.json")
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=4)

    # Print Formatted Report
    print("\n" + "=" * 70)
    print("=== 5-FOLD CROSS VALIDATION BENCHMARK REPORT ===")
    print("=" * 70)
    print(f"Experiment: {os.path.basename(exp_dir)}")
    print(f"Task:       {task.upper()}")
    print("-" * 70)
    for f in range(5):
        key = f"fold_{f}"
        if key in fold_results:
            d = fold_results[key]['mean_dice']
            print(f"Fold {f}:  Dice = {d:.4f} ({fold_results[key]['num_cases']} cases)")
    print("-" * 70)
    print(f"FINAL 5-FOLD SCORE: {overall_mean:.4f} ± {overall_std:.4f}")
    print("=" * 70)

    if not is_binary and "subclass_benchmark" in summary:
        print("\n--- Subclass Breakdown (5-Fold Mean ± Std) ---")
        for cls_name, vals in summary["subclass_benchmark"].items():
            print(f"  * {cls_name:<20}: {vals['mean_dice']:.4f} ± {vals['std_dice']:.4f}")
        print("=" * 70)

    print(f"\n[INFO] Full benchmark report saved to: {out_json}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate 5-Fold Benchmark for PICK")
    parser.add_argument('--exp_dir', type=str, default='/home/u001015/experiments/BHSD_binary_PICK_5folds',
                        help='Path to experiment directory containing fold_0..fold_4')
    parser.add_argument('--root_path', type=str, default='/home/u001015/dataset/BHSD_h5/',
                        help='Path to BHSD H5 dataset directory')
    parser.add_argument('--task', type=str, default='binary', choices=['binary', 'multiclass'],
                        help='Task type: binary or multiclass')
    parser.add_argument('--model', type=str, default='VNet', help='Model name')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    args = parser.parse_args()

    benchmark_experiment(
        exp_dir=args.exp_dir,
        root_path=args.root_path,
        task=args.task,
        model_name=args.model,
        gpu=args.gpu
    )


if __name__ == "__main__":
    main()
