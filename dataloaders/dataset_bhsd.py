import os
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import random
from scipy.ndimage import zoom


class MaskGeneratorBHSD:
    """
    Mask Generator for Masked Image Modeling (MIM) on non-cubic patches (e.g. 32 x 160 x 160).
    Divides the volume into 3D tokens and generates random binary masks with a specified mask ratio.
    """
    def __init__(self, input_size=(32, 160, 160), mask_patch_size=(8, 16, 16), mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.mask_ratio = mask_ratio

        assert input_size[0] % mask_patch_size[0] == 0, f"{input_size[0]} not divisible by {mask_patch_size[0]}"
        assert input_size[1] % mask_patch_size[1] == 0, f"{input_size[1]} not divisible by {mask_patch_size[1]}"
        assert input_size[2] % mask_patch_size[2] == 0, f"{input_size[2]} not divisible by {mask_patch_size[2]}"

        self.rand_size = (
            input_size[0] // mask_patch_size[0],
            input_size[1] // mask_patch_size[1],
            input_size[2] // mask_patch_size[2],
        )
        self.token_count = self.rand_size[0] * self.rand_size[1] * self.rand_size[2]
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape(self.rand_size)
        mask = mask.repeat(self.mask_patch_size[0], axis=0) \
                   .repeat(self.mask_patch_size[1], axis=1) \
                   .repeat(self.mask_patch_size[2], axis=2)
        return mask


class BHSDDataset(Dataset):
    """
    BHSD Dataset loader for PICK semi-supervised segmentation.
    Supports both Binary (binary=True) and Multi-class (binary=False) tasks
    on the same underlying HDF5 dataset.
    """
    def __init__(self, base_dir, split='train', num=None, transform=None, binary=False,
                 patch_size=(32, 160, 160), fold=None):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        self.binary = binary
        self.patch_size = patch_size
        self.fold = fold
        self.mask_generator = MaskGeneratorBHSD(input_size=patch_size)

        list_file = None
        if fold is not None:
            fold_file = os.path.join(base_dir, "splits", f"fold{fold}_{split}.txt")
            if os.path.exists(fold_file):
                list_file = fold_file

        if list_file is None:
            list_file = os.path.join(base_dir, f"{split}.txt")
            if not os.path.exists(list_file):
                alt_map = {'train': 'train.list', 'val': 'val.list', 'test': 'test.list'}
                list_file = os.path.join(base_dir, alt_map.get(split, f"{split}.txt"))

        with open(list_file, 'r') as f:
            self.image_list = [line.strip() for line in f if line.strip()]

        if num is not None and split == 'train':
            self.image_list = self.image_list[:num]

        task_type = "Binary (Hemorrhage vs Background)" if self.binary else "Multi-class (6 classes: 0..5)"
        fold_str = f"Fold {fold}" if fold is not None else "Default Split"
        print(f"[BHSDDataset] {fold_str} | Split '{split}' | Task: {task_type} | Samples: {len(self.image_list)}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        case_name = self.image_list[idx]
        h5_path = os.path.join(self.base_dir, "data", f"{case_name}.h5")

        with h5py.File(h5_path, 'r') as h5f:
            image = h5f['image'][:]
            label = h5f['label'][:]

        # On-the-fly binary conversion if requested
        if self.binary:
            label = (label > 0).astype(np.uint8)
        else:
            label = label.astype(np.uint8)

        sample = {'image': image, 'label': label, 'case': case_name}
        if self.transform:
            sample = self.transform(sample)

        mask = self.mask_generator()
        return sample, mask


# =========================================================================
# 3D Data Augmentation Transforms
# =========================================================================

class RandomRotFlip(object):
    """Randomly rotate and flip the 3D volume along spatial dimensions."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k, axes=(1, 2))
        label = np.rot90(label, k, axes=(1, 2))
        axis = np.random.randint(1, 3)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        sample['image'] = image
        sample['label'] = label
        return sample


class RandomCrop(object):
    """
    Random crop of a 3D volume to target output_size (Z, Y, X).
    Applies symmetric zero-padding if any dimension of the volume is smaller than the target.
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        z, y, x = image.shape

        # Pad if volume is smaller than patch size
        pz = max(self.output_size[0] - z, 0)
        py = max(self.output_size[1] - y, 0)
        px = max(self.output_size[2] - x, 0)

        if pz > 0 or py > 0 or px > 0:
            pad_z = (pz // 2, pz - pz // 2)
            pad_y = (py // 2, py - py // 2)
            pad_x = (px // 2, px - px // 2)
            image = np.pad(image, [pad_z, pad_y, pad_x], mode='constant', constant_values=0)
            label = np.pad(label, [pad_z, pad_y, pad_x], mode='constant', constant_values=0)
            z, y, x = image.shape

        # Random crop starting coordinates
        z1 = np.random.randint(0, max(z - self.output_size[0] + 1, 1))
        y1 = np.random.randint(0, max(y - self.output_size[1] + 1, 1))
        x1 = np.random.randint(0, max(x - self.output_size[2] + 1, 1))

        sample['image'] = image[z1:z1 + self.output_size[0], y1:y1 + self.output_size[1], x1:x1 + self.output_size[2]]
        sample['label'] = label[z1:z1 + self.output_size[0], y1:y1 + self.output_size[1], x1:x1 + self.output_size[2]]
        return sample


class CenterCrop(object):
    """Center crop of a 3D volume to target output_size (Z, Y, X)."""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        z, y, x = image.shape

        pz = max(self.output_size[0] - z, 0)
        py = max(self.output_size[1] - y, 0)
        px = max(self.output_size[2] - x, 0)

        if pz > 0 or py > 0 or px > 0:
            pad_z = (pz // 2, pz - pz // 2)
            pad_y = (py // 2, py - py // 2)
            pad_x = (px // 2, px - px // 2)
            image = np.pad(image, [pad_z, pad_y, pad_x], mode='constant', constant_values=0)
            label = np.pad(label, [pad_z, pad_y, pad_x], mode='constant', constant_values=0)
            z, y, x = image.shape

        z1 = (z - self.output_size[0]) // 2
        y1 = (y - self.output_size[1]) // 2
        x1 = (x - self.output_size[2]) // 2

        sample['image'] = image[z1:z1 + self.output_size[0], y1:y1 + self.output_size[1], x1:x1 + self.output_size[2]]
        sample['label'] = label[z1:z1 + self.output_size[0], y1:y1 + self.output_size[1], x1:x1 + self.output_size[2]]
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors and add channel dimension to image."""
    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        sample['image'] = torch.from_numpy(image)
        sample['label'] = torch.from_numpy(sample['label'].astype(np.int64))
        return sample
