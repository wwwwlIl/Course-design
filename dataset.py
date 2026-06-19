"""
BraTS数据集加载与数据增强
支持3D MRI数据处理
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from scipy import ndimage
import random


class BraTSDataset(Dataset):
    """BraTS 2021 数据集"""

    def __init__(self, data_dir, transform=None, patch_size=(128, 128, 128), train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.patch_size = patch_size
        self.train = train

        self.case_list = sorted([d for d in os.listdir(data_dir)
                                 if os.path.isdir(os.path.join(data_dir, d))
                                 and d.startswith('BraTS2021')])

        print(f"加载数据集: {len(self.case_list)} 个病例")

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        case_path = os.path.join(self.data_dir, self.case_list[idx])

        modalities = ['flair', 't1', 't1ce', 't2']
        images = []

        for modality in modalities:
            file_path = os.path.join(case_path, f"{self.case_list[idx]}_{modality}.nii.gz")
            img = nib.load(file_path).get_fdata()
            images.append(img)

        seg_path = os.path.join(case_path, f"{self.case_list[idx]}_seg.nii.gz")
        mask = nib.load(seg_path).get_fdata()

        image = np.stack(images, axis=0)
        image = self.normalize(image)
        mask = self.map_labels(mask)

        sample = {'image': image, 'mask': mask, 'case_name': self.case_list[idx]}

        if self.transform:
            sample = self.transform(sample)

        if self.train:
            sample = self.random_crop(sample)
        else:
            sample = self.center_crop(sample)

        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['mask'] = torch.from_numpy(sample['mask']).long()

        return sample

    def normalize(self, image):
        """Z-score归一化"""
        for i in range(image.shape[0]):
            modality = image[i]
            mask = modality > 0
            if mask.sum() > 0:
                mean = modality[mask].mean()
                std = modality[mask].std()
                image[i][mask] = (modality[mask] - mean) / (std + 1e-8)
        return image

    def map_labels(self, mask):
        """BraTS标签映射: 4->3"""
        mask[mask == 4] = 3
        return mask

    def random_crop(self, sample):
        image, mask = sample['image'], sample['mask']
        _, h, w, d = image.shape
        ph, pw, pd = self.patch_size

        h_start = random.randint(0, max(0, h - ph))
        w_start = random.randint(0, max(0, w - pw))
        d_start = random.randint(0, max(0, d - pd))

        image_crop = image[:, h_start:h_start + ph, w_start:w_start + pw, d_start:d_start + pd]
        mask_crop = mask[h_start:h_start + ph, w_start:w_start + pw, d_start:d_start + pd]

        sample['image'] = image_crop
        sample['mask'] = mask_crop
        return sample

    def center_crop(self, sample):
        image, mask = sample['image'], sample['mask']
        _, h, w, d = image.shape
        ph, pw, pd = self.patch_size

        h_start = (h - ph) // 2
        w_start = (w - pw) // 2
        d_start = (d - pd) // 2

        image_crop = image[:, h_start:h_start + ph, w_start:w_start + pw, d_start:d_start + pd]
        mask_crop = mask[h_start:h_start + ph, w_start:w_start + pw, d_start:d_start + pd]

        sample['image'] = image_crop
        sample['mask'] = mask_crop
        return sample


class RandomFlip:
    """随机翻转"""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() > 0.5:
            image = np.flip(image, axis=3).copy()
            mask = np.flip(mask, axis=2).copy()

        sample['image'] = image
        sample['mask'] = mask
        return sample


class RandomRotate:
    """随机旋转"""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        k = random.randint(0, 3)
        if k > 0:
            axes = random.choice([(1, 2), (1, 3), (2, 3)])
            image = np.rot90(image, k=k, axes=axes).copy()
            mask = np.rot90(mask, k=k, axes=(axes[0] - 1, axes[1] - 1)).copy()

        sample['image'] = image
        sample['mask'] = mask
        return sample


class RandomIntensity:
    """随机强度扰动"""

    def __init__(self, factor=0.1):
        self.factor = factor

    def __call__(self, sample):
        image = sample['image']

        if random.random() > 0.5:
            scale = 1 + random.uniform(-self.factor, self.factor)
            shift = random.uniform(-self.factor, self.factor)
            image = image * scale + shift

        sample['image'] = image
        return sample


class GaussianNoise:
    """添加高斯噪声"""

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, sample):
        image = sample['image']

        if random.random() > 0.5:
            noise = np.random.normal(0, self.std, image.shape)
            image = image + noise

        sample['image'] = image
        return sample


def get_train_transforms():
    from torchvision import transforms
    return transforms.Compose([
        RandomFlip(),
        RandomRotate(),
        RandomIntensity(factor=0.1),
        GaussianNoise(std=0.01),
    ])


def create_data_loaders(data_dir, batch_size=2, patch_size=(128, 128, 128), num_workers=0):
    train_dataset = BraTSDataset(
        data_dir=data_dir, transform=get_train_transforms(),
        patch_size=patch_size, train=True
    )
    val_dataset = BraTSDataset(
        data_dir=data_dir, transform=None,
        patch_size=patch_size, train=False
    )

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, _ = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    _, val_dataset = torch.utils.data.random_split(
        val_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")

    return train_loader, val_loader