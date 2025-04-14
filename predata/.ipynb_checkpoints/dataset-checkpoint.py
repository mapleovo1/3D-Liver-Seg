import os
import torch
import nibabel as nib
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

class LiverDataset(Dataset):
    def __init__(self, root, transform=None):

        im_nii_paths = sorted(glob(f"{root}/images/*.nii"))
        gt_nii_paths = sorted(glob(f"{root}/labels/*.nii"))

        self.ims, self.gts = self.get_slices(im_nii_paths, gt_nii_paths)
        self.transform = transform
        self.n_cls = 2

        assert len(self.ims) == len(self.gts)

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):

        im, gt = self.ims[idx], self.gts[idx]
        if self.transform: 
            im, gt = self.apply_transformations(im, gt)
        im = self.preprocess_im(im)
        gt[gt > 1] = 1

        return im.float(), gt.unsqueeze(0).long()

    def preprocess_im(self, im):

        max_val = torch.max(im)
        im[im < 0] = 0

        return im / max_val

    def get_slices(self, im_nii_paths, gt_nii_paths):

        ims, gts = [], []

        for index, (im_nii, gt_nii) in enumerate(zip(im_nii_paths, gt_nii_paths)):
            if index == 984: break
            # print(f"nifti file number {index + 1} is being converted...")
            nii_im_data, nii_gt_data = self.read_nii(im_nii, gt_nii)

            for idx, (im, gt) in enumerate(zip(nii_im_data, nii_gt_data)):
                if len(np.unique(gt)) == 2:
                    ims.append(im)
                    gts.append(gt)

        return ims, gts

    def read_nii(self, im, gt):
        return nib.load(im).get_fdata().transpose(2, 1, 0), nib.load(gt).get_fdata().transpose(2, 1, 0)

    def apply_transformations(self, im, gt):
        transformed = self.transform(image=im, mask=gt)
        return transformed["image"], transformed["mask"]

def get_dataloaders(root_dir, batch_size=64, split_ratio=(0.9, 0.05, 0.05)):
    transform = A.Compose([
        A.Resize(256, 256, interpolation=cv2.INTER_NEAREST),
        ToTensorV2(transpose_mask=True)
    ])
    
    full_dataset = LiverDataset(root_dir, transform=transform)
    train_size = int(split_ratio[0] * len(full_dataset))
    val_size = int(split_ratio[1] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    return (
        DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, drop_last=True),
        DataLoader(val_ds, batch_size, shuffle=False, num_workers=4),
        DataLoader(test_ds, 1, shuffle=False, num_workers=4)
    ), full_dataset.n_cls