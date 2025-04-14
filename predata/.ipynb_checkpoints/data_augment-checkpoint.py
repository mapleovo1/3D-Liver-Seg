import glob
import os
import random
import nibabel as nib
import numpy as np
import torch
import scipy.ndimage
from torch.utils.data import Dataset
from scipy.ndimage import map_coordinates, affine_transform
from skimage.measure import label, regionprops


# 配置参数
class AugConfig:
    augmentation_factor = 8  # 增强倍数
    rotation_range = (-15, 15)  # 旋转角度范围(度)
    scale_range = (0.8, 1.2)  # 缩放范围
    shift_range = 0.15  # 平移范围(占图像尺寸比例)
    flip_prob = 0.5  # 各向翻转概率
    elastic_alpha = (150, 300)  # 弹性形变强度范围
    elastic_sigma = (10, 20)  # 弹性形变平滑系数
    noise_level = 0.05  # 噪声强度
    gamma_range = (0.7, 1.5)  # 对比度调整范围


def load_all_data(root_dir):
    img_paths = sorted(glob.glob(os.path.join(root_dir, 'images', '*.nii')))
    label_paths = sorted(glob.glob(os.path.join(root_dir, 'labels', '*.nii')))

    assert len(img_paths) == len(label_paths), "图像和标签数量不匹配！"
    print(f"成功加载 {len(img_paths)} 个样本")
    return img_paths, label_paths


class AdvancedAugmentor:
    @staticmethod
    def random_affine(image, mask):
        """三维仿射变换（旋转+缩放+平移）"""
        # 生成随机参数
        angles = [np.random.uniform(*AugConfig.rotation_range) for _ in range(3)]
        scale = np.random.uniform(*AugConfig.scale_range, size=3)
        shift = [np.random.uniform(-s * AugConfig.shift_range, s * AugConfig.shift_range)
                 for s in image.shape]

        # 构建变换矩阵
        matrix = np.eye(4)
        for i, angle in enumerate(angles):
            c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
            if i == 0:  # X轴旋转
                rot = np.array([[1, 0, 0, 0],
                                [0, c, -s, 0],
                                [0, s, c, 0],
                                [0, 0, 0, 1]])
            elif i == 1:  # Y轴旋转
                rot = np.array([[c, 0, s, 0],
                                [0, 1, 0, 0],
                                [-s, 0, c, 0],
                                [0, 0, 0, 1]])
            else:  # Z轴旋转
                rot = np.array([[c, -s, 0, 0],
                                [s, c, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
            matrix = matrix @ rot

        # 添加缩放和平移
        matrix = matrix @ np.diag([*scale, 1])
        matrix[:3, 3] = shift

        # 应用变换
        img_trans = affine_transform(image, matrix, order=1)
        mask_trans = affine_transform(mask, matrix, order=0)
        return img_trans, mask_trans

    @staticmethod
    def random_flip(image, mask):
        """三维随机翻转"""
        for axis in [0, 1, 2]:
            if random.random() < AugConfig.flip_prob:
                image = np.flip(image, axis=axis)
                mask = np.flip(mask, axis=axis)
        return image, mask

    @staticmethod
    def elastic_deform(image, mask):
        """弹性形变增强"""
        alpha = random.uniform(*AugConfig.elastic_alpha)
        sigma = random.uniform(*AugConfig.elastic_sigma)
        shape = image.shape

        # 生成随机位移场
        dx = scipy.ndimage.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = scipy.ndimage.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = scipy.ndimage.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        # 构建坐标网格
        z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = (z + dz).reshape(-1, 1), (y + dy).reshape(-1, 1), (x + dx).reshape(-1, 1)

        # 应用变形
        deformed_img = map_coordinates(image, indices, order=1).reshape(shape)
        deformed_mask = map_coordinates(mask, indices, order=0).reshape(shape)
        return deformed_img, deformed_mask

    @staticmethod
    def adjust_contrast(image):
        """对比度调整"""
        gamma = np.random.uniform(*AugConfig.gamma_range)
        min_val = image.min()
        max_val = image.max()
        return np.power((image - min_val) / (max_val - min_val), gamma) * (max_val - min_val) + min_val

    @staticmethod
    def add_noise(image):
        """添加医学噪声"""
        noise = np.random.normal(0, AugConfig.noise_level * np.std(image), image.shape)
        return np.clip(image + noise, image.min(), image.max())


class EnhancedDataset(Dataset):
    def __init__(self, img_paths, label_paths):
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.global_mean, self.global_std = self._compute_stats()

    def __len__(self):
        return len(self.img_paths) * AugConfig.augmentation_factor

    def __getitem__(self, idx):
        orig_idx = idx // AugConfig.augmentation_factor
        img, mask = self._load_original(orig_idx)

        # 应用基础增强
        img, mask = AdvancedAugmentor.random_flip(img, mask)
        # 随机选择增强组合
        augmentations = [
            self._apply_elastic,
            self._apply_affine,
            self._adjust_contrast,
            self._add_noise
        ]
        random.shuffle(augmentations)

        # 应用两种随机增强
        for aug in augmentations[:2]:
            img, mask = aug(img, mask)

        return self._to_tensor(img, mask)

    def _load_original(self, idx):
        """加载并标准化数据"""
        img = nib.load(self.img_paths[idx]).get_fdata()
        mask = nib.load(self.label_paths[idx]).get_fdata()
        return (img - self.global_mean) / self.global_std, (mask > 0).astype(np.float32)

    def _compute_stats(self):
        """计算全局统计量"""
        print("计算全局统计量...")
        pixels = []
        for path in self.img_paths:
            img = nib.load(path).get_fdata()
            pixels.append(img.flatten())
        all_pixels = np.concatenate(pixels)
        mean_val, std_val = np.mean(all_pixels), np.std(all_pixels)

        # **打印全局均值和标准差**
        print(f"全局均值: {mean_val:.6f}, 全局标准差: {std_val:.6f}")
        return np.mean(all_pixels), np.std(all_pixels)

    def _apply_elastic(self, img, mask):
        return AdvancedAugmentor.elastic_deform(img, mask)

    def _apply_affine(self, img, mask):
        return AdvancedAugmentor.random_affine(img, mask)

    def _adjust_contrast(self, img, mask):
        return AdvancedAugmentor.adjust_contrast(img), mask

    def _add_noise(self, img, mask):
        return AdvancedAugmentor.add_noise(img), mask

    def _to_tensor(self, img, mask):
        return (
            torch.FloatTensor(img.copy()).unsqueeze(0),
            torch.FloatTensor(mask.copy()).unsqueeze(0)
        )

def save_all_augmentations(dataset, save_dir):
    """保存所有增强数据"""
    img_dir = os.path.join(save_dir, 'images')
    label_dir = os.path.join(save_dir, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    print(f"开始保存增强数据到 {save_dir}...")
    for i in range(len(dataset)):
        img, mask = dataset[i]
        orig_idx = i // AugConfig.augmentation_factor
        base_name = os.path.basename(dataset.img_paths[orig_idx]).split('.')[0]

        # 生成唯一文件名
        filename = f"{base_name}_aug{i % AugConfig.augmentation_factor:03d}.nii"

        # 保存为NIfTI
        nib.save(nib.Nifti1Image(img.numpy().squeeze(), np.eye(4)),
                 os.path.join(img_dir, filename))
        nib.save(nib.Nifti1Image(mask.numpy().squeeze(), np.eye(4)),
                 os.path.join(label_dir, filename))

        if (i + 1) % 100 == 0:
            print(f"已保存 {i + 1}/{len(dataset)} 个样本")
    print("增强数据保存完成！")


# 使用示例
if __name__ == "__main__":
    data_root = "./3Dliverdata"
    save_dir = "./augmented_data"
    os.makedirs(data_root, exist_ok=True)
    img_paths, label_paths = load_all_data(data_root)

    dataset = EnhancedDataset(img_paths, label_paths)
    save_all_augmentations(dataset, save_dir)

    print(f"\n原始数据量: {len(img_paths)}")
    print(f"增强后总数据量: {len(dataset)}")
    print(f"增强倍数: {AugConfig.augmentation_factor}x")