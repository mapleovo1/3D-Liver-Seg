import matplotlib.pyplot as plt
import torch
import numpy as np
import json, os, torch, cv2, random, numpy as np, albumentations as A, nibabel as nib
from matplotlib import pyplot as plt
from glob import glob
from torch.utils.data import random_split, Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import transforms as tfs
from torch import nn
import segmentation_models_pytorch as smp, time
from tqdm import tqdm
from torch.nn import functional as F
from datetime import datetime

class CustomSegmentationDataset(Dataset):

    def __init__(self, root, transformations=None):

        im_nii_paths = sorted(glob(f"{root}/images/*.nii"))
        gt_nii_paths = sorted(glob(f"{root}/labels/*.nii"))

        self.ims, self.gts = self.get_slices(im_nii_paths, gt_nii_paths)
        self.transformations = transformations
        self.n_cls = 2

        assert len(self.ims) == len(self.gts)

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):

        im, gt = self.ims[idx], self.gts[idx]
        if self.transformations: 
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
            if index == 984: 
                break
            nii_im_data, nii_gt_data = self.read_nii(im_nii, gt_nii)

            for idx, (im, gt) in enumerate(zip(nii_im_data, nii_gt_data)):
                if len(np.unique(gt)) == 2:
                    ims.append(im)
                    gts.append(gt)

        return ims, gts

    def read_nii(self, im, gt):
        return nib.load(im).get_fdata().transpose(2, 1, 0), nib.load(gt).get_fdata().transpose(2, 1, 0)

    def apply_transformations(self, im, gt):
        transformed = self.transformations(image=im, mask=gt)
        return transformed["image"], transformed["mask"]


def get_dls(root, transformations, bs, split=[0.9, 0.05, 0.05], ns=4):
    assert sum(split) == 1., "Sum of the split must be exactly 1"

    ds = CustomSegmentationDataset(root=root, transformations=transformations)
    n_cls = ds.n_cls

    tr_len = int(len(ds) * split[0])
    val_len = int(len(ds) * split[1])
    test_len = len(ds) - (tr_len + val_len)

    tr_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [tr_len, val_len, test_len])

    tr_dl = DataLoader(dataset=tr_ds, batch_size=bs, drop_last=True, shuffle=True, num_workers=ns)
    val_dl = DataLoader(dataset=val_ds, batch_size=bs, shuffle=False, num_workers=ns)
    test_dl = DataLoader(dataset=test_ds, batch_size=1, shuffle=False, num_workers=ns)

    return tr_dl, val_dl, test_dl, n_cls

class SEBlock(nn.Module):
    """通道注意力模块(Squeeze-and-Excitation)"""
    def __init__(self, channel, reduction=16):
        """
        参数说明：
        - channel: 输入特征图的通道数
        - reduction: 通道压缩比例(默认16)
        """
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Squeeze操作
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # Excitation操作
        y = self.fc(y).view(b, c, 1, 1)
        # 特征重标定
        return x * y.expand_as(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        x = self.double_conv(x)
        x = self.se(x)  # 添加SE模块
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_se=use_se)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # If bilinear, use bilinear upsampling, else use transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, use_se=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_se = use_se

        # 编码器部分
        self.inc = DoubleConv(n_channels, 64, use_se=use_se)
        self.down1 = Down(64, 128, use_se=use_se)
        self.down2 = Down(128, 256, use_se=use_se)
        self.down3 = Down(256, 512, use_se=use_se)
        self.down4 = Down(512, 1024, use_se=use_se)

        # 解码器部分
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

def plot_image(ax, tensor, title, cmap=None, is_mask=False):
    if is_mask:
        tensor = tensor.squeeze().cpu().numpy().astype(np.uint8)
    else:
        tensor = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
        tensor = (tensor * 255).astype(np.uint8)

    ax.imshow(tensor, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')


def inference(dl, model, device, n_samples=4, save_dir="test_plot"):
    model.eval()

    # 如果目录不存在，则创建
    os.makedirs(save_dir, exist_ok=True)

    # 创建图像画布
    fig, axes = plt.subplots(n_samples, 3, figsize=(18, 24))

    count = 0
    with torch.no_grad():
        for idx, (im, gt) in enumerate(dl):
            if count >= n_samples:
                break

            pred = model(im.to(device))
            pred_mask = torch.argmax(pred, dim=1)

            im_np = im.squeeze(0).squeeze(0).cpu().numpy()
            im_np = (im_np * 255).astype(np.uint8)
            gt_np = gt.squeeze().squeeze(0).cpu().numpy().astype(np.uint8)
            pred_np = pred_mask.squeeze().cpu().numpy().astype(np.uint8)

            axes[count, 0].imshow(im_np, cmap='gray')
            axes[count, 0].set_title("Original Image", fontsize=12)
            axes[count, 0].axis('off')

            axes[count, 1].imshow(gt_np, cmap='gray')
            axes[count, 1].set_title("Ground Truth", fontsize=12)
            axes[count, 1].axis('off')

            axes[count, 2].imshow(pred_np, cmap='gray')
            axes[count, 2].set_title("Predicted Mask", fontsize=12)
            axes[count, 2].axis('off')

            count += 1

    plt.tight_layout(pad=2.0)
    save_path = os.path.join(save_dir, "inference_results.png")
    plt.savefig(save_path)
    print(f"✅ 推理图像已保存至: {save_path}")
    plt.close(fig)

def evaluate_metrics(dl, model, device, n_cls=2):
    model.eval()
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for ims, gts in tqdm(dl, desc="Evaluating"):
            ims = ims.to(device)
            gts = gts.to(device).long().squeeze(1)

            preds = model(ims)
            preds = torch.argmax(preds, dim=1)

            all_preds.append(preds.cpu().detach())
            all_gts.append(gts.cpu().detach())

    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)

    # 新增PA计算
    eps = 1e-10
    correct = (all_preds == all_gts).sum().float()
    total = all_gts.numel()
    pa = (correct + eps) / (total + eps)

    # 原有指标计算
    iou_per_class = []
    for c in range(n_cls):
        pred_c = (all_preds == c)
        gt_c = (all_gts == c)

        if gt_c.sum() == 0:
            iou_per_class.append(float('nan'))
            continue

        intersect = (pred_c & gt_c).sum().float()
        union = (pred_c | gt_c).sum().float()
        iou = (intersect + eps) / (union + eps)
        iou_per_class.append(iou.item())

    valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
    miou = np.mean(valid_ious) if valid_ious else 0.0

    pred_mask = (all_preds == 1)
    gt_mask = (all_gts == 1)
    sum_gt = gt_mask.sum().float()

    if sum_gt == 0:
        dice = 0.0
    else:
        intersect = (pred_mask & gt_mask).sum().float()
        sum_pred = pred_mask.sum().float()
        dice = (2. * intersect + eps) / (sum_pred + sum_gt + eps)

    return miou, dice.item(), pa.item()

if __name__ == '__main__':
    # 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "saved_models/best_model.pt"  

    # 正确加载模型
    model = UNet(n_channels=1, n_classes=2)  
    try:
        model.load_state_dict(
            torch.load(model_path, map_location=device))
        model.to(device)
        print(f"✅ 成功加载模型到 {device}")
    except Exception as e:
        print(f"❌ 加载失败: {str(e)}")
        print("可能原因：")
        print("1. 模型文件路径错误")
        print("2. 模型结构与保存参数不匹配")
        exit(1)

    # 数据预处理（必须与训练一致）
    root = "3Dliverdata"  
    trans = A.Compose([
        A.Resize(256, 256, interpolation=cv2.INTER_NEAREST),
        ToTensorV2(transpose_mask=True)
    ])

    # 获取数据加载器
    _, _, test_dl, _ = get_dls(root, trans, bs=1, ns=0)

    # 运行推理
    print(f"\n🚀 开始推理（设备: {device}）...")
    inference(test_dl, model, device, n_samples=4)

    # 计算指标
    print("\n📈 正在计算评估指标...")
    test_miou, test_dice, test_pa = evaluate_metrics(test_dl, model, device)

    print("\n📊 最终测试结果:")
    print(f"平均交并比 (mIoU): {test_miou:.4f}")
    print(f"Dice系数: {test_dice:.4f}")
    print(f"像素准确率 (PA): {test_pa:.4f}")