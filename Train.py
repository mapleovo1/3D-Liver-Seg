import numpy as np
import torch
import cv2
import torch.nn as nn
from tqdm import tqdm
import segmentation_models_pytorch as smp
from utils.logger import TrainingLogger
from predata.dataset import get_dataloaders
from model.se_unet import SE_UNet
from training.trainer import LiverTrainer

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.dice = smp.losses.DiceLoss(mode='multiclass', classes=[1])
    
    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss

def evaluate_model(model, test_loader, device, n_classes=2):
    model.eval()
    total_pa, total_iou, total_dice = 0, 0, 0
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # 计算PA
            pa = (preds == masks.squeeze(1)).float().mean().item()
            
            # 计算mIoU
            iou_per_class = []
            for c in range(n_classes):
                pred_c = preds == c
                gt_c = masks.squeeze(1) == c
                if gt_c.sum() == 0:
                    continue
                intersect = (pred_c & gt_c).sum().float()
                union = (pred_c | gt_c).sum().float()
                iou = (intersect + 1e-10) / (union + 1e-10)
                iou_per_class.append(iou.item())
            miou = np.nanmean(iou_per_class) if iou_per_class else 0.0
            
            # 计算Dice
            dice = 2 * ((preds == 1) & (masks.squeeze(1) == 1)).sum() / (
                (preds == 1).sum() + (masks.squeeze(1) == 1).sum() + 1e-10
            ).item()
            
            total_pa += pa
            total_iou += miou
            total_dice += dice
    
    return {
        "pa": total_pa / len(test_loader),
        "iou": total_iou / len(test_loader),
        "dice": total_dice / len(test_loader)
    }

if __name__ == "__main__":
    # 初始化组件
    logger = TrainingLogger()
    dataloaders, n_classes = get_dataloaders("augmented_data")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 模型配置
    model = SE_UNet(n_channels=1, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = CombinedLoss(alpha=0.5)
    
    # 训练流程
    trainer = LiverTrainer(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        logger=logger,
        n_classes=n_classes
    )
    trainer.run(epochs=50)
    
    # 测试评估
    model.load_state_dict(torch.load("saved_models/best_model.pt"))
    test_metrics = evaluate_model(model, dataloaders[2], device)
    logger.log(
        f"测试结果 | PA: {test_metrics['pa']:.4f} | "
        f"mIoU: {test_metrics['iou']:.4f} | "
        f"Dice: {test_metrics['dice']:.4f}"
    )