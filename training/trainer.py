import os
import time
import torch
import numpy as np
from tqdm import tqdm


class LiverTrainer:
    def __init__(self, model, dataloaders, criterion, optimizer, device, logger, n_classes=2):
        self.model = model.to(device)
        self.train_loader, self.val_loader, _ = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.n_classes = n_classes
        self.best_loss = np.inf
        self.early_stop_threshold = 5
        self.not_improve = 0

    def _calculate_metrics(self, pred, gt):
        
        eps = 1e-10
        n_cls = self.n_classes
        device = self.device

        pred_ = pred  
        pred = torch.argmax(pred, dim=1) > 0  
        gt = gt.squeeze(1)  # 去除通道维度 (batch, H, W)

        match = torch.eq(pred, gt).int()
        pa = float(match.sum()) / float(match.numel())

        def to_contiguous(inp):
            return inp.contiguous().view(-1)

        pred_flat = to_contiguous(pred)
        gt_flat = to_contiguous(gt)

        iou_per_class = []
        for c in range(n_cls):
            match_pred = (pred_flat == c)
            match_gt = (gt_flat == c)

            if match_gt.long().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(match_pred, match_gt).sum().float().item()
                union = torch.logical_or(match_pred, match_gt).sum().float().item()
                iou = (intersect + eps) / (union + eps)
                iou_per_class.append(iou)

        miou = np.nanmean(iou_per_class)

        pred_mask = (pred == 1)
        gt_mask = (gt == 1)
        intersect = (pred_mask & gt_mask).sum().float()
        dice = (2. * intersect + eps) / (pred_mask.sum() + gt_mask.sum() + eps)

        return pa, miou, dice.item()

    def train_epoch(self):
        self.model.train()
        total_loss, total_pa, total_iou, total_dice = 0, 0, 0, 0
        for images, masks in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks.squeeze(1))
            
            loss.backward()
            self.optimizer.step()
            
            pa, miou, dice = self._calculate_metrics(outputs, masks)
            total_loss += loss.item()
            total_pa += pa
            total_iou += miou
            total_dice += dice
        
        return (total_loss/len(self.train_loader), 
                total_pa/len(self.train_loader),
                total_iou/len(self.train_loader),
                total_dice/len(self.train_loader))

    def validate(self):
        self.model.eval()
        total_loss, total_pa, total_iou, total_dice = 0, 0, 0, 0
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks.squeeze(1))
                
                pa, miou, dice = self._calculate_metrics(outputs, masks)
                total_loss += loss.item()
                total_pa += pa
                total_iou += miou
                total_dice += dice
        
        return (total_loss/len(self.val_loader), 
                total_pa/len(self.val_loader),
                total_iou/len(self.val_loader),
                total_dice/len(self.val_loader))

    def run(self, epochs, save_path="saved_models"):
        os.makedirs(save_path, exist_ok=True)
        for epoch in range(1, epochs+1):
            start_time = time.time()
            
            # 训练阶段
            train_loss, train_pa, train_iou, train_dice = self.train_epoch()
            
            # 验证阶段
            val_loss, val_pa, val_iou, val_dice = self.validate()
            
            # 记录日志
            self.logger.log(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} PA: {train_pa:.4f} IoU: {train_iou:.4f} Dice: {train_dice:.4f} | "
                f"Val Loss: {val_loss:.4f} PA: {val_pa:.4f} IoU: {val_iou:.4f} Dice: {val_dice:.4f} | "
                f"Time: {time.time()-start_time:.1f}s"
            )
            
            # 模型保存与早停
            if val_loss < (self.best_loss - 0.005):
                self.logger.log(f"✅ 损失从 {self.best_loss:.4f} 下降到 {val_loss:.4f}")
                self.best_loss = val_loss
                self.not_improve = 0
                torch.save(self.model.state_dict(), f"{save_path}/best_model.pt")
            else:
                self.not_improve += 1
                if self.not_improve >= self.early_stop_threshold:
                    self.logger.log(f"⏹️ 提前停止训练，连续 {self.early_stop_threshold} 个epoch未提升")
                    break