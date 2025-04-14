![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1%2B-orange)
![License](https://img.shields.io/badge/License-MIT-blue)

# 基于深度学习模型的3D肝脏和肝脏肿瘤分割研究

本项目实现了基于改进 SE-UNet 的肝脏及肝脏肿瘤图像分割方法，适用于 3D 医学图像，目标是提升分割精度和泛化能力，特别应用于肝脏肿瘤的临床辅助诊断任务中。

## ✨ 特点

- 基于 SE（Squeeze-and-Excitation）机制优化 U-Net 架构
- 支持数据增强（如弹性变换、旋转、裁剪等）
- 支持 Dice Loss + Cross Entropy 等组合损失
- 模型训练、评估、保存与日志完整流程
- 支持 3D 医学图像数据


---

##  安装环境

推荐使用 **Python 3.8+**，安装依赖：

```bash
# 安装相关依赖
pip install -r requirements.txt

主要依赖：

torch

torchvision

numpy

scipy

nibabel

matplotlib

scikit-learn

tqdm


## 📁 数据集
本项目基于 Kaggle 提供的 [3D Liver and Liver Tumor Segmentation 数据集](https://www.kaggle.com/datasets/gauravduttakiit/3d-liver-and-liver-tumor-segmentation/data) 进行训练与评估。
> 说明：该数据集包含 CT 扫描图像及其标注，用于医学图像中的肝脏与肿瘤三维分割。下载数据需登录 Kaggle 账户。
> 下载好之后放入 3Dliverdata/ 文件夹。使用 predata/ 中的数据预处理脚本进行标准化、格式转换等预处理。

## 训练模型
```bash
python main.py


## 模型测试
```bash
python test.py 




