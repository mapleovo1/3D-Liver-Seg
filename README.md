# 基于深度学习模型的3D肝脏和肝脏肿瘤分割研究

![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1%2B-orange)

本项目实现了基于改进 SE-UNet 的肝脏及肝脏肿瘤图像分割方法，适用于 3D 医学图像，目标是提升分割精度和泛化能力，特别应用于肝脏肿瘤的临床辅助诊断任务中。

## 安装环境

推荐使用 **Python 3.8+**，安装依赖：


# 安装相关依赖
pip install -r requirements.txt


## 📁 数据集
本项目基于 Kaggle 提供的 [3D Liver and Liver Tumor Segmentation 数据集](https://www.kaggle.com/datasets/gauravduttakiit/3d-liver-and-liver-tumor-segmentation/data) 进行训练与评估。
> 说明：该数据集包含 CT 扫描图像及其标注，用于医学图像中的肝脏与肿瘤三维分割。下载数据需登录 Kaggle 账户。
> 下载好之后放入 3Dliverdata/ 文件夹。使用 predata/ 中的数据预处理脚本进行标准化、格式转换等预处理。

## 训练模型
python main.py


## 模型测试
python test.py 




