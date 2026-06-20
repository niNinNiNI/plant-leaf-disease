# 🌿 农业植物叶片病害识别

> 深度学习与计算机视觉课程大作业 — 基于 PyTorch 的植物叶片病害智能识别系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 项目简介

本项目基于深度卷积神经网络（CNN）对农作物叶片进行病害自动识别，支持 **15 种** 植物病害分类。采用迁移学习策略，利用在 ImageNet 上预训练的 ResNet、EfficientNet、DenseNet、MobileNet、ConvNeXt 等骨干网络，结合 MixUp 数据增强、标签平滑、Focal Loss、测试时增强（TTA）及模型集成等技术，实现高精度、鲁棒的病害识别。

## 🗂 项目结构

```
.
├── main.py                   # 主训练/评估/可视化入口
├── src/
│   ├── config.py             # 全局配置参数
│   ├── dataset.py            # 数据集加载、划分、加权采样
│   ├── models.py             # BasicCNN / ImprovedCNN / TransferLearningModel
│   ├── trainer.py            # 训练器（混合精度、早停、梯度裁剪）
│   ├── losses.py             # CrossEntropy / Focal Loss / Label Smoothing
│   ├── evaluate.py           # 测试评估、错误分析、TTA
│   ├── ensemble.py           # 多模型集成推理
│   ├── transforms.py         # 图像增强管道
│   ├── gradcam.py            # Grad-CAM 可解释性可视化
│   ├── ablation.py           # 消融实验框架
│   └── utils.py              # 随机种子、设备管理、参数统计、绘图
├── results/
│   ├── build_docx.cjs        # 生成 Word 实验报告
│   └── generate_docx.cjs     # 文档生成脚本
├── kaggle/
│   ├── kaggle_notebook.py    # Kaggle Notebook 版本（核心）
│   ├── kaggle_notebook_complete.py  # Kaggle 完整版
│   └── build_ipynb.py        # 构建 .ipynb 工具
├── checkpoints/              # 模型权重保存目录
└── results/figures/          # 训练曲线、混淆矩阵等输出图表
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA（可选，推荐用于 GPU 加速）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib seaborn scikit-learn pandas tqdm pillow opencv-python
```

### 数据集

使用 [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) 数据集，包含 15 种植物叶片病害类别。数据集应放置在：

```
../archive/plantvillage/PlantVillage/
```

每个类别为一个子文件夹，文件夹名为类别名。

### 命令行用法

```bash
# 标准训练（迁移学习 ResNet-50，推荐）
python main.py --model transfer --backbone resnet50 --epochs 80

# 使用其他骨干网络
python main.py --model transfer --backbone efficientnet_b0 --epochs 80
python main.py --model transfer --backbone convnext_tiny --epochs 80

# 启用 MixUp 数据增强
python main.py --model transfer --backbone resnet50 --mixup --mixup_alpha 0.2

# 使用 Focal Loss 处理类别不平衡
python main.py --model transfer --backbone resnet50 --loss focal

# 两阶段训练：先训练分类头，再微调骨干
python main.py --model transfer --backbone resnet50 --finetune

# 从零训练基础 CNN
python main.py --model basic --epochs 50 --dropout 0.5

# 仅评估
python main.py --eval_only --checkpoint checkpoints/transfer_resnet50/best_model.pth

# 启用 TTA 评估
python main.py --eval_only --tta --tta_mode full --checkpoint checkpoints/transfer_resnet50/best_model.pth

# 生成 Grad-CAM 可视化
python main.py --gradcam --checkpoint checkpoints/transfer_resnet50/best_model.pth

# 运行消融实验
python main.py --ablation

# 断点续训
python main.py --model transfer --backbone resnet50 --resume --checkpoint checkpoints/.../best_model.pth

# 指定数据集路径
python main.py --data_root /path/to/PlantVillage
```

## 🧠 模型架构

### 1. BasicCNN
从零搭建的 5 层卷积网络，包含 BatchNorm、Dropout、Global Average Pooling：
```
ConvBlock(3→32) → ConvBlock(32→64) → ConvBlock(64→128)
→ ConvBlock(128→256) → ConvBlock(256→512) → GAP → FC(512→256) → FC(256→15)
```

### 2. ImprovedCNN
引入残差连接的改进网络，含 Stem + 4 个残差阶段：
```
Stem(3→64) → ResLayer1(64) → ResLayer2(128) → ResLayer3(256) → ResLayer4(512) → GAP → FC(512→15)
```

### 3. TransferLearningModel (⭐ 推荐)
基于 ImageNet 预训练模型的迁移学习，支持 8 种骨干网络：

| 骨干网络 | 参数量 | 特点 |
|---------|--------|------|
| `resnet50` | 25.6M | 经典残差网络，平衡精度与效率 |
| `resnet101` | 44.5M | 更深残差网络，更高精度 |
| `efficientnet_b0` | 5.3M | 轻量高效，适合部署 |
| `efficientnet_b3` | 12.2M | 更大 EfficientNet 变体 |
| `densenet121` | 8.0M | 密集连接，特征复用 |
| `densenet169` | 14.1M | 更深 DenseNet |
| `mobilenet_v3` | 5.5M | 移动端优化 |
| `convnext_tiny` | 28.6M | 现代 CNN 设计 |

## ⚙️ 关键技术

### 训练策略
- **两阶段训练**: 阶段一冻结骨干训练分类头 → 阶段二解冻微调
- **智能解冻**: 按 Stage/Block 粒度选择性解冻骨干网络
- **学习率调度**: CosineAnnealing / ReduceLROnPlateau / OneCycle
- **混合精度训练 (AMP)**: 加速训练，减少显存占用
- **早停 (Early Stopping)**: 防止过拟合（默认 patience=20）
- **梯度裁剪**: 稳定训练（max_norm=1.0）

### 损失函数
- **Cross Entropy Loss**: 标准分类损失
- **Focal Loss**: 缓解类别不平衡，聚焦难样本
- **Label Smoothing**: 防止过拟合，提高泛化能力

### 数据增强
- 随机旋转、翻转、缩放、裁剪
- 颜色抖动（亮度、对比度、饱和度）
- **MixUp**: 样本混合增强（α=0.2）

### 评估优化
- **测试时增强 (TTA)**: Simple（2 视图）/ Full（10 视图），提升 1-3% 准确率
- **模型集成**: 平均多模型预测概率，利用模型互补性提升效果

### 可解释性
- **Grad-CAM**: 热力图可视化，展示模型关注的叶片区域

### 分析工具
- 混淆矩阵可视化
- 各类别精确率/召回率/F1 报告
- 错误样本分析（Top-K 混淆对）
- 模型间预测一致性分析

## 📊 消融实验

消融实验对比以下组件对模型性能的影响：

| 实验 | 骨干 | MixUp | 损失 | TTA | 集成 |
|------|------|-------|------|-----|------|
| 基线 | ResNet-50 | ✗ | CE | ✗ | ✗ |
| +数据增强 | ResNet-50 | ✓ | CE | ✗ | ✗ |
| +FocalLoss | ResNet-50 | ✗ | Focal | ✗ | ✗ |
| +LabelSmoothing | ResNet-50 | ✗ | LS | ✗ | ✗ |
| +TTA | ResNet-50 | ✗ | CE | ✓ | ✗ |
| +集成 | 多模型 | — | — | ✗ | ✓ |

```bash
python main.py --ablation
```

## 🔧 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 80 (transfer) / 50 | 训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--lr` | 0.001 | 初始学习率 |
| `--dropout` | 0.5 | Dropout 比例 |
| `--image_size` | 224 | 输入图像尺寸 |
| `--loss` | cross_entropy | 损失函数 (CE / focal / label_smoothing) |
| `--scheduler` | cosine | 学习率调度器 |
| `--mixup_alpha` | 0.2 | MixUp Beta 分布参数 |
| `--seed` | 42 | 随机种子 |

完整参数列表见 `python main.py --help`。

## 📁 输出文件

```
checkpoints/
├── transfer_resnet50/
│   ├── best_model.pth        # 验证集最佳模型
│   └── checkpoint_epoch_*.pth # 周期检查点

results/
├── logs/                     # TensorBoard 训练日志
├── figures/                  # 训练曲线、混淆矩阵、Grad-CAM 热力图
│   ├── training_curves_*.png
│   ├── confusion_matrix.png
│   └── gradcam_*.png
└── metrics.json              # 评估指标 JSON
```

## 📄 生成报告

```bash
# 生成 Word 格式实验报告
node results/generate_docx.cjs
```

## 🙏 致谢

- PlantVillage 数据集：[spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- 预训练模型：[torchvision.models](https://pytorch.org/vision/stable/models.html)
- Grad-CAM：[jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

---

**课程**: 深度学习与计算机视觉  
**语言**: Python / PyTorch  
**类别数**: 15  
**输入尺寸**: 224×224
