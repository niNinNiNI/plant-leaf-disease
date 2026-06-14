# 农业植物叶片病害识别 — 训练操作指南

## 目录

1. [项目概述](#1-项目概述)
2. [环境配置](#2-环境配置)
3. [数据集准备](#3-数据集准备)
4. [项目结构](#4-项目结构)
5. [训练流程](#5-训练流程)
6. [模型评估](#6-模型评估)
7. [Grad-CAM 可视化](#7-grad-cam-可视化)
8. [消融实验](#8-消融实验)
9. [二阶段微调](#9-二阶段微调)
10. [常用命令速查](#10-常用命令速查)
11. [结果文件说明](#11-结果文件说明)

---

## 1. 项目概述

本项目基于 PlantVillage 数据集，使用卷积神经网络（CNN）对农作物叶片图像进行 **15 分类** 任务——判断叶片属于健康状态还是感染了某种特定病害。

### 支持的三类模型

| 模型类型 | `--model` 参数 | 说明 |
|---------|---------------|------|
| 基础 CNN | `basic` | 5层卷积 + 全局平均池化，从零训练 |
| 改进 CNN | `improved` | 引入残差连接，缓解梯度消失 |
| **迁移学习** | `transfer` | 使用 ImageNet 预训练权重，**推荐** |

### 支持的骨干网络（迁移学习）

`resnet50` / `resnet101` / `efficientnet_b0` / `efficientnet_b3` / `densenet121` / `densenet169` / `mobilenet_v3` / `convnext_tiny`

---

## 2. 环境配置

### 2.1 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv plant_disease_env
source plant_disease_env/bin/activate

# 安装 PyTorch（根据 CUDA 版本选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

### 2.2 依赖清单

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| torch | ≥ 2.0 | 深度学习框架 |
| torchvision | ≥ 0.15 | 预训练模型 + 图像变换 |
| matplotlib | ≥ 3.5 | 绘图可视化 |
| seaborn | ≥ 0.12 | 混淆矩阵热力图 |
| scikit-learn | ≥ 1.0 | 数据集划分 + 评估指标 |
| pillow | ≥ 9.0 | 图像加载 |
| numpy | ≥ 1.21 | 数值计算 |
| opencv-python | ≥ 4.5 | Grad-CAM 热力图叠加 |
| tqdm | ≥ 4.60 | 进度条 |
| tensorboard | ≥ 2.10 | 训练日志可视化 |

### 2.3 验证环境

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 3. 数据集准备

### 3.1 数据集说明

- **数据集名称**：PlantVillage
- **图像总数**：21,639 张 RGB 图像
- **类别数量**：15 个类别（涵盖 Pepper、Potato、Tomato 三种作物）
- **下载地址**：https://www.kaggle.com/datasets/emmarex/plantdisease

### 3.2 数据存放路径

默认情况下，程序会在以下两个位置查找数据集：

1. `项目父目录/archive/plantvillage/PlantVillage/`
2. `项目目录/archive/plantvillage/PlantVillage/`

如果你的数据集在其他位置，使用 `--data_root` 参数指定：

```bash
python main.py --data_root /path/to/your/PlantVillage
```

### 3.3 数据集目录结构

```
archive/plantvillage/PlantVillage/
├── Pepper__Bacterial_spot/       (997 张)
├── Pepper__Healthy/               (1478 张)
├── Potato__Early_blight/         (1000 张)
├── Potato__Late_blight/          (1000 张)
├── Potato__Healthy/              (152 张)
├── Tomato__Bacterial_spot/       (2127 张)
├── Tomato__Early_blight/         (1000 张)
├── Tomato__Late_blight/          (1909 张)
├── Tomato__Leaf_Mold/            (952 张)
├── Tomato__Septoria_leaf_spot/   (1771 张)
├── Tomato__Spider_mites/         (1676 张)
├── Tomato__Target_Spot/          (1404 张)
├── Tomato__Yellow_Leaf_Curl_Virus/ (3209 张)
├── Tomato__mosaic_virus/         (373 张)
└── Tomato__Healthy/              (1591 张)
```

### 3.4 数据划分

程序自动按 **7:1.5:1.5** 比例分层划分为训练集、验证集、测试集，确保各类别在各子集中比例一致。

---

## 4. 项目结构

```
个人作业/
├── main.py                 # 主入口，命令行接口
├── requirements.txt        # 依赖清单
├── src/
│   ├── __init__.py
│   ├── config.py           # 全局配置参数
│   ├── models.py           # 模型定义（BasicCNN / ImprovedCNN / TransferLearningModel）
│   ├── dataset.py          # 数据集类 + 数据划分 + DataLoader
│   ├── transforms.py       # 数据增强策略
│   ├── trainer.py          # 训练器（训练循环 + 早停 + AMP + TensorBoard）
│   ├── losses.py           # 损失函数（CE / Focal Loss / Label Smoothing）
│   ├── evaluate.py         # 测试集评估 + 错误样本分析
│   ├── gradcam.py          # Grad-CAM 可解释性可视化
│   ├── ablation.py         # 消融实验框架
│   └── utils.py            # 工具函数（随机种子、设备检测、绘图等）
├── checkpoints/            # 模型保存目录
│   └── transfer_resnet50/  # 按 模型_骨干网络 命名
│       ├── best_model.pth
│       └── checkpoint_epoch_N.pth
├── results/                # 结果输出目录
│   ├── figures/            # 图表
│   ├── logs/               # TensorBoard 日志
│   └── metrics.json        # 评估指标
└── archive/                # 数据集
    └── plantvillage/PlantVillage/
```

---

## 5. 训练流程

### 5.1 标准训练（迁移学习 ResNet-50）

这是**推荐的标准训练方式**，使用预训练 ResNet-50 作为骨干网络，冻结骨干只训练分类头：

```bash
python main.py --model transfer --backbone resnet50 --epochs 80
```

训练过程：

1. 自动加载数据集并划分训练/验证/测试集
2. 计算类别权重（处理类别不平衡）
3. 创建加权采样 DataLoader
4. 加载 ImageNet 预训练 ResNet-50
5. 冻结骨干网络 → 只训练自定义分类头
6. 使用 AdamW 优化器 + Cosine 学习率调度
7. 混合精度训练（AMP）+ 梯度裁剪
8. 每个 epoch 后在验证集上评估
9. 早停机制：20 个 epoch 无提升则停止
10. 自动保存最佳模型到 `checkpoints/transfer_resnet50/best_model.pth`

### 5.2 自定义训练参数

```bash
python main.py \
    --model transfer \
    --backbone resnet50 \
    --epochs 80 \
    --batch_size 32 \
    --lr 0.001 \
    --dropout 0.5 \
    --loss cross_entropy \
    --scheduler cosine \
    --seed 42
```

#### 超参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `transfer` | 模型类型：`basic` / `improved` / `transfer` |
| `--backbone` | `resnet50` | 迁移学习骨干网络 |
| `--epochs` | 80 (transfer) / 50 (其他) | 训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--lr` | 0.001 | 初始学习率 |
| `--dropout` | 0.5 | Dropout 比例（防止背景过拟合） |
| `--loss` | `cross_entropy` | 损失函数：`cross_entropy` / `focal` / `label_smoothing` |
| `--scheduler` | `cosine` | 学习率调度：`cosine` / `plateau` / `step` / `onecycle` |
| `--seed` | 42 | 随机种子（确保可复现） |

### 5.3 训练其他模型

```bash
# 从零训练基础 CNN
python main.py --model basic --epochs 50 --dropout 0.5

# 训练改进版 CNN（含残差连接）
python main.py --model improved --epochs 50

# 使用 EfficientNet-B0 迁移学习
python main.py --model transfer --backbone efficientnet_b0 --epochs 80

# 使用 DenseNet-121 迁移学习
python main.py --model transfer --backbone densenet121 --epochs 80
```

### 5.4 关键训练策略

#### 防止背景过拟合

所有图像在实验室统一背景下拍摄，模型可能"偷懒"学习背景。应对措施：

1. **激进数据增强**：`RandomResizedCrop` + `ColorJitter` + `RandomRotation` + `RandomPerspective` + `GaussianBlur`
2. **高 Dropout**：默认 0.5，可在 0.3–0.7 之间调节
3. **Grad-CAM 验证**：训练后检查热力图是否聚焦叶片区域

#### 处理类别不平衡

- **加权损失函数**：自动计算 inverse frequency 权重
- **加权随机采样**：少数类别被更频繁采样
- **可选 Focal Loss**：`--loss focal` 聚焦难分类样本

#### 损失函数选择

```bash
# 标准交叉熵 + 类别权重（默认推荐）
python main.py --loss cross_entropy

# Focal Loss — 适合类别严重不平衡场景
python main.py --loss focal

# Label Smoothing — 防止过拟合，提升泛化
python main.py --loss label_smoothing
```

### 5.5 查看训练日志

```bash
# 启动 TensorBoard
tensorboard --logdir results/logs --port 6006

# 然后在浏览器打开 http://localhost:6006
```

---

## 6. 模型评估

### 6.1 仅评估模式

使用已训练好的模型在测试集上评估：

```bash
python main.py \
    --eval_only \
    --model transfer \
    --backbone resnet50 \
    --checkpoint checkpoints/transfer_resnet50/best_model.pth
```

评估输出包括：

- **整体指标**：Accuracy、Precision、Recall、F1-Score
- **分类报告**：每个类别的精确率/召回率/F1
- **各类别准确率柱状图** → `results/figures/per_class_accuracy.png`
- **混淆矩阵**（原始 + 归一化）→ `results/figures/confusion_matrix.png`
- **Top 10 最易混淆类别对**
- **错误样本可视化** → `results/figures/misclassified_samples.png`
  - 红色边框 = 高置信度错误（>80%）

### 6.2 评估指标说明

```
总体准确率 (Accuracy):      XX.XX%
宏平均精确率 (Precision):   XX.XX%
宏平均召回率 (Recall):      XX.XX%
宏平均 F1-Score:            XX.XX%
```

采用宏平均（macro averaging），对所有类别平等对待，能更好反映模型在少数类别上的表现。

---

## 7. Grad-CAM 可视化

Grad-CAM 用于验证模型是否真正关注叶片病害区域（而非背景），这是评判模型质量的关键依据。

```bash
python main.py \
    --gradcam \
    --model transfer \
    --backbone resnet50 \
    --checkpoint checkpoints/transfer_resnet50/best_model.pth
```

输出：

- Grad-CAM 热力图叠加图像 → `results/figures/gradcam_visualization.png`
- **绿色标题** = 分类正确
- **红色标题** = 分类错误
- **红色热力区** = 模型关注的区域

> 📌 **判断标准**：如果热力区集中在叶片背景而非病斑区域，说明模型发生了背景过拟合，需要加强数据增强或调整 Dropout。

---

## 8. 消融实验

消融实验用于系统地分析不同超参数和配置对模型性能的影响。这是课程作业的 **加分项**。

```bash
python main.py --ablation
```

### 实验维度

| 实验组 | 变量 | 取值 |
|--------|------|------|
| Dropout 比例 | `[0.0, 0.3, 0.5, 0.7]` | 分析正则化强度 |
| 数据增强策略 | `[无增强, 基础增强, 完整增强]` | 分析增强贡献 |
| 学习率 | `[0.01, 0.001, 0.0001]` | 分析收敛性 |
| 批次大小 | `[16, 32, 64]` | 分析稳定性 |
| 模型架构 | `[BasicCNN, ImprovedCNN, ResNet-50, EfficientNet-B0, MobileNetV3, DenseNet-121]` | 架构对比 |
| 损失函数 | `[CE, Focal, LabelSmoothing]` | 损失函数对比 |

### 消融实验输出

实验完成后生成：
- 各组对比柱状图 → `results/figures/ablation_*.png`
- 汇总对比表格（终端输出）

---

## 9. 二阶段微调

二阶段训练可获得更高准确率：

```bash
python main.py \
    --model transfer \
    --backbone resnet50 \
    --epochs 80 \
    --finetune
```

### 训练流程

```
阶段一（冻结骨干）:
  ├── 冻结 ResNet-50 全部卷积层
  ├── 只训练新添加的分类头
  ├── 80 epochs, lr=0.001
  └── 目的：让分类头先收敛

阶段二（微调）:
  ├── 解冻骨干最后 30 层
  ├── 使用更低的 lr=0.0001
  ├── 30 epochs
  └── 目的：针对任务数据微调特征提取
```

> ⚠️ 微调阶段使用更低的初始学习率（`finetune_lr=0.0001`），避免破坏预训练特征。

---

## 10. 常用命令速查

```bash
# ==================== 训练 ====================

# 标准训练（推荐）
python main.py

# 标准训练 + 二阶段微调
python main.py --finetune

# 使用 Focal Loss
python main.py --loss focal

# 使用 OneCycle 调度器
python main.py --scheduler onecycle

# 不同骨干网络
python main.py --backbone efficientnet_b0
python main.py --backbone densenet121
python main.py --backbone mobilenet_v3

# 自定义超参数
python main.py --epochs 100 --batch_size 64 --lr 0.0005 --dropout 0.3

# ==================== 评估 ====================

# 仅测试集评估
python main.py --eval_only --checkpoint checkpoints/transfer_resnet50/best_model.pth

# ==================== 可视化 ====================

# Grad-CAM 热力图
python main.py --gradcam --checkpoint checkpoints/transfer_resnet50/best_model.pth

# 查看 TensorBoard
tensorboard --logdir results/logs

# ==================== 实验 ====================

# 消融实验
python main.py --ablation
```

---

## 11. 结果文件说明

训练完成后，所有结果保存在以下目录：

### `checkpoints/` — 模型文件

```
checkpoints/
└── transfer_resnet50/
    ├── best_model.pth           # 验证集上表现最佳的模型
    └── checkpoint_epoch_N.pth   # 每 10 个 epoch 的定期保存
```

每个 `.pth` 文件包含：
- `model_state_dict` — 模型参数
- `optimizer_state_dict` — 优化器状态（可用于恢复训练）
- `scheduler_state_dict` — 调度器状态
- `val_acc` / `best_val_acc` — 准确率
- `history` — 训练历史

### `results/figures/` — 可视化图表

| 文件名 | 内容 |
|--------|------|
| `training_curves_*.png` | 训练/验证 Loss 和 Accuracy 曲线 |
| `per_class_accuracy.png` | 各类别准确率柱状图 |
| `confusion_matrix.png` | 混淆矩阵（计数 + 归一化） |
| `misclassified_samples.png` | 错误分类样本展示 |
| `gradcam_visualization.png` | Grad-CAM 热力图叠加 |

### `results/logs/` — TensorBoard 日志

```bash
tensorboard --logdir results/logs
```

包含的训练指标：`Loss/train`、`Loss/val`、`Accuracy/train`、`Accuracy/val`、`LR`

### `results/metrics.json` — 评估指标

```json
{
  "model": "transfer_resnet50",
  "epochs": 80,
  "batch_size": 32,
  "learning_rate": 0.001,
  "dropout": 0.5,
  "loss_function": "cross_entropy",
  "accuracy": 0.xxxx,
  "precision": 0.xxxx,
  "recall": 0.xxxx,
  "f1": 0.xxxx
}
```

---

## 配置参数参考

所有可配置的默认参数集中在 `src/config.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `IMAGE_SIZE` | 224 | 输入图像尺寸 |
| `BATCH_SIZE` | 32 | 训练批次大小 |
| `NUM_WORKERS` | 4 | DataLoader 工作线程数 |
| `EPOCHS` | 80 | 默认训练轮数 |
| `LEARNING_RATE` | 0.001 | 初始学习率 |
| `WEIGHT_DECAY` | 1e-4 | 权重衰减（AdamW） |
| `FINETUNE_LR` | 0.0001 | 微调阶段学习率 |
| `FINETUNE_EPOCHS` | 30 | 微调阶段轮数 |
| `DROPOUT_RATE` | 0.5 | Dropout 比例 |
| `EARLY_STOP_PATIENCE` | 20 | 早停耐心值 |
| `GRAD_CLIP_MAX_NORM` | 1.0 | 梯度裁剪阈值 |
| `USE_AMP` | True | 混合精度训练 |
| `SEED` | 42 | 随机种子 |
| `TRAIN_RATIO` | 0.70 | 训练集比例 |
| `VAL_RATIO` | 0.15 | 验证集比例 |
| `TEST_RATIO` | 0.15 | 测试集比例 |
