# Kaggle GPU 使用指南 — 农业植物叶片病害识别

## 目录

1. [Kaggle GPU 概述](#1-kaggle-gpu-概述)
2. [准备工作：打包项目文件](#2-准备工作打包项目文件)
3. [上传数据到 Kaggle](#3-上传数据到-kaggle)
4. [创建并运行 Notebook](#4-创建并运行-notebook)
5. [下载训练结果](#5-下载训练结果)
6. [常见问题与技巧](#6-常见问题与技巧)
7. [各方案对比](#7-各方案对比)

---

## 1. Kaggle GPU 概述

| 项目 | 说明 |
|------|------|
| **GPU 型号** | Tesla T4 ×2（16GB 显存）或 P100 |
| **免费额度** | 每周 30 小时 GPU 时间 |
| **单次时长** | 最长 9 小时（80 epoch 约需 3–5 小时） |
| **预装环境** | PyTorch、TensorFlow、CUDA 均已预装 |
| **存储** | `/kaggle/input/` 读数据集，`/kaggle/working/` 写输出 |
| **费用** | 完全免费 |

### 适用场景

- ✅ 你的 21,639 张图 + ResNet-50 80 epoch 完全够用
- ✅ 消融实验（多组对比）可开多个 Notebook 排队跑
- ❌ 需要训练几天几夜的大模型（超出 9h 限制）
- ❌ 需要超过 16GB 显存的超大 batch_size

---

## 2. 准备工作：打包项目文件

### 2.1 打包项目代码

```bash
cd /home/nini/文档/深度学习大作业/个人作业

# 赋予执行权限
chmod +x prepare_for_kaggle.sh

# 运行打包脚本（生成 plant_code.tar.gz 和 plantvillage.tar.gz）
./prepare_for_kaggle.sh
```

**打包内容说明：**

- `plant_code.tar.gz`：包含 `main.py`、`requirements.txt`、`src/` 目录
- `plantvillage.tar.gz`：包含 `PlantVillage/` 下 15 个类别的图像文件夹

### 2.2 手动打包（备用方式）

如果脚本无法运行，手动执行：

```bash
# 打包代码
tar -czf plant_code.tar.gz \
    --exclude='checkpoints' \
    --exclude='results' \
    --exclude='__pycache__' \
    --exclude='archive' \
    --exclude='*.tar.gz' \
    main.py requirements.txt src/

# 打包数据（假设数据在 archive/plantvillage/PlantVillage/）
tar -czf plantvillage.tar.gz \
    -C /home/nini/文档/深度学习大作业/个人作业/archive/plantvillage/ \
    PlantVillage/
```

---

## 3. 上传数据到 Kaggle

### 3.1 通过网页上传（推荐）

1. 打开 https://www.kaggle.com 并登录
2. 右上角 **Create → New Dataset**
3. 将 `plant_code.tar.gz` **直接拖拽**到上传区域
4. 填写标题：`plant-disease-code`，设为 **Private**
5. 点击 **Create**
6. 重复步骤 2–5，上传 `plantvillage.tar.gz`
   - 标题：`plantvillage-dataset`，设为 **Private**
   - ⚠️ 数据集较大（约 2–3 GB），上传需要 10–20 分钟

### 3.2 数据集目录结构要求

Kaggle 上的数据集解压后应该保持如下结构：

```
plantvillage-dataset/
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

Notebook 中的 Cell 2 会自动扫描 `/kaggle/input/` 并识别包含 Pepper/Potato/Tomato 子文件夹的数据集。

### 3.3 通过 Kaggle API 上传（高级）

```bash
# 安装 Kaggle CLI
pip install kaggle

# 去 Kaggle → Settings → API → Create New Token，下载 kaggle.json
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 进入数据集目录
cd /home/nini/文档/深度学习大作业/个人作业/archive/plantvillage/PlantVillage

# 创建元数据文件 dataset-metadata.json
cat > dataset-metadata.json << 'EOF'
{
  "title": "PlantVillage-15-Classes",
  "id": "你的用户名/plantvillage-15-classes",
  "licenses": [{ "name": "CC0-1.0" }]
}
EOF

# 上传
kaggle datasets create -p . --dir-mode zip
```

---

## 4. 创建并运行 Notebook

### 4.1 方式一：上传已准备好的 .ipynb（推荐）

项目已包含 `kaggle_train.ipynb`，可直接上传：

1. Kaggle 首页 → **Create → New Notebook**
2. 菜单栏 **File → Upload Notebook** → 选择 `kaggle_train.ipynb`
3. 右侧面板设置：
   - **Accelerator**：选 `GPU T4 x2`
   - **Persistence**：选 `Files only`
4. 左侧 **Add Data**（🔍 搜索图标）：
   - 搜索 `plant-disease-code` → 点击 **Add**
   - 搜索 `plantvillage-dataset` → 点击 **Add**
5. 点击 **Run All**（或逐 Cell 运行）

### 4.2 方式二：手动复制 Cell 内容

如果 .ipynb 上传失败，可以用 `kaggle_notebook_complete.py` 中的内容：

1. 创建一个空白 Notebook
2. 将 `kaggle_notebook_complete.py` 中每个 `# ╔═══ Cell N` 区块的内容复制到对应 Cell
3. 逐 Cell 运行

### 4.3 Notebook Cell 说明

| Cell | 功能 | 预计耗时 |
|------|------|---------|
| Cell 1 | 安装依赖 + 检测 GPU | ~30 秒 |
| Cell 2 | 自动查找数据集路径 | ~5 秒 |
| Cell 3 | 解压项目代码 | ~10 秒 |
| Cell 4 | 🚀 训练 ResNet-50 (80 epochs) | ~3–5 小时 |
| Cell 5 | 测试评估 + Grad-CAM + 打包下载 | ~10 分钟 |

### 4.4 训练参数调整

如需修改训练参数，在 Cell 4 中找到以下变量并修改：

```python
# 在 Cell 4 中修改这些值
Config.BATCH_SIZE = 64        # T4 16GB 用 64，想更快可试 96
# ...
trainer.train(train_loader, val_loader, epochs=80)  # 可调整 epoch 数
```

### 4.5 监控训练进度

训练过程中，可以在 Notebook 中查看实时输出（Loss/Accuracy 每个 epoch 打印一次）。

查看 TensorBoard（如需）：

```python
# 在 Notebook 中启动 TensorBoard
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/results/logs
```

---

## 5. 下载训练结果

### 5.1 通过 Notebook（推荐）

Cell 5 运行完毕后会显示下载链接，点击即可下载 `plant_disease_output.tar.gz`。

也可以到右侧 **Output** 面板手动下载。

### 5.2 文件内容

解压后的结果目录结构：

```
plant_disease_output/
├── checkpoints/
│   └── transfer_resnet50/
│       ├── best_model.pth            # 验证集最佳模型
│       └── checkpoint_epoch_N.pth    # 定期保存的检查点
└── results/
    ├── metrics.json                  # 评估指标 (JSON)
    ├── figures/
    │   ├── training_curves.png       # 训练/验证曲线
    │   ├── per_class_accuracy.png    # 各类别准确率
    │   ├── confusion_matrix.png      # 混淆矩阵
    │   ├── misclassified_samples.png # 错误样本可视化
    │   └── gradcam_visualization.png # Grad-CAM 热力图
    └── logs/
        └── transfer_resnet50/        # TensorBoard 日志
```

### 5.3 解压到本地

```bash
# 下载后解压到项目目录
tar -xzf plant_disease_output.tar.gz -C /home/nini/文档/深度学习大作业/个人作业/

# 现在本地也可以做评估和可视化
python main.py --eval_only --checkpoint checkpoints/transfer_resnet50/best_model.pth
python main.py --gradcam --checkpoint checkpoints/transfer_resnet50/best_model.pth
```

---

## 6. 常见问题与技巧

### 6.1 数据集找不到

**症状**：Cell 2 输出 `Dataset not found!`

**解决**：
1. 确认左侧 **Add Data** 已添加 PlantVillage 数据集
2. 确认数据集的子文件夹名包含 `Pepper`、`Potato`、`Tomato`
3. 如果上传的是 `.tar.gz`，Kaggle 会自动解压，确保解压后目录结构正确
4. 手动指定路径：在 Cell 4 开头加 `DATA_ROOT = "/kaggle/input/你的数据集名称"`

### 6.2 GPU 不可用

**症状**：Cell 1 输出 `GPU not available`

**解决**：
1. 右侧面板 **Accelerator** 选 `GPU T4 x2`
2. 注意：Kaggle 每周 30 小时额度用完后就只能用 CPU 了
3. 在 https://www.kaggle.com/settings 查看剩余额度

### 6.3 Session 断开

**症状**：训练中途 Notebook 自动断开

**原因**：
- Kaggle 空闲超时（无输出约 1 小时后断开）
- 超过单次 9 小时限制
- 浏览器关闭

**解决**：
- 确保每个 epoch 都有输出（trainer 默认会打印）
- 不要关闭浏览器标签页
- 使用早停机制（项目已配置 `EARLY_STOP_PATIENCE=20`）
- 利用 checkpoint 恢复训练：

```python
# 恢复训练示例（修改 Cell 4）
checkpoint = torch.load("/kaggle/working/checkpoints/transfer_resnet50/checkpoint_epoch_40.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# 然后从 epoch 41 继续训练
```

### 6.4 加速训练技巧

```python
# 1. 增大 batch_size（T4 16GB 可试试 96–128）
Config.BATCH_SIZE = 96

# 2. 减少 num_workers（Kaggle 建议 ≤2）
Config.NUM_WORKERS = 2

# 3. 使用混合精度训练（已默认开启）
Config.USE_AMP = True

# 4. 减少 epoch（如果只是想快速验证）
trainer.train(train_loader, val_loader, epochs=30)
```

### 6.5 同时跑多组实验（消融实验）

可以创建多个 Notebook，每个 Notebook 跑一组参数：

- Notebook 1：`resnet50` + `cross_entropy` + `dropout=0.5`
- Notebook 2：`efficientnet_b0` + `cross_entropy` + `dropout=0.5`
- Notebook 3：`resnet50` + `focal_loss` + `dropout=0.3`

每个 Notebook 修改 Cell 4 中相应的参数即可。

### 6.6 上传文件大小限制

Kaggle Dataset 单文件限制：
- 网页上传：无明确限制（大文件慢但稳定）
- API 上传：单文件最大 20 GB
- 你的数据集约 2–3 GB，完全没问题

---

## 7. 各方案对比

| 特性 | Kaggle | AutoDL | Google Colab | 学校服务器 |
|------|--------|--------|-------------|-----------|
| **费用** | 免费 | ¥1.5–8/h | 免费/¥70月 | 免费 |
| **GPU** | T4 16GB | 4090/A5000/A100 | T4/A100 | 不确定 |
| **每周时长** | 30h | 按量付费 | 有限制 | 不限 |
| **单次最长** | 9h | 不限 | 12h(Pro) | 不限 |
| **预装环境** | PyTorch 等 | 多种镜像 | PyTorch 等 | 看情况 |
| **网络要求** | 需科学上网 | 国内直连 | 需科学上网 | 校内/VPN |
| **适合场景** | 免费训练 | 大模型/长时间 | 快速验证 | 实验室资源 |

### 推荐策略

```
代码调试/小规模验证  → 本地 CPU 或 Colab
正式训练 (80 epoch)  → Kaggle（免费）或 AutoDL（付费更快）
多组消融实验         → Kaggle 多开 Notebook 排队
超大规模训练         → AutoDL 租 A100
```

---

## 附录：生成的文件清单

| 文件 | 路径 | 说明 |
|------|------|------|
| 打包脚本 | `prepare_for_kaggle.sh` | 一键打包代码和数据 |
| Kaggle Notebook | `kaggle_train.ipynb` | 直接上传到 Kaggle |
| 纯文本 Notebook | `kaggle_notebook_complete.py` | Cell 内容参考 |
| Notebook 生成器 | `build_ipynb.py` | 重新生成 .ipynb |
| 本指南 | `Kaggle_GPU_使用指南.md` | 本文档 |
