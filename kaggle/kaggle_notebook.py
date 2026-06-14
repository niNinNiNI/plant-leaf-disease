"""
Kaggle Notebook 完整训练脚本
=============================
使用方法:
  1. 在 Kaggle 创建一个新 Notebook
  2. 把本文件内容复制到第一个 Cell
  3. 在 Kaggle 右侧 Add Data → 添加你的 plantvillage 数据集
  4. 修改下面的 DATASET_NAME 为你的数据集名称
  5. Run All

数据集结构要求:
  /kaggle/input/你的数据集名/
    ├── Pepper__Bacterial_spot/
    ├── Pepper__Healthy/
    ├── Potato__Early_blight/
    └── ... (共15个类别文件夹)
"""

# ============================================================
# Cell 1: 安装依赖 & 环境检测
# ============================================================
!pip install seaborn opencv-python tensorboard -q

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader


# ============================================================
# Cell 2: 配置路径（修改这里的 DATASET_NAME）
# ============================================================
import os
import sys

# ⚠️ 改成你在 Kaggle 上创建的数据集名称
# 格式: "你的kaggle用户名/数据集名"
DATASET_NAME = "YOUR_USERNAME/plantvillage-dataset"

DATA_ROOT = f"/kaggle/input/{DATASET_NAME.split('/')[-1]}"
# 如果上面的路径不对，尝试直接列出 /kaggle/input/
input_dirs = os.listdir("/kaggle/input")
print("可用的数据集:", input_dirs)

# 自动找 PlantVillage 目录
for d in input_dirs:
    full = f"/kaggle/input/{d}"
    subdirs = os.listdir(full) if os.path.isdir(full) else []
    # PlantVillage 包含 Pepper/Potato/Tomato 文件夹
    if any("Pepper" in s or "Potato" in s or "Tomato" in s for s in subdirs):
        DATA_ROOT = full
        break

print(f"数据集路径: {DATA_ROOT}")
print(f"类别文件夹: {os.listdir(DATA_ROOT)}")


# ============================================================
# Cell 3: 拷贝 src 代码到工作目录（因为需要 import）
# ============================================================
# 从本地项目拷贝 src 目录
# 方法：把 src/ 下的文件直接写进 /kaggle/working/src/

!mkdir -p /kaggle/working/src

# ====== config.py ======
%%writefile /kaggle/working/src/config.py
"""
全局配置参数
"""
import os

class Config:
    """全局配置"""

    # ==================== 路径 ====================
    BASE_DIR = "/kaggle/working"
    DATA_ROOT = ""  # 运行时动态设置
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    RESULT_DIR = os.path.join(BASE_DIR, "results")
    FIGURE_DIR = os.path.join(RESULT_DIR, "figures")
    LOG_DIR = os.path.join(RESULT_DIR, "logs")

    # ==================== 数据 ====================
    IMAGE_SIZE = 224
    NUM_CLASSES = 15
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # ImageNet 标准化参数
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ==================== 训练 ====================
    BATCH_SIZE = 32
    NUM_WORKERS = 2        # Kaggle 上线程数不要太高
    EPOCHS = 80
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    FINETUNE_LR = 0.0001
    FINETUNE_EPOCHS = 30

    # ==================== 正则化 ====================
    DROPOUT_RATE = 0.5
    LABEL_SMOOTHING = 0.0

    # ==================== 早停 ====================
    EARLY_STOP_PATIENCE = 20

    # ==================== 混合精度 ====================
    USE_AMP = True

    # ==================== 梯度裁剪 ====================
    GRAD_CLIP_MAX_NORM = 1.0

    # ==================== 随机种子 ====================
    SEED = 42

    # ==================== 迁移学习 ====================
    FREEZE_BACKBONE = True
    BACKBONE = "resnet50"


# ====== __init__.py ======
%%writefile /kaggle/working/src/__init__.py
# src package


# ====== 复制其他 src 文件 ======
# 把本地项目的 src/*.py 内容上传
# 方法: 在本地运行以下命令打包 src，然后上传为 Kaggle Dataset：
#   tar -czf src_code.tar.gz src/
# 然后在 Kaggle Add Data 添加这个代码数据集


# ============================================================
# Cell 4: 解压本地 src 代码（如果你上传了 src_code.tar.gz）
# ============================================================
# 如果有上传代码包，解压它：
# !tar -xzf /kaggle/input/plant-code/src_code.tar.gz -C /kaggle/working/
# 如果没有，则手动写每个文件（见下面 Cell 5）


# ============================================================
# Cell 5: 直接训练（使用 subprocess 调用 main.py，最简单）
# ============================================================
import subprocess

# 如果你把 main.py 也打包上传了：
main_py = "/kaggle/working/main.py"

# 或者直接运行你本地上传的 main.py：
# main_py = "/kaggle/input/plant-code/main.py"

cmd = f"""
python {main_py} \
    --model transfer \
    --backbone resnet50 \
    --epochs 80 \
    --batch_size 64 \
    --lr 0.001 \
    --dropout 0.5 \
    --data_root {DATA_ROOT}
"""

print("执行命令:", cmd)
!{cmd}


# ============================================================
# Cell 6: 打包下载结果
# ============================================================
import os
import tarfile
from IPython.display import FileLink

# 打包结果目录
output_tar = "/kaggle/working/plant_disease_results.tar.gz"

!cd /kaggle/working && tar -czf {output_tar} \
    checkpoints/ \
    results/ \
    --ignore-failed-read 2>/dev/null

if os.path.exists(output_tar):
    print(f"✅ 结果已打包: {output_tar}")
    print(f"   文件大小: {os.path.getsize(output_tar) / 1024 / 1024:.1f} MB")
    print("\n👇 点击右侧 Output 标签下载 results.tar.gz")
else:
    print("❌ 未找到结果文件")

# 显示关键指标
metrics_path = "/kaggle/working/results/metrics.json"
if os.path.exists(metrics_path):
    import json
    with open(metrics_path) as f:
        metrics = json.load(f)
    print(f"\n📊 训练结果:")
    print(f"   准确率: {metrics.get('accuracy', 'N/A')}")
    print(f"   F1:     {metrics.get('f1', 'N/A')}")
    print(f"   精度:   {metrics.get('precision', 'N/A')}")
    print(f"   召回:   {metrics.get('recall', 'N/A')}")
