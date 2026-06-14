#!/usr/bin/env python3
"""生成可直接上传到 Kaggle 的 .ipynb 文件"""
import json

cells = []

def add_md(source):
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [source]
    })

def add_code(source):
    cells.append({
        'cell_type': 'code',
        'metadata': {},
        'source': [source],
        'outputs': []
    })

# ==================== Cell 0: Title ====================
add_md("""# 🌿 农业植物叶片病害识别 — Kaggle GPU 训练

**基于 ResNet-50 迁移学习 | PlantVillage 15分类 | PyTorch**

---
### ⚡ 使用前请确认:
1. 右侧 **Accelerator** 选择 **GPU T4 x2**
2. 左侧 **Add Data** → 添加你的 **PlantVillage 数据集**
3. 左侧 **Add Data** → 添加 **plant-disease-code** 代码包

### 📋 执行顺序: Cell 1 → 2 → 3 → 4 → 5, 逐行点击运行
""")

# ==================== Cell 1: 环境 ====================
add_code("""# ============================================================
# Cell 1: 安装依赖 & 检测 GPU
# ============================================================
!pip install seaborn opencv-python tensorboard -q

import torch
import os
import sys

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")
    !nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else:
    print("ERROR: GPU not available! Select GPU T4 x2 in right panel Accelerator")
""")

# ==================== Cell 2: Find dataset ====================
add_code("""# ============================================================
# Cell 2: 自动查找 PlantVillage 数据集路径
# ============================================================
DATA_ROOT = None

print("搜索 /kaggle/input 中的数据集...")
for d in sorted(os.listdir("/kaggle/input")):
    full = f"/kaggle/input/{d}"
    if not os.path.isdir(full):
        continue
    try:
        subdirs = sorted(os.listdir(full))
    except Exception:
        continue

    # PlantVillage 特征: 包含 Pepper/Potato/Tomato 子目录
    plant_count = sum(1 for s in subdirs
                      if any(k in s for k in ["Pepper", "Potato", "Tomato"]))
    print(f"  {d}: {len(subdirs)} dirs, {plant_count} plant-related")

    if plant_count >= 3:
        DATA_ROOT = full
        print(f"    => MATCH! Using this dataset")
        break

if DATA_ROOT is None:
    raise RuntimeError(
        "Dataset not found!\\n"
        "Please Add Data -> search and add your PlantVillage dataset.\\n"
        "It should contain folders like Pepper__xxx, Potato__xxx, Tomato__xxx"
    )

CLASSES = sorted(os.listdir(DATA_ROOT))
print(f"\\nDataset: {DATA_ROOT}")
print(f"Classes: {len(CLASSES)}")
for c in CLASSES:
    count = len(os.listdir(f"{DATA_ROOT}/{c}"))
    print(f"  {c}: {count} images")
""")

# ==================== Cell 3: Extract code ====================
add_code("""# ============================================================
# Cell 3: 解压项目代码
# ============================================================
code_found = False
for d in os.listdir("/kaggle/input"):
    full = f"/kaggle/input/{d}"
    if not os.path.isdir(full):
        continue
    for f in os.listdir(full):
        if 'tar.gz' in f:
            print(f"Code package found: {f}")
            !tar -xzf {full}/{f} -C /kaggle/working/
            code_found = True
            break
    if code_found:
        break

if code_found:
    print("Code extracted to /kaggle/working/")
    print("\\nFiles:")
    !ls -la /kaggle/working/
    print("\\nsrc/ files:")
    !ls /kaggle/working/src/
else:
    print("WARNING: No .tar.gz code package found in /kaggle/input/")
    print("Did you Add Data -> plant-disease-code?")
    print("\\nThe next cells will still work if code is elsewhere.")
""")

# ==================== Cell 4: Train ====================
add_code("""# ============================================================
# Cell 4: 🚀 开始训练 (ResNet-50 迁移学习, 80 epochs)
# ============================================================
import sys
sys.path.insert(0, "/kaggle/working")

# 创建输出目录
os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
os.makedirs("/kaggle/working/results/figures", exist_ok=True)
os.makedirs("/kaggle/working/results/logs", exist_ok=True)

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---- 导入项目模块 ----
from src.config import Config
from src.utils import set_seed, get_device, count_parameters, plot_training_curves
from src.dataset import split_dataset, compute_class_weights, create_dataloaders
from src.models import create_model
from src.losses import get_criterion
from src.trainer import Trainer

# 适配 Kaggle 环境
Config.DATA_ROOT = DATA_ROOT
Config.BATCH_SIZE = 64
Config.NUM_WORKERS = 2
Config.CHECKPOINT_DIR = "/kaggle/working/checkpoints"
Config.RESULT_DIR = "/kaggle/working/results"
Config.FIGURE_DIR = "/kaggle/working/results/figures"
Config.LOG_DIR = "/kaggle/working/results/logs"

set_seed(42)
device = get_device()
print(f"Device: {device}")

# ---- 数据加载 ----
print("\\nLoading dataset...")
train_ds, val_ds, test_ds, class_to_idx, _ = split_dataset(
    DATA_ROOT,
    train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
    seed=42, image_size=224
)

num_classes = len(class_to_idx)
class_names = list(class_to_idx.keys())
print(f"Classes: {num_classes}")
print(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

class_weights = compute_class_weights(train_ds, num_classes)
train_loader, val_loader, test_loader = create_dataloaders(
    train_ds, val_ds, test_ds,
    num_classes=num_classes,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS
)

# ---- 模型 ----
print("\\nCreating ResNet-50 model...")
model = create_model(
    model_type="transfer", backbone="resnet50",
    num_classes=num_classes, dropout_rate=0.5,
    freeze_backbone=True
)
count_parameters(model)

# ---- 训练配置 ----
criterion = get_criterion("cross_entropy", class_weights=class_weights.to(device))
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6)

trainer = Trainer(
    model=model, device=device,
    criterion=criterion, optimizer=optimizer, scheduler=scheduler,
    use_amp=True, early_stop_patience=20,
    save_dir="/kaggle/working/checkpoints/transfer_resnet50",
    log_dir="/kaggle/working/results/logs/transfer_resnet50",
    grad_clip_max_norm=1.0
)

# ---- 开始训练 ----
print("\\n" + "=" * 60)
print("ResNet-50 Transfer Learning | 80 epochs | batch=64")
print("=" * 60)

history = trainer.train(train_loader, val_loader, epochs=80)

plot_training_curves(history, save_path="/kaggle/working/results/figures/training_curves.png")
print("\\nTraining complete!")
""")

# ==================== Cell 5: Evaluate + Package ====================
add_code("""# ============================================================
# Cell 5: 测试评估 + Grad-CAM + 打包下载
# ============================================================
from src.evaluate import evaluate_on_test, analyze_misclassified
from src.gradcam import visualize_gradcam
import json

checkpoint = "/kaggle/working/checkpoints/transfer_resnet50/best_model.pth"
idx_to_class = {v: k for k, v in class_to_idx.items()}

# ---- Test evaluation ----
print("\\n" + "=" * 60)
print("Test Set Evaluation")
print("=" * 60)
results = evaluate_on_test(
    model, test_loader, class_names, device,
    checkpoint_path=checkpoint
)
analyze_misclassified(model, test_ds, idx_to_class, device)

# ---- Grad-CAM ----
print("\\n" + "=" * 60)
print("Grad-CAM Visualization")
print("=" * 60)
visualize_gradcam(
    model, test_ds, device, class_names,
    model_type="transfer", backbone="resnet50",
    num_samples=9
)

# ---- Save metrics ----
metrics = {
    "model": "transfer_resnet50", "epochs": 80, "batch_size": 64,
    "accuracy": float(results["accuracy"]), "precision": float(results["precision"]),
    "recall": float(results["recall"]), "f1": float(results["f1"]),
}
with open("/kaggle/working/results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print(f"\\nAccuracy: {metrics['accuracy']*100:.2f}%")
print(f"F1-Score:  {metrics['f1']:.4f}")

# ---- Package for download ----
!cd /kaggle/working && tar -czf plant_disease_output.tar.gz \\
    checkpoints/ results/ --ignore-failed-read 2>/dev/null

from IPython.display import FileLink, display

out_path = "/kaggle/working/plant_disease_output.tar.gz"
size_mb = os.path.getsize(out_path) / 1024 / 1024
print(f"\\nOutput: plant_disease_output.tar.gz ({size_mb:.1f} MB)")
print("Click below to download, or check Output tab on the right:")
display(FileLink("plant_disease_output.tar.gz"))
print("\\nAll done!")
""")

# ==================== Build notebook ====================
notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.10.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

outpath = '/home/nini/文档/深度学习大作业/个人作业/kaggle_train.ipynb'
with open(outpath, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f'✅ Generated: {outpath}')
print(f'   Cells: {len(cells)}')
