"""
================================================================================
Kaggle Notebook — 农业植物叶片病害识别 完整训练
================================================================================

使用方法（只需 5 步）:
  Step 1: 运行 prepare_for_kaggle.sh 打包代码
  Step 2: 在 Kaggle 创建两个 Dataset:
          - plant-disease-code (上传 plant_code.tar.gz)
          - plantvillage-15 (上传 plantvillage.tar.gz)
  Step 3: 在 Kaggle 创建 New Notebook（GPU T4 x2）
  Step 4: Add Data → 添加上面两个 Dataset
  Step 5: 按顺序运行下面的 Cell

================================================================================
"""

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Cell 1: 环境检查 & 安装依赖                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

!pip install seaborn opencv-python tensorboard -q

import torch
import os
print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"显存:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    !nvidia-smi


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Cell 2: 查找数据集 & 解压代码                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os
import sys

# ---- 查找数据集路径 ----
DATA_ROOT = None
for d in os.listdir("/kaggle/input"):
    full = f"/kaggle/input/{d}"
    if not os.path.isdir(full):
        continue
    try:
        subdirs = os.listdir(full)
    except:
        continue
    # PlantVillage 的特征: 包含 Pepper/Potato/Tomato 子文件夹
    if sum(1 for s in subdirs if "Pepper" in s or "Potato" in s or "Tomato" in s) >= 3:
        DATA_ROOT = full
        break

if DATA_ROOT is None:
    print("❌ 未找到 PlantVillage 数据集!")
    print("请在右侧 Add Data 添加数据集，确保数据包含 Pepper/Potato/Tomato 文件夹")
    print("\n当前 /kaggle/input/ 内容:")
    !ls -la /kaggle/input/
    raise SystemExit(1)

print(f"✅ 数据集: {DATA_ROOT}")
classes = sorted(os.listdir(DATA_ROOT))
print(f"   类别数: {len(classes)}")
print(f"   类别:   {classes}")

# ---- 解压代码 ----
CODE_TAR = None
for d in os.listdir("/kaggle/input"):
    if d.startswith("plant") and d.endswith("code"):
        for f in os.listdir(f"/kaggle/input/{d}"):
            if f.endswith(".tar.gz"):
                CODE_TAR = f"/kaggle/input/{d}/{f}"
                break

if CODE_TAR and os.path.exists(CODE_TAR):
    print(f"\n✅ 代码包: {CODE_TAR}")
    !tar -xzf {CODE_TAR} -C /kaggle/working/
    print("   代码已解压到 /kaggle/working/")
    !ls /kaggle/working/
else:
    print("\n⚠️  未找到代码包，将使用内嵌的训练代码")
    # 如果没有上传代码包，下面的 Cell 会自动使用 main.py 的直接调用方式


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Cell 3: 直接训练                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import sys
sys.path.insert(0, "/kaggle/working")

# 设置环境变量让 config.py 能找到路径
os.environ["DATA_ROOT"] = DATA_ROOT

# 直接导入项目模块训练
from src.config import Config

# 修改配置适配 Kaggle 环境
Config.DATA_ROOT = DATA_ROOT
Config.BATCH_SIZE = 64          # T4 16GB 可以用 64
Config.NUM_WORKERS = 2          # Kaggle 限制
Config.CHECKPOINT_DIR = "/kaggle/working/checkpoints"
Config.RESULT_DIR = "/kaggle/working/results"
Config.FIGURE_DIR = "/kaggle/working/results/figures"
Config.LOG_DIR = "/kaggle/working/results/logs"

# 创建必要目录
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.FIGURE_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils import set_seed, get_device, count_parameters, plot_training_curves
from src.dataset import split_dataset, compute_class_weights, create_dataloaders
from src.models import create_model
from src.losses import get_criterion
from src.trainer import Trainer, get_scheduler
from src.evaluate import evaluate_on_test, analyze_misclassified
from src.gradcam import visualize_gradcam

# 固定随机种子
set_seed(Config.SEED)
device = get_device()
print(f"设备: {device}")

# ---- 加载数据 ----
print("\n加载 PlantVillage 数据集...")
train_dataset, val_dataset, test_dataset, class_to_idx, eval_data = split_dataset(
    DATA_ROOT,
    train_ratio=Config.TRAIN_RATIO,
    val_ratio=Config.VAL_RATIO,
    test_ratio=Config.TEST_RATIO,
    seed=Config.SEED,
    image_size=Config.IMAGE_SIZE
)

num_classes = len(class_to_idx)
idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = list(class_to_idx.keys())
print(f"类别数: {num_classes}")
print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

# 类别权重（处理不平衡）
class_weights = compute_class_weights(train_dataset, num_classes)

# DataLoader
train_loader, val_loader, test_loader = create_dataloaders(
    train_dataset, val_dataset, test_dataset,
    num_classes=num_classes,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS
)

# ---- 创建模型 ----
print("\n创建 ResNet-50 迁移学习模型...")
model = create_model(
    model_type="transfer",
    backbone="resnet50",
    num_classes=num_classes,
    dropout_rate=0.5,
    freeze_backbone=True   # 冻结骨干
)
count_parameters(model)

# ---- 训练配置 ----
criterion = get_criterion("cross_entropy", class_weights=class_weights.to(device))
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6)

# ---- 开始训练 ----
trainer = Trainer(
    model=model,
    device=device,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    use_amp=True,
    early_stop_patience=20,
    save_dir="/kaggle/working/checkpoints/transfer_resnet50",
    log_dir="/kaggle/working/results/logs/transfer_resnet50",
    grad_clip_max_norm=1.0
)

print("\n" + "=" * 60)
print("开始训练: ResNet-50 迁移学习 (80 epochs, batch=64)")
print("=" * 60)

history = trainer.train(train_loader, val_loader, epochs=80)

# 绘制训练曲线
plot_training_curves(
    history,
    save_path="/kaggle/working/results/figures/training_curves.png"
)

# ---- 测试集评估 ----
print("\n" + "=" * 60)
print("测试集评估")
print("=" * 60)

best_checkpoint = "/kaggle/working/checkpoints/transfer_resnet50/best_model.pth"
results = evaluate_on_test(
    model, test_loader, class_names, device,
    checkpoint_path=best_checkpoint
)

analyze_misclassified(model, test_dataset, idx_to_class, device)

# ---- Grad-CAM ----
visualize_gradcam(
    model, test_dataset, device, class_names,
    model_type="transfer", backbone="resnet50",
    num_samples=9
)

# ---- 保存指标 ----
import json
metrics = {
    "model": "transfer_resnet50",
    "epochs": 80,
    "batch_size": 64,
    "accuracy": float(results["accuracy"]),
    "precision": float(results["precision"]),
    "recall": float(results["recall"]),
    "f1": float(results["f1"]),
}
with open("/kaggle/working/results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ 训练完成!")
print(f"准确率: {metrics['accuracy']*100:.2f}%")
print(f"F1:      {metrics['f1']:.4f}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Cell 4: 打包结果并下载                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import tarfile
import os
from IPython.display import FileLink, display

output = "/kaggle/working/plant_disease_output.tar.gz"

# 打包 checkpoints + results
!cd /kaggle/working && tar -czf plant_disease_output.tar.gz \
    checkpoints/ \
    results/ \
    --ignore-failed-read 2>/dev/null

size_mb = os.path.getsize(output) / 1024 / 1024
print(f"📦 输出包: plant_disease_output.tar.gz ({size_mb:.1f} MB)")
print(f"📁 包含内容:")
!tar -tzf /kaggle/working/plant_disease_output.tar.gz | head -30
print("...")
print(f"\n👇 点击下方链接下载，或到右侧 Output 面板下载")

display(FileLink("plant_disease_output.tar.gz"))

print("\n🎉 全部完成！下载 plant_disease_output.tar.gz 到本地解压即可。")
