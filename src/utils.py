"""
工具函数：随机种子、参数统计、训练曲线绘制
"""
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """获取可用设备"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("使用 CPU")
    return device


def count_parameters(model):
    """统计模型参数量和计算量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{'='*60}")
    print(f"模型参数统计")
    print(f"{'='*60}")
    print(f"  总参数量:     {total_params:>12,}")
    print(f"  可训练参数:   {trainable_params:>12,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  不可训练参数: {total_params - trainable_params:>12,}")

    size_mb = total_params * 4 / (1024 * 1024)
    print(f"  模型大小:     {size_mb:>10.1f} MB")

    print(f"\n  各模块参数量:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"    {name:<25s}: {params:>10,}")

    return total_params, trainable_params


def plot_training_curves(history, save_path='results/figures/training_curves.png'):
    """
    绘制训练过程中的 Loss 和 Accuracy 曲线
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # ---- (1) Loss 曲线 ----
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    min_val_loss_idx = np.argmin(history['val_loss']) + 1
    min_val_loss = history['val_loss'][min_val_loss_idx - 1]
    ax.annotate(f'Min Val Loss: {min_val_loss:.3f}\n(Epoch {min_val_loss_idx})',
                xy=(min_val_loss_idx, min_val_loss),
                xytext=(min_val_loss_idx + 5, min_val_loss + 0.2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    # ---- (2) Accuracy 曲线 ----
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Training Accuracy')
    ax.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Validation Accuracy')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    max_val_acc_idx = np.argmax(history['val_acc']) + 1
    max_val_acc = history['val_acc'][max_val_acc_idx - 1]
    ax.annotate(f'Max Val Acc: {max_val_acc:.1f}%\n(Epoch {max_val_acc_idx})',
                xy=(max_val_acc_idx, max_val_acc),
                xytext=(max_val_acc_idx + 5, max_val_acc - 5),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')

    # ---- (3) 学习率变化 ----
    ax = axes[1, 0]
    if 'lr' in history:
        ax.plot(epochs, history['lr'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    # ---- (4) 过拟合分析 ----
    ax = axes[1, 1]
    generalization_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    ax.fill_between(epochs, 0, generalization_gap, alpha=0.3, color='orange')
    ax.plot(epochs, generalization_gap, 'o-', linewidth=1.5, color='darkorange', markersize=3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Generalization Gap (%)', fontsize=12)
    ax.set_title('Overfitting Indicator (Train Acc - Val Acc)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Training Progress Summary', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存到: {save_path}")
