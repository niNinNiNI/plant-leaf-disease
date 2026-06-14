"""
模型评估与可视化
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)
from tqdm import tqdm


@torch.no_grad()
def evaluate_on_test(model, test_loader, class_names, device,
                     checkpoint_path=None, save_dir='results/figures'):
    """
    在测试集上完整评估模型
    """
    # 加载最佳模型
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载最佳模型 (Val Acc: {checkpoint.get('val_acc', 'N/A')}%)")

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(test_loader, desc="测试集评估"):
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    num_classes = len(class_names)

    # ============================
    # 1. 整体指标
    # ============================
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    print(f"\n{'='*60}")
    print(f"测试集评估结果")
    print(f"{'='*60}")
    print(f"  总体准确率 (Accuracy):     {accuracy*100:.2f}%")
    print(f"  宏平均精确率 (Precision):  {precision*100:.2f}%")
    print(f"  宏平均召回率 (Recall):     {recall*100:.2f}%")
    print(f"  宏平均 F1-Score:           {f1*100:.2f}%")

    # ============================
    # 2. 分类报告
    # ============================
    short_names = [name.replace('___', ' ').replace('__', ' ')[:30] for name in class_names]

    print(f"\n{'─'*80}")
    print(f"详细分类报告")
    print(f"{'─'*80}")
    print(classification_report(
        all_labels, all_preds,
        target_names=short_names,
        digits=3,
        zero_division=0
    ))

    # ============================
    # 3. 各类别准确率柱状图
    # ============================
    per_class_acc = []
    for i in range(num_classes):
        mask = all_labels == i
        if mask.sum() > 0:
            acc = (all_preds[mask] == i).mean() * 100
        else:
            acc = 0
        per_class_acc.append(acc)

    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.RdYlGn([acc / 100 for acc in per_class_acc])
    bars = ax.barh(short_names, per_class_acc, color=colors, edgecolor='gray', linewidth=0.5)
    ax.axvline(x=accuracy * 100, color='blue', linestyle='--', linewidth=2,
               label=f'Overall: {accuracy*100:.1f}%')
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy on Test Set', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.legend(fontsize=11)

    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"各类别准确率图已保存")

    # ============================
    # 4. 混淆矩阵
    # ============================
    cm = confusion_matrix(all_labels, all_preds)

    # 归一化混淆矩阵（按行）
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    # 原始计数
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=short_names, yticklabels=short_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # 归一化
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=short_names, yticklabels=short_names,
                ax=axes[1], cbar_kws={'label': 'Normalized'})
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存")

    # ============================
    # 5. 最易混淆的类别对分析
    # ============================
    print(f"\n{'─'*60}")
    print(f"最易混淆的类别对 (Top 10)")
    print(f"{'─'*60}")

    errors = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i][j] > 0:
                errors.append({
                    'true': class_names[i],
                    'pred': class_names[j],
                    'count': cm[i][j],
                    'rate': cm[i][j] / cm[i].sum() if cm[i].sum() > 0 else 0
                })

    errors.sort(key=lambda x: x['count'], reverse=True)

    for i, err in enumerate(errors[:10]):
        print(f"  {i+1}. {err['true'][:30]} → {err['pred'][:30]}: "
              f"{err['count']} 次 ({err['rate']*100:.1f}%)")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_acc': per_class_acc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def analyze_misclassified(model, test_dataset, idx_to_class, device,
                          num_samples=20, save_dir='results/figures'):
    """展示并分析模型分类错误的样本"""
    model.eval()

    misclassified = []

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="收集错误样本"):
            image, true_label = test_dataset[i]
            image_batch = image.unsqueeze(0).to(device)

            output = model(image_batch)
            probs = F.softmax(output, dim=1)
            pred_label = output.argmax(1).item()
            confidence = probs[0][pred_label].item()

            if pred_label != true_label:
                misclassified.append({
                    'image': image,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': confidence,
                    'true_name': idx_to_class[true_label],
                    'pred_name': idx_to_class[pred_label]
                })

    if not misclassified:
        print("没有发现错误分类的样本！")
        return None

    # 按置信度排序（高置信度错误更值得关注）
    misclassified.sort(key=lambda x: x['confidence'], reverse=True)

    # 可视化前 num_samples 个错误样本
    n = min(num_samples, len(misclassified))
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for idx in range(n):
        sample = misclassified[idx]
        img = sample['image'].cpu().numpy().transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img, 0, 1)

        axes[idx].imshow(img)
        axes[idx].set_title(
            f"True: {sample['true_name'][:20]}\n"
            f"Pred: {sample['pred_name'][:20]}\n"
            f"Conf: {sample['confidence']:.2f}",
            fontsize=8
        )

        if sample['confidence'] > 0.8:
            for spine in axes[idx].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)

        axes[idx].axis('off')

    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'错误分类样本分析 (共 {len(misclassified)} 个错误)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'misclassified_samples.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n错误样本统计:")
    print(f"  总错误数: {len(misclassified)}")
    print(f"  高置信度错误(>80%): {sum(1 for m in misclassified if m['confidence'] > 0.8)}")
    print(f"  低置信度错误(<50%): {sum(1 for m in misclassified if m['confidence'] < 0.5)}")

    return misclassified
