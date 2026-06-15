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


def _tta_transforms(image_size=224):
    """
    生成 TTA 所需的图像变换列表。

    返回 (transforms_list, descriptions):
        transforms_list: 每个元素是一个 callable，输入 PIL Image，输出 tensor
        descriptions: 对应的描述字符串
    """
    from torchvision import transforms as T

    tta_transforms = []
    descs = []

    # 基础：resize 到 256，然后做不同裁剪
    base_size = int(image_size * 1.14)  # ≈256 for 224

    # 1. 中心裁剪（标准评估）
    tta_transforms.append(T.Compose([
        T.Resize((base_size, base_size)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    descs.append('center')

    # 2. 中心裁剪 + 水平翻转
    tta_transforms.append(T.Compose([
        T.Resize((base_size, base_size)),
        T.CenterCrop(image_size),
        T.RandomHorizontalFlip(p=1.0),  # 始终翻转
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    descs.append('center+hflip')

    # 3-6: 四角裁剪
    for corner_name, corner in [('topleft', (0, 0)),
                                  ('topright', (base_size - image_size, 0)),
                                  ('bottomleft', (0, base_size - image_size)),
                                  ('bottomright', (base_size - image_size, base_size - image_size))]:
        tta_transforms.append(T.Compose([
            T.Resize((base_size, base_size)),
            T.Lambda(lambda img, c=corner: img.crop(
                (c[0], c[1], c[0] + image_size, c[1] + image_size))),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
        descs.append(corner_name)

        # 每个角 + 水平翻转
        tta_transforms.append(T.Compose([
            T.Resize((base_size, base_size)),
            T.Lambda(lambda img, c=corner: img.crop(
                (c[0], c[1], c[0] + image_size, c[1] + image_size))),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
        descs.append(f'{corner_name}+hflip')

    return tta_transforms, descs


_SIMPLE_TTA_TRANSFORMS = None
_SIMPLE_TTA_DESCS = None


def _get_simple_tta_transforms(image_size=224):
    """Simple TTA: center crop + its horizontal flip (2 views)"""
    global _SIMPLE_TTA_TRANSFORMS, _SIMPLE_TTA_DESCS
    if _SIMPLE_TTA_TRANSFORMS is None or _SIMPLE_TTA_TRANSFORMS[0].transforms[1].size[0] != int(image_size * 1.14):
        from torchvision import transforms as T
        base_size = int(image_size * 1.14)
        _SIMPLE_TTA_TRANSFORMS = [
            T.Compose([
                T.Resize((base_size, base_size)),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            T.Compose([
                T.Resize((base_size, base_size)),
                T.CenterCrop(image_size),
                T.RandomHorizontalFlip(p=1.0),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        ]
        _SIMPLE_TTA_DESCS = ['center', 'center+hflip']
    return _SIMPLE_TTA_TRANSFORMS, _SIMPLE_TTA_DESCS


@torch.no_grad()
def evaluate_with_tta(model, test_dataset, class_names, device,
                      checkpoint_path=None, tta_mode='simple',
                      save_dir='results/figures'):
    """
    使用测试时增强 (TTA) 评估模型

    参数:
        model: 模型实例
        test_dataset: 测试集 Dataset（返回 PIL Image + label）
        class_names: 类别名称列表
        device: 设备
        checkpoint_path: 模型权重路径
        tta_mode: 'simple' (2 views) 或 'full' (10 views)
        save_dir: 图表保存目录
    """
    from torch.utils.data import DataLoader
    from PIL import Image

    # 加载最佳模型
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载最佳模型 (Val Acc: {checkpoint.get('val_acc', 'N/A')}%)")

    model.eval()

    # 获取 TTA 变换
    image_size = 224  # 默认
    if tta_mode == 'simple':
        tta_transforms, tta_descs = _get_simple_tta_transforms(image_size)
    else:
        tta_transforms, tta_descs = _tta_transforms(image_size)

    num_views = len(tta_transforms)
    print(f"\nTTA 模式: {tta_mode} ({num_views} 个视图)")
    print(f"  视图列表: {', '.join(tta_descs)}")

    all_preds = []
    all_labels = []
    all_probs = []

    # 对每个样本，聚合多个视图的预测
    for i in tqdm(range(len(test_dataset)), desc="TTA 评估"):
        # 获取原始 PIL 图像
        if hasattr(test_dataset, 'dataset') and hasattr(test_dataset, 'indices'):
            # Subset 类型
            img_path, true_label = test_dataset.dataset.samples[test_dataset.indices[i]]
        else:
            img_path, true_label = test_dataset.samples[i]

        try:
            pil_image = Image.open(img_path).convert('RGB')
        except Exception:
            pil_image = Image.new('RGB', (224, 224), (0, 0, 0))

        # 多视图推理
        batch_views = []
        for tta_transform in tta_transforms:
            view_tensor = tta_transform(pil_image)
            batch_views.append(view_tensor)

        batch = torch.stack(batch_views).to(device)

        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)

        # 平均概率
        avg_probs = probs.mean(dim=0)
        pred = avg_probs.argmax().item()

        all_preds.append(pred)
        all_labels.append(true_label)
        all_probs.append(avg_probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    num_classes = len(class_names)

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    print(f"\n{'='*60}")
    print(f"TTA 评估结果 ({tta_mode}, {num_views} views)")
    print(f"{'='*60}")
    print(f"  总体准确率 (Accuracy):     {accuracy*100:.2f}%")
    print(f"  宏平均精确率 (Precision):  {precision*100:.2f}%")
    print(f"  宏平均召回率 (Recall):     {recall*100:.2f}%")
    print(f"  宏平均 F1-Score:           {f1*100:.2f}%")

    # 各类别 F1
    per_class_precision, per_class_recall, per_class_f1, _ = \
        precision_recall_fscore_support(all_labels, all_preds, zero_division=0)

    short_names = [name.replace('___', ' ').replace('__', ' ')[:30] for name in class_names]

    print(f"\n{'─'*80}")
    print(f"各类别 TTA 评估详情")
    print(f"{'─'*80}")
    print(f"{'类别':<35s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print(f"{'─'*80}")
    for i in range(num_classes):
        mask = all_labels == i
        support = mask.sum()
        print(f"{short_names[i]:<35s} {per_class_precision[i]*100:>9.1f}% "
              f"{per_class_recall[i]*100:>9.1f}% {per_class_f1[i]*100:>9.1f}% {support:>10d}")

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    # 与标准评估对比
    print(f"\n{'─'*60}")
    print(f"TTA 混淆分析 (Top 10)")
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
    for rank, err in enumerate(errors[:10]):
        print(f"  {rank+1}. {err['true'][:30]} → {err['pred'][:30]}: "
              f"{err['count']} 次 ({err['rate']*100:.1f}%)")

    # 绘制 TTA 混淆矩阵
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=short_names, yticklabels=short_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title(f'TTA Confusion Matrix (Counts, {tta_mode})', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=short_names, yticklabels=short_names,
                ax=axes[1], cbar_kws={'label': 'Normalized'})
    axes[1].set_title(f'TTA Confusion Matrix (Normalized, {tta_mode})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_tta_{tta_mode}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"TTA 混淆矩阵已保存")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_f1': per_class_f1,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'tta_mode': tta_mode,
        'num_views': num_views,
    }
