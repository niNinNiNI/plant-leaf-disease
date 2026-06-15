"""
模型集成评估 — 平均多个模型的预测概率
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

from src.models import create_model
from src.dataset import split_dataset


@torch.no_grad()
def ensemble_evaluate(model_configs, test_dataset, class_names, device,
                       save_dir='results/figures'):
    """
    多模型集成评估：加载多个模型，平均概率后预测。

    参数:
        model_configs: list of dict, 每个包含:
            - model_type: str
            - backbone: str
            - checkpoint_path: str
            - dropout_rate: float (可选)
        test_dataset: 测试集 Dataset（返回 PIL Image / tensor + label）
        class_names: 类别名列表
        device: 设备
        save_dir: 图表保存目录

    返回:
        results dict
    """
    from torch.utils.data import DataLoader

    models = []
    model_names = []

    for cfg in model_configs:
        print(f"\n加载模型: {cfg['backbone']}")
        print(f"  路径: {cfg['checkpoint_path']}")

        model = create_model(
            model_type=cfg.get('model_type', 'transfer'),
            backbone=cfg['backbone'],
            num_classes=len(class_names),
            dropout_rate=cfg.get('dropout_rate', 0.5),
            freeze_backbone=False
        ).to(device)

        checkpoint = torch.load(cfg['checkpoint_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
        model_names.append(cfg.get('name', cfg['backbone']))
        print(f"  Val Acc: {checkpoint.get('val_acc', checkpoint.get('best_val_acc', 'N/A'))}%")

    # 使用 test_dataset 的底层 dataset 的 transform
    if hasattr(test_dataset, 'dataset') and hasattr(test_dataset, 'transform'):
        test_transform = test_dataset.dataset.transform
    elif hasattr(test_dataset, 'transform'):
        test_transform = test_dataset.transform
    else:
        from src.transforms import get_eval_transform
        test_transform = get_eval_transform()

    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"\n{'='*60}")
    print(f"集成推理 ({len(models)} 个模型)")
    print(f"{'='*60}")

    all_preds = []
    all_labels = []
    all_ensemble_probs = []
    all_individual_preds = {name: [] for name in model_names}

    for images, labels in tqdm(test_loader, desc="集成评估"):
        images = images.to(device)

        # 收集所有模型的概率
        ensemble_probs = torch.zeros(images.size(0), len(class_names), device=device)

        for i, model in enumerate(models):
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            ensemble_probs += probs

            # 记录每个模型的独立预测
            preds = outputs.argmax(1).cpu().numpy()
            all_individual_preds[model_names[i]].extend(preds)

        # 平均概率
        ensemble_probs /= len(models)
        preds = ensemble_probs.argmax(1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_ensemble_probs.extend(ensemble_probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_ensemble_probs = np.array(all_ensemble_probs)

    num_classes = len(class_names)
    short_names = [name.replace('___', ' ').replace('__', ' ')[:30] for name in class_names]

    # ============================
    # 1. 集成指标
    # ============================
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    print(f"\n{'='*60}")
    print(f"模型集成评估结果")
    print(f"{'='*60}")
    print(f"  参与模型: {', '.join(model_names)}")
    print(f"  总体准确率 (Accuracy):     {accuracy*100:.2f}%")
    print(f"  宏平均精确率 (Precision):  {precision*100:.2f}%")
    print(f"  宏平均召回率 (Recall):     {recall*100:.2f}%")
    print(f"  宏平均 F1-Score:           {f1*100:.2f}%")

    # ============================
    # 2. 各模型单独对比
    # ============================
    print(f"\n{'─'*60}")
    print(f"单模型 vs 集成 对比")
    print(f"{'─'*60}")
    for name in model_names:
        ind_acc = accuracy_score(all_labels, all_individual_preds[name])
        ind_f1 = precision_recall_fscore_support(
            all_labels, all_individual_preds[name], average='macro', zero_division=0
        )[2]
        print(f"  {name:<30s}  Acc: {ind_acc*100:.2f}%  F1: {ind_f1*100:.2f}%")
    print(f"  {'→ 集成 (平均概率)':<30s}  Acc: {accuracy*100:.2f}%  F1: {f1*100:.2f}%")

    # ============================
    # 3. 各类别详情
    # ============================
    per_class_precision, per_class_recall, per_class_f1, _ = \
        precision_recall_fscore_support(all_labels, all_preds, zero_division=0)

    print(f"\n{'─'*80}")
    print(f"各类别集成评估详情")
    print(f"{'─'*80}")
    print(f"{'类别':<35s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print(f"{'─'*80}")
    for i in range(num_classes):
        mask = all_labels == i
        support = mask.sum()
        print(f"{short_names[i]:<35s} {per_class_precision[i]*100:>9.1f}% "
              f"{per_class_recall[i]*100:>9.1f}% {per_class_f1[i]*100:>9.1f}% {support:>10d}")

    # ============================
    # 4. 分类报告
    # ============================
    print(f"\n{'─'*80}")
    print(f"详细分类报告")
    print(f"{'─'*80}")
    print(classification_report(
        all_labels, all_preds, target_names=short_names, digits=3, zero_division=0
    ))

    # ============================
    # 5. 混淆矩阵
    # ============================
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=short_names, yticklabels=short_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Ensemble Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=short_names, yticklabels=short_names,
                ax=axes[1], cbar_kws={'label': 'Normalized'})
    axes[1].set_title('Ensemble Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_ensemble.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"集成混淆矩阵已保存")

    # ============================
    # 6. 模型间一致性分析
    # ============================
    if len(models) >= 2:
        print(f"\n{'─'*60}")
        print(f"模型间预测一致性")
        print(f"{'─'*60}")
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                preds_i = np.array(all_individual_preds[model_names[i]])
                preds_j = np.array(all_individual_preds[model_names[j]])
                agreement = (preds_i == preds_j).mean() * 100
                print(f"  {model_names[i]} vs {model_names[j]}: {agreement:.1f}% 一致")

        # 找出集成修正单模型错误的样本
        ensemble_correct = all_preds == all_labels
        for name in model_names:
            single_correct = np.array(all_individual_preds[name]) == all_labels
            fixed = (~single_correct) & ensemble_correct
            broken = single_correct & (~ensemble_correct)
            print(f"  集成修正了 {name} 的 {fixed.sum()} 个错误，引入了 {broken.sum()} 个新错误")

    # ============================
    # 7. 错误样本分析
    # ============================
    print(f"\n{'─'*60}")
    print(f"Top 10 混淆对（集成）")
    print(f"{'─'*60}")
    errors = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i][j] > 0:
                errors.append({
                    'true': class_names[i],
                    'pred': class_names[j],
                    'count': int(cm[i][j]),
                    'rate': float(cm[i][j] / cm[i].sum()) if cm[i].sum() > 0 else 0
                })
    errors.sort(key=lambda x: x['count'], reverse=True)
    for rank, err in enumerate(errors[:10]):
        print(f"  {rank+1}. {err['true'][:30]} → {err['pred'][:30]}: "
              f"{err['count']} 次 ({err['rate']*100:.1f}%)")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_f1': per_class_f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_ensemble_probs,
        'model_names': model_names,
        'individual_predictions': all_individual_preds,
    }
