#!/usr/bin/env python3
"""
农业植物叶片病害识别 — 主训练脚本

用法:
    # 标准训练（迁移学习 ResNet-50）
    python main.py --model transfer --backbone resnet50 --epochs 80

    # 仅评估
    python main.py --eval_only --checkpoint checkpoints/transfer_resnet50/best_model.pth

    # 生成 Grad-CAM 可视化
    python main.py --gradcam --checkpoint checkpoints/transfer_resnet50/best_model.pth

    # 运行消融实验
    python main.py --ablation

    # 从零训练 BasicCNN
    python main.py --model basic --epochs 50 --dropout 0.5
"""
import os
import sys
import argparse

# 确保 src 包可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import Config
from src.utils import set_seed, get_device, count_parameters, plot_training_curves
from src.dataset import split_dataset, compute_class_weights, create_dataloaders
from src.models import create_model
from src.losses import get_criterion
from src.trainer import Trainer, get_scheduler
from src.evaluate import evaluate_on_test, analyze_misclassified
from src.gradcam import visualize_gradcam
from src.ablation import run_ablation_study, visualize_ablation_results


def parse_args():
    parser = argparse.ArgumentParser(description='植物叶片病害识别')

    # 模型选择
    parser.add_argument('--model', type=str, default='transfer',
                        choices=['basic', 'improved', 'transfer'],
                        help='模型类型 (default: transfer)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='迁移学习骨干网络 (default: resnet50)')

    # 超参数
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数 (default: 80 for transfer, 50 for others)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.5)

    # 训练策略
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='冻结骨干网络 (default: True)')
    parser.add_argument('--unfreeze_backbone', action='store_true',
                        help='解冻骨干网络进行微调')
    parser.add_argument('--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal', 'label_smoothing'])
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'step', 'onecycle'])

    # 模式
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估（不训练）')
    parser.add_argument('--gradcam', action='store_true',
                        help='生成 Grad-CAM 可视化')
    parser.add_argument('--ablation', action='store_true',
                        help='运行消融实验')
    parser.add_argument('--finetune', action='store_true',
                        help='运行二阶段微调')

    # 路径
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='checkpoint 路径（用于评估/继续训练）')
    parser.add_argument('--data_root', type=str, default=None,
                        help='数据集根目录')

    # 其他
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # 默认 epochs
    if args.epochs is None:
        if args.model == 'transfer':
            args.epochs = 80
        else:
            args.epochs = 50

    return args


def main():
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 获取设备
    device = get_device()

    # 数据集路径
    data_root = args.data_root or Config.DATA_ROOT
    if not os.path.exists(data_root):
        print(f"错误: 数据集路径不存在: {data_root}")
        print("请使用 --data_root 指定正确的路径")
        sys.exit(1)

    # ============================================================
    # 模式: 消融实验
    # ============================================================
    if args.ablation:
        print("=" * 60)
        print("启动消融实验")
        print("=" * 60)
        experiments = run_ablation_study(data_root, device)
        visualize_ablation_results(experiments)
        print("消融实验完成!")
        return

    # ============================================================
    # 数据加载
    # ============================================================
    print("\n" + "=" * 60)
    print("加载数据")
    print("=" * 60)

    train_dataset, val_dataset, test_dataset, class_to_idx, eval_data = split_dataset(
        data_root,
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO,
        test_ratio=Config.TEST_RATIO,
        seed=args.seed,
        image_size=Config.IMAGE_SIZE
    )

    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = list(class_to_idx.keys())

    class_weights = compute_class_weights(train_dataset, num_classes)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        num_classes=num_classes,
        batch_size=args.batch_size,
        num_workers=Config.NUM_WORKERS
    )

    # ============================================================
    # 模型创建
    # ============================================================
    if not args.eval_only or not args.checkpoint:
        print("\n" + "=" * 60)
        print(f"创建模型: {args.model} (backbone={args.backbone})")
        print("=" * 60)

        freeze = args.freeze_backbone and not args.unfreeze_backbone
        model = create_model(
            model_type=args.model,
            backbone=args.backbone,
            num_classes=num_classes,
            dropout_rate=args.dropout,
            freeze_backbone=freeze
        )
        count_parameters(model)
    else:
        model = create_model(
            model_type=args.model,
            backbone=args.backbone,
            num_classes=num_classes,
            dropout_rate=args.dropout,
            freeze_backbone=False
        )

    # ============================================================
    # 仅评估模式
    # ============================================================
    if args.eval_only:
        print("\n" + "=" * 60)
        print("评估模式")
        print("=" * 60)
        checkpoint_path = args.checkpoint or f'checkpoints/{args.model}_{args.backbone}/best_model.pth'
        model = model.to(device)
        results = evaluate_on_test(
            model, test_loader, class_names, device,
            checkpoint_path=checkpoint_path
        )
        analyze_misclassified(model, test_dataset, idx_to_class, device)
        return

    # ============================================================
    # Grad-CAM 模式
    # ============================================================
    if args.gradcam:
        print("\n" + "=" * 60)
        print("Grad-CAM 可视化")
        print("=" * 60)
        checkpoint_path = args.checkpoint or f'checkpoints/{args.model}_{args.backbone}/best_model.pth'
        model = model.to(device)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"已加载模型: {checkpoint_path}")
        visualize_gradcam(
            model, test_dataset, device, class_names,
            model_type=args.model, backbone=args.backbone,
            num_samples=9
        )
        return

    # ============================================================
    # 训练
    # ============================================================
    print("\n" + "=" * 60)
    print("准备训练")
    print("=" * 60)

    # 损失函数
    criterion = get_criterion(args.loss, class_weights=class_weights.to(device))
    print(f"损失函数: {args.loss}")

    # 优化器
    if args.model == 'transfer' and not args.freeze_backbone:
        # 分类头使用较高学习率，骨干使用较低学习率
        optimizer = optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': args.lr * 0.1},
            {'params': model.backbone.fc.parameters(), 'lr': args.lr}
        ], weight_decay=Config.WEIGHT_DECAY)
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY
        )
    print(f"优化器: AdamW (lr={args.lr})")

    # 学习率调度器
    scheduler = get_scheduler(
        optimizer, args.scheduler,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        lr=args.lr
    )
    print(f"调度器: {args.scheduler}")

    # 创建训练器
    save_dir = f'checkpoints/{args.model}_{args.backbone}'
    log_dir = f'results/logs/{args.model}_{args.backbone}'
    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        use_amp=Config.USE_AMP,
        early_stop_patience=Config.EARLY_STOP_PATIENCE,
        save_dir=save_dir,
        log_dir=log_dir,
        grad_clip_max_norm=Config.GRAD_CLIP_MAX_NORM
    )

    # ---- 阶段一: 冻结骨干训练 ----
    print(f"\n{'='*60}")
    print(f"阶段一: 训练分类头 (冻结骨干, {args.epochs} epochs)")
    print(f"{'='*60}")
    history = trainer.train(train_loader, val_loader, epochs=args.epochs)

    # 绘制训练曲线
    plot_training_curves(
        history,
        save_path=f'results/figures/training_curves_{args.model}_{args.backbone}.png'
    )

    # ---- 阶段二: 微调（可选） ----
    if args.finetune and args.model == 'transfer':
        print(f"\n{'='*60}")
        print("阶段二: 微调 (解冻骨干最后几层)")
        print(f"{'='*60}")

        model.unfreeze_backbone(num_layers_to_unfreeze=30)

        trainer.optimizer = optim.AdamW(
            model.parameters(), lr=Config.FINETUNE_LR, weight_decay=Config.WEIGHT_DECAY
        )
        trainer.scheduler = CosineAnnealingLR(
            trainer.optimizer, T_max=Config.FINETUNE_EPOCHS, eta_min=1e-6
        )
        trainer.early_stop_patience = 10

        history_finetune = trainer.train(
            train_loader, val_loader, epochs=Config.FINETUNE_EPOCHS
        )

        plot_training_curves(
            history_finetune,
            save_path=f'results/figures/training_curves_finetune_{args.model}_{args.backbone}.png'
        )

    # ============================================================
    # 测试集评估
    # ============================================================
    print(f"\n{'='*60}")
    print("测试集评估")
    print(f"{'='*60}")

    best_checkpoint = f'{save_dir}/best_model.pth'
    results = evaluate_on_test(
        model, test_loader, class_names, device,
        checkpoint_path=best_checkpoint
    )

    # 错误样本分析
    analyze_misclassified(model, test_dataset, idx_to_class, device)

    # ============================================================
    # Grad-CAM 可视化
    # ============================================================
    print(f"\n{'='*60}")
    print("Grad-CAM 可视化")
    print(f"{'='*60}")
    visualize_gradcam(
        model, test_dataset, device, class_names,
        model_type=args.model, backbone=args.backbone,
        num_samples=9
    )

    # ============================================================
    # 保存评估指标
    # ============================================================
    import json
    metrics_path = 'results/metrics.json'
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    metrics = {
        'model': f'{args.model}_{args.backbone}',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'dropout': args.dropout,
        'loss_function': args.loss,
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1': float(results['f1']),
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n评估指标已保存到: {metrics_path}")

    print(f"\n{'='*60}")
    print("全部完成!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
