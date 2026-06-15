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
from src.evaluate import evaluate_on_test, analyze_misclassified, evaluate_with_tta
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
    parser.add_argument('--mixup', action='store_true',
                        help='使用 MixUp 数据增强')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='MixUp Beta 分布参数 (default: 0.2)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='输入图像尺寸, 推荐224/384 (default: 224)')
    parser.add_argument('--unfreeze_strategy', type=str, default='auto',
                        choices=['auto', 'last_n_params', 'stages'],
                        help='骨干解冻策略: auto=自动选择, last_n_params=参数级, stages=阶段级')
    parser.add_argument('--unfreeze_stages', type=int, default=2,
                        help='解冻最后几个 stage (仅 stages 策略有效, default: 2)')

    # 模式
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估（不训练）')
    parser.add_argument('--tta', action='store_true',
                        help='使用测试时增强 (TTA) 评估')
    parser.add_argument('--tta_mode', type=str, default='simple',
                        choices=['simple', 'full'],
                        help='TTA 模式: simple=2视图, full=10视图 (default: simple)')
    parser.add_argument('--gradcam', action='store_true',
                        help='生成 Grad-CAM 可视化')
    parser.add_argument('--ablation', action='store_true',
                        help='运行消融实验')
    parser.add_argument('--finetune', action='store_true',
                        help='运行二阶段微调')
    parser.add_argument('--resume', action='store_true',
                        help='从 checkpoint 恢复训练（断点续训）')
    parser.add_argument('--skip_phase1', action='store_true',
                        help='跳过阶段一，直接进入微调阶段（需配合 --resume 和 --finetune）')

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
        image_size=args.image_size
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
        print("评估模式" + (" (含 TTA)" if args.tta else ""))
        print("=" * 60)
        checkpoint_path = args.checkpoint or f'checkpoints/{args.model}_{args.backbone}/best_model.pth'
        model = model.to(device)

        if args.tta:
            # TTA 评估
            results = evaluate_with_tta(
                model, test_dataset, class_names, device,
                checkpoint_path=checkpoint_path,
                tta_mode=args.tta_mode
            )
        else:
            # 标准评估
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

    # 创建训练器 — 构建唯一实验标签
    loss_suffix = f'_{args.loss}' if args.loss != 'cross_entropy' else ''
    mixup_suffix = '_mixup' if args.mixup else ''
    size_suffix = f'_{args.image_size}' if args.image_size != 224 else ''
    # 非默认解冻策略也纳入标签，避免新旧日志混淆
    unfreeze_suffix = ''
    if args.unfreeze_strategy != 'auto' or args.unfreeze_stages != 2:
        unfreeze_suffix = f'_uf{args.unfreeze_strategy}{args.unfreeze_stages}'
    experiment_tag = f'{loss_suffix}{mixup_suffix}{size_suffix}{unfreeze_suffix}'
    save_dir = f'checkpoints/{args.model}_{args.backbone}{experiment_tag}'
    log_dir = f'results/logs/{args.model}_{args.backbone}{experiment_tag}'
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
        grad_clip_max_norm=Config.GRAD_CLIP_MAX_NORM,
        use_mixup=args.mixup,
        mixup_alpha=args.mixup_alpha
    )

    # ---- 断点续训 ----
    if args.resume:
        resume_path = args.checkpoint or os.path.join(save_dir, 'best_model.pth')
        if not os.path.exists(resume_path):
            print(f"错误: 找不到 checkpoint: {resume_path}")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"从 checkpoint 恢复训练")
        print(f"{'='*60}")
        print(f"  路径: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        resumed_epoch = checkpoint.get('epoch', 0)
        trainer.start_epoch = resumed_epoch + 1
        trainer.best_val_acc = checkpoint.get('best_val_acc', 0)
        trainer.best_epoch = checkpoint.get('epoch', 0)
        trainer.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)

        # 恢复训练历史（确保是 dict 类型）
        saved_history = checkpoint.get('history', None)
        if saved_history is not None and isinstance(saved_history, dict):
            trainer.history = saved_history

        print(f"  已恢复 epoch: {resumed_epoch}")
        print(f"  最佳 Val Acc: {trainer.best_val_acc:.2f}%")
        print(f"  将从 epoch {trainer.start_epoch} 继续训练")

    # ---- 判断是否跳过阶段一 ----
    skip_phase1 = args.skip_phase1 or (args.resume and args.finetune)

    if skip_phase1:
        print(f"\n{'='*60}")
        print(f"跳过阶段一，直接进入微调阶段")
        print(f"{'='*60}")
        # 从 checkpoint 加载最佳模型权重（如果尚未通过 resume 加载）
        resume_path = args.checkpoint or os.path.join(save_dir, 'best_model.pth')
        if not args.resume and os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            best_val_acc = checkpoint.get('val_acc', checkpoint.get('best_val_acc', 0))
            best_epoch = checkpoint.get('epoch', 0)
            print(f"  已加载 checkpoint: {resume_path}")
            print(f"  阶段一最佳 Val Acc: {best_val_acc:.2f}% (epoch {best_epoch})")
        elif args.resume:
            print(f"  已通过 --resume 加载模型，阶段一最佳 Val Acc: {trainer.best_val_acc:.2f}%")
        elif not os.path.exists(resume_path):
            print(f"  警告: 未找到 checkpoint ({resume_path})，将从随机初始化继续")
    else:
        # ---- 阶段一: 冻结骨干训练 ----
        print(f"\n{'='*60}")
        print(f"阶段一: 训练分类头 (冻结骨干, {args.epochs} epochs)")
        print(f"{'='*60}")
        history = trainer.train(train_loader, val_loader, epochs=args.epochs)

        # 绘制训练曲线
        plot_training_curves(
            history,
            save_path=f'results/figures/training_curves_{args.model}_{args.backbone}{experiment_tag}.png'
        )

    # ---- 阶段二: 微调（可选） ----
    if args.finetune and args.model == 'transfer':
        print(f"\n{'='*60}")
        print("阶段二: 微调 (解冻骨干最后几层)")
        print(f"{'='*60}")

        # 智能解冻: auto 策略对 ConvNeXt 使用 stage 级解冻, 对 ResNet 使用参数级
        if args.unfreeze_strategy == 'stages':
            model.unfreeze_backbone(
                num_layers_to_unfreeze=args.unfreeze_stages,
                strategy='stages'
            )
        elif args.unfreeze_strategy == 'last_n_params':
            model.unfreeze_backbone(
                num_layers_to_unfreeze=30,
                strategy='last_n_params'
            )
        else:  # auto
            model.unfreeze_backbone(
                num_layers_to_unfreeze=args.unfreeze_stages
                if args.backbone.startswith('convnext') else 30,
                strategy='auto'
            )

        # 只优化可训练参数
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        trainer.optimizer = optim.AdamW(
            trainable_params, lr=Config.FINETUNE_LR, weight_decay=Config.WEIGHT_DECAY
        )
        trainer.scheduler = CosineAnnealingLR(
            trainer.optimizer, T_max=Config.FINETUNE_EPOCHS, eta_min=1e-6
        )
        # 重置早停相关状态
        trainer.early_stop_patience = 10
        trainer.best_val_acc = 0.0
        trainer.best_epoch = 0
        trainer.epochs_without_improvement = 0
        trainer.start_epoch = 1
        trainer.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }

        history_finetune = trainer.train(
            train_loader, val_loader, epochs=Config.FINETUNE_EPOCHS
        )

        plot_training_curves(
            history_finetune,
            save_path=f'results/figures/training_curves_finetune_{args.model}_{args.backbone}{experiment_tag}.png'
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

    # ---- 准备评估指标（供 TTA 追加） ----
    import json
    metrics = {
        'model': f'{args.model}_{args.backbone}',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'dropout': args.dropout,
        'loss_function': args.loss,
        'image_size': args.image_size,
        'mixup': args.mixup,
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1': float(results['f1']),
    }

    # ---- TTA 评估（可选） ----
    if args.tta:
        print(f"\n{'='*60}")
        print(f"TTA 评估 (模式: {args.tta_mode})")
        print(f"{'='*60}")
        tta_results = evaluate_with_tta(
            model, test_dataset, class_names, device,
            checkpoint_path=best_checkpoint,
            tta_mode=args.tta_mode
        )

        # 在 metrics 中追加 TTA 指标
        metrics['tta_accuracy'] = float(tta_results['accuracy'])
        metrics['tta_precision'] = float(tta_results['precision'])
        metrics['tta_recall'] = float(tta_results['recall'])
        metrics['tta_f1'] = float(tta_results['f1'])
        metrics['tta_mode'] = args.tta_mode
        metrics['tta_num_views'] = tta_results['num_views']
        print(f"\n  TTA 准确率: {tta_results['accuracy']*100:.2f}%")
        print(f"  TTA F1:     {tta_results['f1']*100:.2f}%")

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
    metrics_path = f'results/metrics{experiment_tag}.json'
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n评估指标已保存到: {metrics_path}")

    print(f"\n{'='*60}")
    print("全部完成!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
