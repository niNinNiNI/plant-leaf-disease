"""
训练器：含训练循环、早停、AMP、TensorBoard、checkpoint 管理
"""
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, CosineAnnealingLR, StepLR, OneCycleLR
)
import numpy as np
from tqdm import tqdm


def mixup_data(x, y, alpha=0.2):
    """
    MixUp 数据增强: 随机混合两个 batch 中的样本

    公式:
        λ ~ Beta(α, α)
        mixed_x = λ * x + (1-λ) * x[shuffled]
        mixed_y = λ * y_onehot + (1-λ) * y_onehot[shuffled]

    返回:
        mixed_x, y_a, y_b, lam
        损失计算: loss = lam * criterion(output, y_a) + (1-lam) * criterion(output, y_b)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp 损失: 对两个标签分别计算损失的加权和"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_scheduler(optimizer, scheduler_type='cosine', **kwargs):
    """获取学习率调度器"""
    if scheduler_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('epochs', 50),
            eta_min=kwargs.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min', factor=0.5, patience=5,
            min_lr=1e-6, verbose=True
        )
    elif scheduler_type == 'step':
        return StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 15),
            gamma=0.1
        )
    elif scheduler_type == 'onecycle':
        steps_per_epoch = kwargs.get('steps_per_epoch', 100)
        epochs = kwargs.get('epochs', 50)
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get('lr', 0.001),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=0.3
        )
    else:
        return None


class Trainer:
    """
    完整的训练器，包含:
    - 训练/验证循环
    - 早停 (Early Stopping)
    - 模型保存
    - TensorBoard 日志
    - 学习率调度
    - 混合精度训练 (AMP)
    """

    def __init__(self, model, device,
                 criterion=None, optimizer=None, scheduler=None,
                 use_amp=True,
                 early_stop_patience=15,
                 save_dir='./checkpoints',
                 log_dir='./logs',
                 grad_clip_max_norm=1.0,
                 use_mixup=False,
                 mixup_alpha=0.2,
                 start_epoch=1):

        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.use_amp = use_amp and device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.early_stop_patience = early_stop_patience
        self.grad_clip_max_norm = grad_clip_max_norm
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # MixUp 数据增强
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        # TensorBoard
        self.writer = SummaryWriter(log_dir)

        # 训练历史
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.epoch_times = []
        self.start_epoch = start_epoch

    def train_epoch(self, train_loader, epoch):
        """训练一个 epoch"""
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d} [Train]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # MixUp 数据增强
            if self.use_mixup:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, alpha=self.mixup_alpha
                )

            # 混合精度训练
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(images)
                if self.use_mixup:
                    loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_norm=self.grad_clip_max_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # OneCycleLR 需要每个 batch 更新
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            # 统计 (MixUp 模式下用原始 labels 的预测准确率作为粗略参考)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            if self.use_mixup:
                # 用 lam 加权估计正确数
                correct += (lam * predicted.eq(labels_a).float().sum()
                            + (1 - lam) * predicted.eq(labels_b).float().sum()).item()
            else:
                correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.1f}%'
            })

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self, val_loader, epoch):
        """验证"""
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []
        all_probs = []

        pbar = tqdm(val_loader, desc=f"Epoch {epoch:3d} [Val  ]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            probs = F.softmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.1f}%'
            })

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc, all_preds, all_labels, all_probs

    def train(self, train_loader, val_loader, epochs=50):
        """完整训练流程"""
        resume_msg = f" (从 epoch {self.start_epoch} 恢复)" if self.start_epoch > 1 else ""
        print(f"\n{'='*60}")
        print(f"开始训练 ({epochs} epochs){resume_msg}")
        print(f"{'='*60}")
        print(f"  设备: {self.device}")
        print(f"  优化器: {type(self.optimizer).__name__}")
        print(f"  混合精度: {self.use_amp}")
        print(f"  MixUp: {self.use_mixup}" + (f" (alpha={self.mixup_alpha})" if self.use_mixup else ""))
        print(f"  早停耐心值: {self.early_stop_patience}")
        if self.start_epoch > 1:
            print(f"  当前最佳 Val Acc: {self.best_val_acc:.2f}% (epoch {self.best_epoch})")

        for epoch in range(self.start_epoch, epochs + 1):
            epoch_start = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # 验证
            val_loss, val_acc, val_preds, val_labels, val_probs = \
                self.validate(val_loader, epoch)

            # 学习率调度（epoch 级别）
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif not isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            # TensorBoard 记录
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)

            # 打印结果
            print(f"\nEpoch {epoch:3d}/{epochs} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"LR: {current_lr:.2e}")
            print(f"  Train — Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   — Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            # 早停检查
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_without_improvement = 0

                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f"  ✓ 最佳模型已保存 (Val Acc: {val_acc:.2f}%)")
            else:
                self.epochs_without_improvement += 1
                improvement = self.best_val_acc - val_acc
                print(f"  — 未改进 ({self.epochs_without_improvement}/{self.early_stop_patience}), "
                      f"距最佳: -{improvement:.2f}%")

            # 早停
            if self.epochs_without_improvement >= self.early_stop_patience:
                print(f"\n{'='*60}")
                print(f"早停触发! 最佳验证准确率: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
                print(f"{'='*60}")
                break

            # 定期保存
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_acc, is_best=False)

        self.writer.close()

        # 输出训练总结
        print(f"\n{'='*60}")
        print(f"训练完成总结")
        print(f"{'='*60}")
        print(f"  最佳 Epoch: {self.best_epoch}")
        print(f"  最佳验证准确率: {self.best_val_acc:.2f}%")
        print(f"  总 Epoch 数: {epoch}")

        return self.history

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'epochs_without_improvement': self.epochs_without_improvement
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)

        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        self.history = checkpoint.get('history', self.history)
        return checkpoint.get('epoch', 0)
