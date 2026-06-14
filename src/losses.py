"""
损失函数定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss — 解决类别不平衡和难易样本不平衡

    公式: FL = -α(1-p_t)^γ * log(p_t)

    参数:
        alpha: 类别权重，shape=[num_classes]
        gamma: 聚焦参数，越大越关注难分类样本。推荐: 2.0
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_criterion(loss_type='cross_entropy', class_weights=None):
    """
    获取损失函数

    参数:
        loss_type: 'cross_entropy' | 'focal' | 'label_smoothing'
        class_weights: 类别权重张量
    """
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == 'focal':
        return FocalLoss(alpha=class_weights, gamma=2.0)
    elif loss_type == 'label_smoothing':
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    else:
        raise ValueError(f"未知损失函数类型: {loss_type}")
